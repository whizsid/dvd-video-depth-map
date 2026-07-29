"""Per-frame depth noise scan + edge-preserving removal (GPU/CPU).

Scans each frame with a resolution-scaled kernel (local mean), flags outliers
against the frame depth range, snaps them, then polishes with a self-guided
bilateral. Parallel frame dispatch mirrors ``ParallelUpscaler``: thread pool
on CPU, single worker on CUDA/MPS.
"""

from __future__ import annotations

import math
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from upsample import FrameMemmapStore


@dataclass(frozen=True)
class DenoiseParams:
    """Kernel + thresholds for spatial depth denoise."""

    kernel_size: int = 5
    noise_sigma: float = 2.5
    sigma_spatial: float = 1.5
    sigma_range: float = 0.06
    strength: float = 1.0


def default_denoise_params(height: int, width: int) -> DenoiseParams:
    """Pick an odd kernel from the shorter video side (~0.35%, clamped 3–15)."""
    short = max(1, min(int(height), int(width)))
    k = int(round(short * 0.0035))
    if k % 2 == 0:
        k += 1
    k = max(3, min(15, k))
    return DenoiseParams(
        kernel_size=k,
        noise_sigma=2.5,
        sigma_spatial=max(1.0, k / 3.0),
        sigma_range=0.06,
        strength=1.0,
    )


def _odd_kernel(k: int) -> int:
    k = max(3, int(k))
    if k % 2 == 0:
        k += 1
    return k


def _depth_range(disp: np.ndarray) -> float:
    finite = disp[np.isfinite(disp)]
    if finite.size < 64:
        return 1.0
    lo, hi = np.percentile(finite, [2.0, 98.0])
    return float(max(hi - lo, 1e-6))


def _bilateral_self(
    disp: np.ndarray,
    radius: int,
    sigma_spatial: float,
    sigma_range: float,
) -> np.ndarray:
    """Self-guided bilateral (CPU), same accumulate pattern as JBU."""
    h, w = disp.shape[:2]
    disp = disp.astype(np.float32, copy=False)
    acc = np.zeros((h, w), dtype=np.float32)
    acc_w = np.zeros((h, w), dtype=np.float32)
    inv_2ss = 1.0 / (2.0 * sigma_spatial * sigma_spatial)
    inv_2sr = 1.0 / (2.0 * sigma_range * sigma_range)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.roll(disp, shift=(dy, dx), axis=(0, 1))
            w_spatial = math.exp(-(dx * dx + dy * dy) * inv_2ss)
            diff = disp - shifted
            w = w_spatial * np.exp(-(diff * diff) * inv_2sr)
            acc += w * shifted
            acc_w += w
    return (acc / np.maximum(acc_w, 1e-8)).astype(np.float32)


def _bilateral_self_torch(
    disp: np.ndarray,
    radius: int,
    sigma_spatial: float,
    sigma_range: float,
    *,
    device: str,
) -> np.ndarray:
    """Self-guided bilateral on CUDA or MPS."""
    import torch

    dev = torch.device(device)
    d = torch.from_numpy(np.ascontiguousarray(disp, dtype=np.float32)).to(dev)
    h, w = d.shape
    acc = torch.zeros((h, w), device=dev, dtype=torch.float32)
    acc_w = torch.zeros((h, w), device=dev, dtype=torch.float32)
    inv_2ss = 1.0 / (2.0 * sigma_spatial * sigma_spatial)
    inv_2sr = 1.0 / (2.0 * sigma_range * sigma_range)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = torch.roll(d, shifts=(dy, dx), dims=(0, 1))
            w_spatial = math.exp(-(dx * dx + dy * dy) * inv_2ss)
            diff = d - shifted
            weight = w_spatial * torch.exp(-(diff * diff) * inv_2sr)
            acc = acc + weight * shifted
            acc_w = acc_w + weight
    out = acc / torch.clamp(acc_w, min=1e-8)
    return out.detach().float().cpu().numpy().astype(np.float32, copy=False)


def _box_blur(disp: np.ndarray, kernel: int) -> np.ndarray:
    """Reflect-padded box mean via integral image (no OpenCV)."""
    k = _odd_kernel(kernel)
    pad = k // 2
    x = np.pad(disp.astype(np.float64, copy=False), ((pad, pad), (pad, pad)), mode="reflect")
    # Inclusive integral: ii[y+1,x+1] = sum of x[0:y+1, 0:x+1]
    ii = np.pad(x, ((1, 0), (1, 0)), mode="constant")
    np.cumsum(ii, axis=0, out=ii)
    np.cumsum(ii, axis=1, out=ii)
    h, w = disp.shape
    # Window sum for top-left (i,j) covering [i:i+k, j:j+k] in padded x
    # → ii[i+k, j+k] - ii[i, j+k] - ii[i+k, j] + ii[i, j]
    s = (
        ii[k : k + h, k : k + w]
        - ii[0:h, k : k + w]
        - ii[k : k + h, 0:w]
        + ii[0:h, 0:w]
    )
    return (s / float(k * k)).astype(np.float32)


def _local_mean_std_cpu(disp: np.ndarray, kernel: int) -> tuple[np.ndarray, np.ndarray]:
    disp32 = disp.astype(np.float32, copy=False)
    mean = _box_blur(disp32, kernel)
    mean_sq = _box_blur(disp32 * disp32, kernel)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def _local_mean_std_torch(
    disp: np.ndarray, kernel: int, *, device: str
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as F

    k = _odd_kernel(kernel)
    pad = k // 2
    dev = torch.device(device)
    x = torch.from_numpy(np.ascontiguousarray(disp, dtype=np.float32)).to(dev)
    xb = x.view(1, 1, *x.shape)
    xb_pad = F.pad(xb, (pad, pad, pad, pad), mode="reflect")
    mean = F.avg_pool2d(xb_pad, kernel_size=k, stride=1)
    mean_sq = F.avg_pool2d(xb_pad * xb_pad, kernel_size=k, stride=1)
    var = torch.clamp(mean_sq - mean * mean, min=0.0)
    std = torch.sqrt(var)
    return (
        mean.view(*x.shape).detach().float().cpu().numpy().astype(np.float32, copy=False),
        std.view(*x.shape).detach().float().cpu().numpy().astype(np.float32, copy=False),
    )


def _resolve_torch_device(device: str) -> str | None:
    """Return a usable torch device string, or None to fall back to CPU."""
    device = str(device or "cpu").lower()
    if device in ("", "cpu"):
        return None
    try:
        import torch
    except Exception:
        return None
    if device.startswith("cuda"):
        return device if torch.cuda.is_available() else None
    if device.startswith("mps"):
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return None
    return None


def denoise_frame(
    disp: np.ndarray,
    params: DenoiseParams,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """
    Scan ``disp`` with a local kernel, flag noise, replace with bilateral smooth.

    Outliers are pixels whose deviation from the local mean exceeds a fraction of
    the frame depth range (local std alone is unreliable on impulse spikes).
    Flagged pixels are first snapped toward the local mean, then polished with a
    self-guided bilateral so edges stay sharp.

    ``device``: ``\"cpu\"``, ``\"cuda\"``, or ``\"mps\"``.
    """
    if disp.ndim != 2:
        raise ValueError(f"denoise_frame expects [H,W], got {disp.shape}")
    src = np.asarray(disp, dtype=np.float32)
    finite = np.isfinite(src)
    if int(finite.sum()) < 64:
        return src
    work = np.where(finite, src, 0.0).astype(np.float32, copy=False)

    k = _odd_kernel(params.kernel_size)
    radius = max(1, k // 2)
    depth_rng = _depth_range(work)
    sigma_range = max(1e-6, float(params.sigma_range) * depth_rng)
    sigma_spatial = max(1.0, float(params.sigma_spatial))
    strength = float(np.clip(params.strength, 0.0, 1.0))
    # Absolute + robust local gate: ignore std inflated by the spike itself.
    abs_thresh = float(params.noise_sigma) * 0.012 * depth_rng
    abs_thresh = max(abs_thresh, 1e-4 * depth_rng)

    torch_dev = _resolve_torch_device(device)
    if torch_dev is not None:
        mean, std = _local_mean_std_torch(work, k, device=torch_dev)
    else:
        mean, std = _local_mean_std_cpu(work, k)

    # Exclude the center sample so impulse spikes do not bias the reference.
    n = float(k * k)
    mean_excl = ((mean * n) - work) / max(n - 1.0, 1.0)
    residual = np.abs(work - mean_excl)
    local_thresh = float(params.noise_sigma) * np.maximum(std, abs_thresh)
    # Require absolute depth-range gate so impulse noise cannot hide in its own std.
    mask = (residual > abs_thresh) & (residual > 0.5 * local_thresh) & finite
    if not np.any(mask) or strength <= 0.0:
        return src

    # Snap outliers to neighbor mean so bilateral is not guided by the spike.
    snapped = np.where(mask, mean_excl, work).astype(np.float32, copy=False)
    if torch_dev is not None:
        smooth = _bilateral_self_torch(
            snapped, radius, sigma_spatial, sigma_range, device=torch_dev
        )
    else:
        smooth = _bilateral_self(snapped, radius, sigma_spatial, sigma_range)

    cleaned = work * (1.0 - strength) + smooth * strength
    out = np.where(mask, cleaned, work).astype(np.float32, copy=False)
    out = np.where(finite, out, src)
    return out


def denoise_depth_stack(
    depth: np.ndarray,
    params: DenoiseParams,
    *,
    device: str = "cpu",
    workers: int = 1,
) -> np.ndarray:
    """Denoise a ``[T,H,W]`` or ``[T,H,W,C]`` stack; returns same layout."""
    arr = np.asarray(depth, dtype=np.float32)
    had_channel = False
    if arr.ndim == 4:
        had_channel = True
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr.mean(axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Expected depth stack [T,H,W], got {depth.shape}")

    t = arr.shape[0]
    out = np.empty_like(arr)
    torch_dev = _resolve_torch_device(device)
    # GPU paths are serialized on one device.
    n_workers = 1 if torch_dev is not None else max(1, int(workers))

    if n_workers == 1 or t == 1:
        for i in range(t):
            out[i] = denoise_frame(arr[i], params, device=device)
            if (i + 1) % 8 == 0 or i + 1 == t:
                print(f"  [denoise] {i + 1}/{t}", flush=True)
    else:
        with ThreadPoolExecutor(
            max_workers=n_workers, thread_name_prefix="dvd-dn"
        ) as pool:
            futs = {
                pool.submit(denoise_frame, arr[i], params, device=device): i
                for i in range(t)
            }
            done = 0
            for fut, i in futs.items():
                out[i] = fut.result()
                done += 1
                if done % 8 == 0 or done == t:
                    print(f"  [denoise] {done}/{t}", flush=True)

    if had_channel:
        return out[..., None]
    return out


class ParallelDenoiser:
    """
    Denoise frames already resident in a float16 ``FrameMemmapStore``.

    Jobs run in frame order through a coordinator + thread pool (CPU) or a
    single CUDA/MPS worker — same concurrency model as ``ParallelUpscaler``.
    """

    def __init__(
        self,
        store: "FrameMemmapStore",
        params: DenoiseParams,
        *,
        workers: int = 2,
        device: str = "cpu",
    ):
        self.store = store
        self.params = params
        self.total_frames = int(store.total_frames)
        self.device = str(device or "cpu")
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._completed = 0

        torch_dev = _resolve_torch_device(self.device)
        if torch_dev is not None:
            self.workers = 1
            self._pool_size = 1
            print(
                f"Denoise device: {torch_dev} "
                f"(kernel={params.kernel_size}, 1 worker)",
                flush=True,
            )
        else:
            self.workers = max(1, int(workers))
            try:
                from resources import max_cpu_threads

                pool_cap = max_cpu_threads()
            except Exception:
                pool_cap = os.cpu_count() or 4
            self._pool_size = max(self.workers, pool_cap)
            print(
                f"Denoise device: cpu "
                f"(kernel={params.kernel_size}, threads={self.workers})",
                flush=True,
            )

    def run(self) -> None:
        """Scan + denoise every frame in the memmap (in place)."""
        if self.total_frames <= 0:
            return
        t0 = time.perf_counter()
        print(
            f"Parallel denoise: {self.store.width}x{self.store.height} x "
            f"{self.total_frames} kernel={self.params.kernel_size} "
            f"noise_sigma={self.params.noise_sigma}",
            flush=True,
        )
        pool = ThreadPoolExecutor(
            max_workers=self._pool_size, thread_name_prefix="dvd-dn"
        )
        inflight: dict[int, Future] = {}
        try:
            next_submit = 0
            while next_submit < self.total_frames or inflight:
                while (
                    next_submit < self.total_frames
                    and len(inflight) < self.workers
                ):
                    idx = next_submit
                    next_submit += 1
                    inflight[idx] = pool.submit(
                        _denoise_memmap_frame,
                        self.store,
                        idx,
                        self.params,
                        self.device,
                    )
                if not inflight:
                    break
                # Prefer completing the oldest in-flight frame for steady progress.
                idx = min(inflight)
                fut = inflight.pop(idx)
                fut.result()
                with self._lock:
                    self._completed += 1
                    done = self._completed
                if done % 8 == 0 or done == self.total_frames:
                    print(f"  [denoise] {done}/{self.total_frames}", flush=True)
            self.store.flush()
        except BaseException as exc:  # noqa: BLE001
            self._error = exc
            print(f"  [denoise] ERROR: {type(exc).__name__}: {exc}", flush=True)
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=False)
        elapsed = time.perf_counter() - t0
        print(
            f"Denoise complete: {self.total_frames} frames in {elapsed:.1f}s",
            flush=True,
        )


def _denoise_memmap_frame(
    store: "FrameMemmapStore",
    idx: int,
    params: DenoiseParams,
    device: str,
) -> int:
    frame = np.asarray(store[idx], dtype=np.float32)
    cleaned = denoise_frame(frame, params, device=device)
    store[idx] = cleaned.astype(np.float16, copy=False)
    return idx
