"""RGB-guided sharp depth upsampling (Joint Bilateral Upsampling).

Ported from the depth-anything / da3_video pipeline:
Lanczos flats + Kopf JBU at depth discontinuities, guided by full-res video luminance.
"""

from __future__ import annotations

import math
import os
import queue
import shutil
import tempfile
import threading
import time
import json
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from cache_root import fat32_max_file_bytes
from da3_stabilize import normalize_disparity_to_bgr_u8


@dataclass(frozen=True)
class UpsampleParams:
    edge_radius: int = 2
    sigma_range: float = 0.08
    edge_strength: float = 1.0
    depth_gate: float = 0.85


def cleanup_stale_up_caches(
    cache_dir: Path,
    keep_pid: int | None = None,
    *,
    keep_persistent: bool = True,
) -> None:
    """Remove leftover ephemeral ``up_cache_<pid>`` dirs from crashed runs.

    Persistent video-keyed caches (``up_cache_v_*``) are kept when
    ``keep_persistent`` is True so ``--keep-upsample-cache`` survives cleanup.
    """
    if not cache_dir.is_dir():
        return
    freed = 0
    for pattern in ("up_cache_*", ".up_cache_*"):
        for path in cache_dir.glob(pattern):
            if not path.is_dir():
                continue
            name = path.name.lstrip(".")
            # Persistent reusable caches: up_cache_v_<stem>_<hash>
            if keep_persistent and (
                name.startswith("up_cache_v_") or name.startswith(".up_cache_v_")
            ):
                continue
            suffix = name.split("_")[-1]
            if keep_pid is not None and suffix == str(keep_pid):
                continue
            # Only auto-delete ephemeral pid caches (numeric suffix).
            if not suffix.isdigit():
                continue
            try:
                before = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            except OSError:
                before = 0
            shutil.rmtree(path, ignore_errors=True)
            freed += before
    if freed > 0:
        print(
            f"Cleared stale upsample caches under {cache_dir} "
            f"(~{freed / (1024**3):.2f} GiB)",
            flush=True,
        )


class FrameMemmapStore:
    """Float16 frame store, optionally split across files under a max size.

    On FAT32 a single file cannot exceed ~4 GiB; this splits into ``depth_f16_XXXX.dat``
    chunks when needed while preserving ``store[i] = ...`` / ``store[i0:i1]`` access.
    """

    def __init__(
        self,
        cache_dir: Path,
        total_frames: int,
        height: int,
        width: int,
        *,
        max_file_bytes: int | None = None,
        reuse: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.total_frames = int(total_frames)
        self.height = int(height)
        self.width = int(width)
        self.reused = False
        frame_bytes = self.height * self.width * 2
        if frame_bytes <= 0:
            raise ValueError("invalid frame geometry for memmap store")
        if max_file_bytes is None:
            frames_per_chunk = self.total_frames
        else:
            frames_per_chunk = max(1, int(max_file_bytes // frame_bytes))
            if frames_per_chunk < 1:
                raise RuntimeError(
                    f"Single float16 frame is {frame_bytes / (1024**2):.1f} MiB, "
                    f"which exceeds the filesystem max file size "
                    f"({max_file_bytes / (1024**3):.2f} GiB). Lower output resolution."
                )
        self.frames_per_chunk = frames_per_chunk
        self._mms: list[np.memmap] = []
        self.paths: list[Path] = []
        remaining = self.total_frames
        chunk_i = 0
        n_chunks = (
            1
            if frames_per_chunk >= self.total_frames
            else int(math.ceil(self.total_frames / frames_per_chunk))
        )

        # Probe whether a complete prior allocation exists (skip FAT32 zero-fill).
        can_reuse = reuse
        if can_reuse:
            probe_remaining = self.total_frames
            probe_i = 0
            while probe_remaining > 0:
                n = min(probe_remaining, self.frames_per_chunk)
                path = self.cache_dir / f"depth_f16_{probe_i:04d}.dat"
                expect = n * frame_bytes
                try:
                    if not path.is_file() or path.stat().st_size != expect:
                        can_reuse = False
                        break
                except OSError:
                    can_reuse = False
                    break
                probe_remaining -= n
                probe_i += 1
            if can_reuse and probe_i != n_chunks:
                can_reuse = False

        mode = "r+" if can_reuse else "w+"
        self.reused = can_reuse
        # mode="w+" zero-fills each file; on FAT32 USB this can take many minutes
        # with no other logs — print per-chunk progress so it does not look hung.
        while remaining > 0:
            n = min(remaining, self.frames_per_chunk)
            path = self.cache_dir / f"depth_f16_{chunk_i:04d}.dat"
            part_bytes = n * frame_bytes
            action = "reopening" if can_reuse else "allocating"
            print(
                f"  [upsample] {action} memmap part {chunk_i + 1}/{n_chunks} "
                f"({part_bytes / (1024**3):.2f} GiB) -> {path.name} …",
                flush=True,
            )
            t0 = time.perf_counter()
            mm = np.memmap(
                path,
                dtype=np.float16,
                mode=mode,
                shape=(n, self.height, self.width),
            )
            elapsed = time.perf_counter() - t0
            print(
                f"  [upsample] part {chunk_i + 1}/{n_chunks} ready "
                f"({elapsed:.1f}s)",
                flush=True,
            )
            self.paths.append(path)
            self._mms.append(mm)
            remaining -= n
            chunk_i += 1

    @property
    def nbytes(self) -> int:
        return self.total_frames * self.height * self.width * 2

    def __setitem__(self, idx: int, value: np.ndarray) -> None:
        chunk, local = divmod(int(idx), self.frames_per_chunk)
        self._mms[chunk][local] = value

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.total_frames)
            if step != 1:
                raise IndexError("FrameMemmapStore only supports step=1 slices")
            out = np.empty(
                (stop - start, self.height, self.width), dtype=np.float16
            )
            for i, g in enumerate(range(start, stop)):
                chunk, local = divmod(g, self.frames_per_chunk)
                out[i] = self._mms[chunk][local]
            return out
        chunk, local = divmod(int(key), self.frames_per_chunk)
        return self._mms[chunk][local]

    def flush(self) -> None:
        for mm in self._mms:
            mm.flush()

    def close(self) -> None:
        for mm in self._mms:
            try:
                mm.flush()
            except Exception:
                pass
            del mm
        self._mms.clear()


def default_upsample_params(
    infer_h: int, infer_w: int, orig_h: int, orig_w: int
) -> UpsampleParams:
    upscale = max(orig_h / max(infer_h, 1), orig_w / max(infer_w, 1))
    edge_radius = max(1, int(math.ceil(upscale)))
    return UpsampleParams(
        edge_radius=edge_radius,
        sigma_range=0.08,
        edge_strength=1.0,
        depth_gate=0.85,
    )


def _depth_edge_weight(depth: np.ndarray, feather: float) -> np.ndarray:
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    finite = np.isfinite(depth)
    if int(finite.sum()) < 64:
        return np.zeros_like(mag, dtype=np.float32)
    rng_lo, rng_hi = np.percentile(depth[finite], [2.0, 98.0])
    scale = float(rng_hi - rng_lo)
    if scale <= 1e-6:
        return np.zeros_like(mag, dtype=np.float32)
    w = np.clip(mag / (0.5 * scale), 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)
    if feather > 0:
        w = cv2.GaussianBlur(w, (0, 0), feather)
    return w.astype(np.float32)


def joint_bilateral_upsample(
    disp_low: np.ndarray,
    guide_gray: np.ndarray,
    radius: int,
    sigma_range: float,
    sigma_spatial: float | None = None,
) -> np.ndarray:
    """Joint Bilateral Upsampling (Kopf et al.) guided by full-res grayscale."""
    hi_h, hi_w = guide_gray.shape[:2]
    lo_h, lo_w = disp_low.shape[:2]
    if sigma_spatial is None:
        sigma_spatial = max(1.0, float(radius))

    guide_low = cv2.resize(guide_gray, (lo_w, lo_h), interpolation=cv2.INTER_AREA)
    disp_low = disp_low.astype(np.float32)

    acc = np.zeros((hi_h, hi_w), dtype=np.float32)
    acc_w = np.zeros((hi_h, hi_w), dtype=np.float32)
    inv_2ss = 1.0 / (2.0 * sigma_spatial * sigma_spatial)
    inv_2sr = 1.0 / (2.0 * sigma_range * sigma_range)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            d_shift = np.roll(disp_low, shift=(dy, dx), axis=(0, 1))
            g_shift = np.roll(guide_low, shift=(dy, dx), axis=(0, 1))
            d_up = cv2.resize(d_shift, (hi_w, hi_h), interpolation=cv2.INTER_NEAREST)
            g_up = cv2.resize(g_shift, (hi_w, hi_h), interpolation=cv2.INTER_NEAREST)
            w_spatial = math.exp(-(dx * dx + dy * dy) * inv_2ss)
            diff = guide_gray - g_up
            w = w_spatial * np.exp(-(diff * diff) * inv_2sr)
            acc += w * d_up
            acc_w += w

    return (acc / np.maximum(acc_w, 1e-8)).astype(np.float32)


def joint_bilateral_upsample_cuda(
    disp_low: np.ndarray,
    guide_gray: np.ndarray,
    radius: int,
    sigma_range: float,
    sigma_spatial: float | None = None,
    *,
    device: str = "cuda",
) -> np.ndarray:
    """Same JBU as ``joint_bilateral_upsample``, accumulated on CUDA."""
    import torch
    import torch.nn.functional as F

    hi_h, hi_w = guide_gray.shape[:2]
    lo_h, lo_w = disp_low.shape[:2]
    if sigma_spatial is None:
        sigma_spatial = max(1.0, float(radius))
    inv_2ss = 1.0 / (2.0 * sigma_spatial * sigma_spatial)
    inv_2sr = 1.0 / (2.0 * sigma_range * sigma_range)

    dev = torch.device(device)
    guide = torch.from_numpy(np.ascontiguousarray(guide_gray, dtype=np.float32)).to(
        dev, non_blocking=True
    )
    disp = torch.from_numpy(np.ascontiguousarray(disp_low, dtype=np.float32)).to(
        dev, non_blocking=True
    )
    # guide_low via area downsample
    guide_b = guide.view(1, 1, hi_h, hi_w)
    guide_low = F.interpolate(guide_b, size=(lo_h, lo_w), mode="area").view(lo_h, lo_w)
    disp_b = disp.view(1, 1, lo_h, lo_w)

    acc = torch.zeros((hi_h, hi_w), device=dev, dtype=torch.float32)
    acc_w = torch.zeros((hi_h, hi_w), device=dev, dtype=torch.float32)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            d_shift = torch.roll(disp, shifts=(dy, dx), dims=(0, 1)).view(1, 1, lo_h, lo_w)
            g_shift = torch.roll(guide_low, shifts=(dy, dx), dims=(0, 1)).view(1, 1, lo_h, lo_w)
            d_up = F.interpolate(d_shift, size=(hi_h, hi_w), mode="nearest").view(hi_h, hi_w)
            g_up = F.interpolate(g_shift, size=(hi_h, hi_w), mode="nearest").view(hi_h, hi_w)
            w_spatial = math.exp(-(dx * dx + dy * dy) * inv_2ss)
            diff = guide - g_up
            w = w_spatial * torch.exp(-(diff * diff) * inv_2sr)
            acc = acc + w * d_up
            acc_w = acc_w + w

    out = acc / torch.clamp(acc_w, min=1e-8)
    return out.detach().float().cpu().numpy().astype(np.float32, copy=False)


def _depth_edge_weight_cuda(depth: np.ndarray, feather: float, *, device: str = "cuda") -> np.ndarray:
    import torch
    import torch.nn.functional as F

    dev = torch.device(device)
    d = torch.from_numpy(np.ascontiguousarray(depth, dtype=np.float32)).to(dev)
    # Sobel via conv2d
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=dev
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=dev
    ).view(1, 1, 3, 3)
    x = d.view(1, 1, *d.shape)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy).view(*d.shape)
    finite = torch.isfinite(d)
    if int(finite.sum().item()) < 64:
        return np.zeros(depth.shape, dtype=np.float32)
    vals = d[finite]
    # percentile via sort (small enough at infer res)
    sorted_v, _ = torch.sort(vals.reshape(-1))
    n = sorted_v.numel()
    lo = sorted_v[max(0, int(0.02 * (n - 1)))]
    hi = sorted_v[min(n - 1, int(0.98 * (n - 1)))]
    scale = float((hi - lo).item())
    if scale <= 1e-6:
        return np.zeros(depth.shape, dtype=np.float32)
    w = torch.clamp(mag / (0.5 * scale), 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)
    if feather > 0:
        # approx Gaussian blur with repeated box / F.conv gaussian
        sigma = float(feather)
        k = max(3, int(round(sigma * 6)) | 1)
        half = k // 2
        xs = torch.arange(k, device=dev, dtype=torch.float32) - half
        ker = torch.exp(-(xs * xs) / (2.0 * sigma * sigma))
        ker = ker / ker.sum()
        kx1 = ker.view(1, 1, 1, k)
        ky1 = ker.view(1, 1, k, 1)
        wb = w.view(1, 1, *w.shape)
        wb = F.conv2d(F.pad(wb, (half, half, 0, 0), mode="reflect"), kx1)
        wb = F.conv2d(F.pad(wb, (0, 0, half, half), mode="reflect"), ky1)
        w = wb.view(*w.shape)
    return w.detach().float().cpu().numpy().astype(np.float32, copy=False)


def sharp_upsample(
    disp_low: np.ndarray,
    guide_gray: np.ndarray,
    out_size: tuple[int, int],
    params: UpsampleParams,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """
    Lanczos base in flats + full JBU at depth discontinuities.

    ``guide_gray`` must be full-res float32 in [0, 1] (RGB luminance).
    ``out_size`` is (width, height).
    ``device``: ``\"cpu\"`` (OpenCV/NumPy) or ``\"cuda\"`` (Torch JBU on GPU).
    """
    out_w, out_h = out_size
    if guide_gray.shape[:2] != (out_h, out_w):
        guide_gray = cv2.resize(guide_gray, (out_w, out_h), interpolation=cv2.INTER_AREA)
    guide_gray = np.clip(guide_gray.astype(np.float32), 0.0, 1.0)

    # Lanczos base stays on CPU (cheap vs JBU); bicubic fallback if needed.
    base = cv2.resize(disp_low, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4).astype(
        np.float32
    )
    jbu_radius = max(1, int(round(params.edge_radius / 2.0)))
    use_cuda = str(device).startswith("cuda")
    if use_cuda:
        try:
            import torch

            if not torch.cuda.is_available():
                use_cuda = False
        except Exception:
            use_cuda = False

    if use_cuda:
        jbu = joint_bilateral_upsample_cuda(
            disp_low,
            guide_gray,
            radius=jbu_radius,
            sigma_range=params.sigma_range,
            device=device,
        )
        dw_low = _depth_edge_weight_cuda(disp_low, feather=1.0, device=device)
    else:
        jbu = joint_bilateral_upsample(
            disp_low,
            guide_gray,
            radius=jbu_radius,
            sigma_range=params.sigma_range,
        )
        dw_low = _depth_edge_weight(disp_low, feather=1.0)

    dw = cv2.resize(dw_low, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    if params.depth_gate > 0:
        gate = (1.0 - params.depth_gate) + params.depth_gate * dw
    else:
        gate = np.ones_like(dw)
    ew = np.clip(gate.astype(np.float32) * float(params.edge_strength), 0.0, 1.0)
    dw_mean = float(np.nanmean(dw))
    noise_factor = float(np.clip((dw_mean - 0.08) / 0.22, 0.0, 1.0))
    noise_damp = 1.0 - 0.45 * noise_factor
    ew = np.clip(ew * noise_damp, 0.0, 1.0)
    return (base * (1.0 - ew) + jbu * ew).astype(np.float32)


def depth_frame_to_disp(frame: np.ndarray) -> np.ndarray:
    """Convert DVD depth frame [H,W,C] or [H,W] to single-channel float32."""
    if frame.ndim == 3:
        if frame.shape[-1] == 1:
            return frame[..., 0].astype(np.float32)
        return frame.mean(axis=-1).astype(np.float32)
    return frame.astype(np.float32)


class _GuideGrayReader:
    """Sequential full-res grayscale guide frames from the source video."""

    def __init__(self, video_path: str | Path, out_w: int, out_h: int):
        self.path = str(video_path)
        self.out_w = out_w
        self.out_h = out_h
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video for upsample guides: {self.path}")
        self.next_idx = 0

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def get(self, idx: int) -> np.ndarray:
        if idx < self.next_idx:
            self.close()
            self.cap = cv2.VideoCapture(self.path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot reopen video for guides: {self.path}")
            self.next_idx = 0

        while self.next_idx <= idx:
            assert self.cap is not None
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError(
                    f"Failed reading guide frame {self.next_idx} from {self.path}"
                )
            if self.next_idx == idx:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                if gray.shape[0] != self.out_h or gray.shape[1] != self.out_w:
                    gray = cv2.resize(
                        gray, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA
                    )
                self.next_idx += 1
                return gray
            self.next_idx += 1
        raise RuntimeError(f"Guide reader skipped past frame {idx}")


UP_CACHE_META = "meta.json"
UP_CACHE_DONE = "done.bin"
UP_CACHE_VERSION = 1


def _video_fingerprint(video_path: Path) -> dict:
    resolved = video_path.resolve()
    st = resolved.stat()
    return {
        "video_path": str(resolved),
        "video_size": int(st.st_size),
        "video_mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
    }


def _upsample_meta(
    video_path: Path,
    *,
    total_frames: int,
    out_w: int,
    out_h: int,
    params: UpsampleParams,
) -> dict:
    meta = _video_fingerprint(video_path)
    meta.update(
        {
            "version": UP_CACHE_VERSION,
            "total_frames": int(total_frames),
            "out_w": int(out_w),
            "out_h": int(out_h),
            "params": asdict(params),
        }
    )
    return meta


def _meta_matches(existing: dict, expected: dict) -> bool:
    keys = (
        "version",
        "video_path",
        "video_size",
        "video_mtime_ns",
        "total_frames",
        "out_w",
        "out_h",
        "params",
    )
    try:
        return all(existing.get(k) == expected.get(k) for k in keys)
    except Exception:
        return False


class ParallelUpscaler:
    """
    Upscale finalized depth frames while DVD runs the next window.

    Jobs are submitted in frame order. A coordinator thread reads RGB guides and
    dispatches JBU work to a thread pool (CPU) or a single CUDA worker.
    Completed frames are written into float16 memmap file(s).
    """

    def __init__(
        self,
        video_path: str | Path,
        out_size: tuple[int, int],
        params: UpsampleParams,
        *,
        total_frames: int,
        workers: int = 2,
        max_inflight: int | None = None,
        cache_dir: str | Path | None = None,
        keep_cache: bool = False,
        device: str = "cpu",
    ):
        if total_frames <= 0:
            raise ValueError("total_frames must be > 0")
        self.video_path = Path(video_path)
        self.out_w, self.out_h = out_size
        self.params = params
        self.total_frames = int(total_frames)
        self.keep_cache = bool(keep_cache)
        self.device = str(device or "cpu")
        self._lock = threading.Lock()
        # CUDA JBU is serialized on one GPU — extra CPU workers fight for VRAM copies.
        if self.device.startswith("cuda"):
            self.workers = 1
            self.max_inflight = 1
            self._pool_size = 1
            print(
                f"Upsample device: {self.device} (JBU on GPU, 1 worker)",
                flush=True,
            )
        else:
            self.workers = max(1, int(workers))
            self.max_inflight = max_inflight or max(2, self.workers * 2)
            try:
                from resources import max_cpu_threads

                pool_cap = max_cpu_threads()
            except Exception:
                pool_cap = os.cpu_count() or 4
            self._pool_size = max(self.workers, pool_cap)
            print(f"Upsample device: cpu (JBU threads={self.workers})", flush=True)

        expected_meta = _upsample_meta(
            self.video_path,
            total_frames=self.total_frames,
            out_w=self.out_w,
            out_h=self.out_h,
            params=params,
        )
        reuse = False

        if cache_dir is None:
            cache_dir = Path(tempfile.mkdtemp(prefix="dvd_up_"))
            self.keep_cache = False
        else:
            cache_dir = Path(cache_dir)
            if cache_dir.exists():
                meta_path = cache_dir / UP_CACHE_META
                if self.keep_cache and meta_path.is_file():
                    try:
                        existing = json.loads(meta_path.read_text(encoding="utf-8"))
                        reuse = _meta_matches(existing, expected_meta)
                    except Exception:
                        reuse = False
                if not reuse:
                    shutil.rmtree(cache_dir, ignore_errors=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir
        self._meta_path = self.cache_dir / UP_CACHE_META

        bytes_needed = self.total_frames * self.out_h * self.out_w * 2
        if not reuse:
            free = shutil.disk_usage(self.cache_dir).free
            # Keep OS headroom; memmap needs the full file reserved up front.
            headroom = max(512 * 1024**2, int(bytes_needed * 0.15))
            if free < bytes_needed + headroom:
                raise RuntimeError(
                    f"Not enough free disk for upsample cache: need "
                    f"~{(bytes_needed + headroom) / (1024**3):.2f} GiB, "
                    f"have {free / (1024**3):.2f} GiB free under {self.cache_dir}. "
                    f"Free space on the cache volume (or delete old "
                    f"dit_cache_* / up_cache_* folders) and retry."
                )

        max_file = fat32_max_file_bytes(self.cache_dir)
        fat_note = " (FAT32 4 GiB splits; USB can take several minutes)" if max_file else ""
        if reuse:
            print(
                f"Reusing upsample cache: {bytes_needed / (1024**3):.2f} GiB "
                f"under {self.cache_dir}",
                flush=True,
            )
        else:
            print(
                f"Allocating upsample cache: {bytes_needed / (1024**3):.2f} GiB "
                f"under {self.cache_dir}{fat_note}",
                flush=True,
            )
        self._mm = FrameMemmapStore(
            self.cache_dir,
            self.total_frames,
            self.out_h,
            self.out_w,
            max_file_bytes=max_file,
            reuse=reuse,
        )
        if self.keep_cache:
            try:
                self._meta_path.write_text(
                    json.dumps(expected_meta, indent=2), encoding="utf-8"
                )
            except OSError as exc:
                print(f"  [upsample] warning: could not write meta.json ({exc})", flush=True)

        self._done_path = self.cache_dir / UP_CACHE_DONE
        self._done = np.zeros(self.total_frames, dtype=np.bool_)
        self._pre_done = 0
        if reuse and self.keep_cache and self._done_path.is_file():
            try:
                loaded = np.fromfile(self._done_path, dtype=np.uint8)
                if loaded.size == self.total_frames:
                    self._done = loaded.astype(np.bool_, copy=False)
                    self._pre_done = int(self._done.sum())
                    if self._pre_done:
                        print(
                            f"  [upsample] skipping {self._pre_done}/{self.total_frames} "
                            f"frames already complete in cache",
                            flush=True,
                        )
            except OSError as exc:
                print(f"  [upsample] warning: could not read done.bin ({exc})", flush=True)
        self._error: BaseException | None = None
        self._submitted = 0
        self._completed = 0
        self._jobs: queue.Queue = queue.Queue()
        self._coord = threading.Thread(
            target=self._coordinator, name="dvd-up-coord", daemon=True
        )
        self._coord.start()
        n_parts = len(self._mm.paths)
        part_note = f", {n_parts} FAT32-safe parts" if n_parts > 1 else ""
        keep_note = ", keep=on" if self.keep_cache else ""
        reuse_note = ", reused" if self._mm.reused else ""
        print(
            f"Parallel upscaler ready: {self.out_w}x{self.out_h} x {self.total_frames} "
            f"JBU workers={self.workers}/{self._pool_size} "
            f"edge_radius={params.edge_radius} "
            f"cache={self.cache_dir} ({bytes_needed / (1024**3):.2f} GiB{part_note}"
            f"{reuse_note}{keep_note})",
            flush=True,
        )

    def set_workers(self, workers: int, max_inflight: int | None = None) -> None:
        """Dynamically resize active JBU concurrency (governor-driven)."""
        if self.device.startswith("cuda"):
            return
        with self._lock:
            new_w = max(1, min(int(workers), self._pool_size))
            if new_w != self.workers:
                print(f"  [upsample] workers {self.workers} -> {new_w}", flush=True)
            self.workers = new_w
            self.max_inflight = max_inflight or max(2, self.workers * 2)

    def pending(self) -> int:
        with self._lock:
            return max(0, self._submitted - self._completed)

    def submit_range(self, depth_bthwc: np.ndarray, global_start: int, global_end: int) -> None:
        """Submit frames [global_start, global_end) from a [1,T,H,W,C] depth slice."""
        if global_end <= global_start:
            return
        if self._error is not None:
            raise self._error
        if global_end > self.total_frames:
            raise ValueError(
                f"submit_range end {global_end} exceeds total_frames {self.total_frames}"
            )
        local_t = 0
        queued = 0
        for g in range(global_start, global_end):
            if self._done[g]:
                local_t += 1
                continue
            frame = depth_bthwc[0, local_t]
            disp = depth_frame_to_disp(frame)
            self._jobs.put(("frame", g, disp))
            with self._lock:
                self._submitted += 1
            queued += 1
            local_t += 1
        if queued:
            print(
                f"  queued upsample frames [{global_start}:{global_end}) "
                f"new={queued} pending={self.pending()}",
                flush=True,
            )

    def finish(self) -> None:
        """Drain workers; keep float16 memmap on disk for streaming export.

        Do **not** materialize ``[T,H,W]`` float32 in RAM — at native 4K that is
        ~20 GiB and triggers ``Killed: 9`` on 8 GB Macs. Call
        ``export_grayscale_mp4`` (or ``iter_frames``) then ``close``.
        """
        self._jobs.put(("stop", None, None))
        self._coord.join()
        if self._error is not None:
            self._cleanup_cache()
            raise self._error
        missing = np.flatnonzero(~self._done)
        if missing.size:
            # Persist partial progress so --keep-upsample-cache can resume.
            self._persist_done()
            self._cleanup_cache()
            raise RuntimeError(
                f"Missing upsampled frames: {missing[:8].tolist()}"
                f"{'...' if missing.size > 8 else ''} "
                f"({missing.size}/{self.total_frames} missing; "
                f"cache kept at {self.cache_dir})"
            )
        self._persist_done()
        print(
            f"Upsample complete: {self.total_frames} frames @ "
            f"{self.out_w}x{self.out_h} (memmap, not loaded into RAM)",
            flush=True,
        )

    def _persist_done(self) -> None:
        if not self.keep_cache:
            return
        try:
            self._done.astype(np.uint8).tofile(self._done_path)
        except OSError as exc:
            print(f"  [upsample] warning: could not write done.bin ({exc})", flush=True)

    def depth_range(self, *, chunk: int = 4) -> tuple[float, float]:
        """Min/max over the float16 memmap without loading the full stack."""
        if getattr(self, "_mm", None) is None:
            raise RuntimeError("Upsample cache already closed")
        d_min = np.inf
        d_max = -np.inf
        chunk = max(1, int(chunk))
        for i0 in range(0, self.total_frames, chunk):
            i1 = min(self.total_frames, i0 + chunk)
            block = np.asarray(self._mm[i0:i1], dtype=np.float32)
            d_min = min(d_min, float(np.nanmin(block)))
            d_max = max(d_max, float(np.nanmax(block)))
        if not np.isfinite(d_min) or not np.isfinite(d_max):
            raise RuntimeError("Upsampled depth has no finite values")
        return d_min, d_max

    def percentile_range(
        self,
        *,
        percentile: float = 2.0,
        chunk: int = 4,
        sample_per_frame: int = 20000,
    ) -> tuple[float, float]:
        """Sampled percentile range without materializing the full stack."""
        if getattr(self, "_mm", None) is None:
            raise RuntimeError("Upsample cache already closed")
        samples: list[np.ndarray] = []
        chunk = max(1, int(chunk))
        for i0 in range(0, self.total_frames, chunk):
            i1 = min(self.total_frames, i0 + chunk)
            block = np.asarray(self._mm[i0:i1], dtype=np.float32)
            for j in range(block.shape[0]):
                flat = block[j].reshape(-1)
                flat = flat[np.isfinite(flat)]
                if flat.size == 0:
                    continue
                if flat.size > sample_per_frame:
                    idx = np.random.randint(0, flat.size, size=sample_per_frame)
                    flat = flat[idx]
                samples.append(flat)
        if not samples:
            raise RuntimeError("Upsampled depth has no finite values")
        all_s = np.concatenate(samples)
        lo = float(np.percentile(all_s, float(percentile)))
        hi = float(np.percentile(all_s, 100.0 - float(percentile)))
        if hi <= lo:
            hi = lo + 1e-6
        return lo, hi

    def iter_frames(self):
        """Yield float32 ``[H, W]`` frames from the memmap (one at a time)."""
        if getattr(self, "_mm", None) is None:
            raise RuntimeError("Upsample cache already closed")
        for i in range(self.total_frames):
            yield np.asarray(self._mm[i], dtype=np.float32)

    def export_grayscale_mp4(
        self,
        path: str | Path,
        fps: float,
        *,
        quality: int = 6,
    ) -> Path:
        """Two-pass normalize + stream-encode; never holds more than one frame."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        print("Scanning percentile depth range from memmap...", flush=True)
        lo, hi = self.percentile_range(percentile=2.0)
        print(
            f"Saving grayscale depth video -> {path} "
            f"(stream {self.total_frames} x {self.out_w}x{self.out_h})",
            flush=True,
        )
        del quality  # cv2 writer keeps the DA3-style path simple and stable.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(path), fourcc, float(fps), (self.out_w, self.out_h), isColor=True
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for: {path}")
        try:
            for i, frame in enumerate(self.iter_frames()):
                writer.write(normalize_disparity_to_bgr_u8(frame, lo, hi))
                done = i + 1
                if done % 32 == 0 or done == self.total_frames:
                    print(f"  [encode] {done}/{self.total_frames}", flush=True)
        finally:
            writer.release()
            self.close()
        return path

    def close(self) -> None:
        """Release memmap handles; delete cache unless ``keep_cache`` is set."""
        self._cleanup_cache(delete=not self.keep_cache)

    def _cleanup_cache(self, *, delete: bool | None = None) -> None:
        if delete is None:
            delete = not self.keep_cache
        try:
            mm = getattr(self, "_mm", None)
            if mm is not None:
                try:
                    mm.flush()
                    mm.close()
                except Exception:
                    pass
                self._mm = None  # type: ignore[assignment]
        except Exception:
            pass
        if not delete:
            if self.keep_cache:
                print(
                    f"Keeping upsample cache at {self.cache_dir} "
                    f"(reuse with --keep-upsample-cache)",
                    flush=True,
                )
            return
        try:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        except OSError:
            pass

    def _active_limits(self) -> tuple[int, int]:
        with self._lock:
            return self.workers, self.max_inflight

    def _coordinator(self) -> None:
        guide_reader: _GuideGrayReader | None = None
        pool: ThreadPoolExecutor | None = None
        inflight: dict[int, Future] = {}
        try:
            guide_reader = _GuideGrayReader(self.video_path, self.out_w, self.out_h)
            pool = ThreadPoolExecutor(
                max_workers=self._pool_size, thread_name_prefix="dvd-up"
            )
            while True:
                kind, idx, disp = self._jobs.get()
                if kind == "stop":
                    break
                assert idx is not None and disp is not None
                workers, max_inflight = self._active_limits()
                limit = max(1, min(workers, max_inflight, self._pool_size))
                while len(inflight) >= limit:
                    self._collect_done(inflight, block=True)
                    workers, max_inflight = self._active_limits()
                    limit = max(1, min(workers, max_inflight, self._pool_size))
                self._collect_done(inflight, block=False)
                guide = guide_reader.get(idx)
                fut = pool.submit(
                    _upsample_to_memmap,
                    disp,
                    guide,
                    (self.out_w, self.out_h),
                    self.params,
                    self._mm,
                    idx,
                    self.device,
                )
                inflight[idx] = fut

            while inflight:
                self._collect_done(inflight, block=True)
            self._mm.flush()
        except BaseException as exc:  # noqa: BLE001 — store and surface via finish()
            self._error = _enrich_os_error(exc, self.cache_dir)
            print(f"  [upsample] ERROR: {type(self._error).__name__}: {self._error}", flush=True)
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=False)
            if guide_reader is not None:
                guide_reader.close()

    def _collect_done(self, inflight: dict[int, Future], *, block: bool) -> None:
        if not inflight:
            return
        if block:
            idx = min(inflight)
            fut = inflight.pop(idx)
            done_idx = fut.result()
            self._done[done_idx] = True
            with self._lock:
                self._completed += 1
                done = self._completed + self._pre_done
                submitted = self._submitted + self._pre_done
            if done % 8 == 0 or done == submitted or done == self.total_frames:
                print(f"  [upsample] {done}/{self.total_frames}", flush=True)
            if self.keep_cache and (done % 16 == 0 or done == self.total_frames):
                self._persist_done()
            return

        done_ids = [i for i, f in inflight.items() if f.done()]
        for idx in done_ids:
            fut = inflight.pop(idx)
            done_idx = fut.result()
            self._done[done_idx] = True
            with self._lock:
                self._completed += 1
                done = self._completed + self._pre_done
                submitted = self._submitted + self._pre_done
            if done % 8 == 0 or done == submitted or done == self.total_frames:
                print(f"  [upsample] {done}/{self.total_frames}", flush=True)
            if self.keep_cache and (done % 16 == 0 or done == self.total_frames):
                self._persist_done()


def _enrich_os_error(exc: BaseException, cache_dir: Path) -> BaseException:
    if not isinstance(exc, OSError):
        return exc
    try:
        free = shutil.disk_usage(cache_dir).free
        free_gb = free / (1024**3)
    except Exception:
        free_gb = float("nan")
    return RuntimeError(
        f"{exc} (while writing upsample cache under {cache_dir}; "
        f"~{free_gb:.2f} GiB free). Free disk space on the cache volume and retry — "
        f"e.g. remove unused dit_cache_* / up_cache_* folders."
    )


def _upsample_to_memmap(
    disp_low: np.ndarray,
    guide_gray: np.ndarray,
    out_size: tuple[int, int],
    params: UpsampleParams,
    mm: FrameMemmapStore,
    idx: int,
    device: str = "cpu",
) -> int:
    up = sharp_upsample(disp_low, guide_gray, out_size, params, device=device)
    mm[idx] = up.astype(np.float16, copy=False)
    return idx
