#!/usr/bin/env python3
"""Generate a grayscale depth-map video with EnVision-Research DVD on Apple MPS.

On 8GB Apple Silicon the full Wan DiT+VAE cannot reside in MPS. This script keeps
DiT weights on CPU (resident across VAE), runs the VAE on CPU float16, and swaps
a few DiT blocks to MPS at a time. DiT is parked to a float16 mmap blob only on OOM.
"""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

# Reduce noisy shutdown leaks from tokenizers / OpenCV / torch workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.95")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.70")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
DVD_ROOT = REPO_ROOT / "vendor" / "DVD"
sys.path.insert(0, str(DVD_ROOT))
sys.path.insert(0, str(DVD_ROOT / "test_script"))

from accelerate import Accelerator  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from peft.tuners.lora.layer import Linear as LoraLinear  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from examples.wanvideo.model_training.WanTrainingModule import (  # noqa: E402
    WanTrainingModule,
)

import cv2  # noqa: E402

from da3_stabilize import (  # noqa: E402
    StabilizeParams,
    apply_flow_temporal_median,
    apply_shot_band_lock,
    derive_stabilize_params,
    lock_range as da3_lock_range,
    normalize_disparity_to_bgr_u8,
    probe_shot_ranges,
)
from upsample import (  # noqa: E402
    ParallelUpscaler,
    UpsampleParams,
    cleanup_stale_up_caches,
    default_upsample_params,
)
from cache_root import (  # noqa: E402
    resolve_cache_root,
    resolve_dit_cache_dir,
    up_cache_dir,
    up_cache_dir_for_video,
)
from resources import ResourceGovernor, max_cpu_threads  # noqa: E402
from pipeline import run_window_pipeline  # noqa: E402

# OpenCV / Torch: stay within the CPU soft-cap; avoid OpenCL on Mac.
try:
    cv2.setNumThreads(max_cpu_threads())
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass
try:
    torch.set_num_threads(max_cpu_threads())
except Exception:
    pass

DIT_CACHE_VERSION = 2
DIT_BLOB_NAME = "weights.f16.bin"

# Track open captures so we can release them on normal or abrupt interpreter exit.
_OPEN_CAPTURES: list[cv2.VideoCapture] = []


def _release_all_captures() -> None:
    while _OPEN_CAPTURES:
        cap = _OPEN_CAPTURES.pop()
        try:
            if cap is not None and cap.isOpened():
                cap.release()
        except Exception:
            pass


atexit.register(_release_all_captures)


def open_video_capture(video_path: str | Path) -> cv2.VideoCapture:
    """Prefer FFMPEG backend; AVFoundation can stall on some Mac decodes."""
    path = str(video_path)
    for backend in (getattr(cv2, "CAP_FFMPEG", 0), getattr(cv2, "CAP_AVFOUNDATION", 0), 0):
        cap = cv2.VideoCapture(path, backend) if backend else cv2.VideoCapture(path)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            _OPEN_CAPTURES.append(cap)
            return cap
    raise ValueError(f"Cannot open video: {video_path}")


def _forget_capture(cap: cv2.VideoCapture | None) -> None:
    if cap is None:
        return
    try:
        _OPEN_CAPTURES.remove(cap)
    except ValueError:
        pass
    try:
        if cap.isOpened():
            cap.release()
    except Exception:
        pass


def probe_video(video_path: str | Path) -> tuple[float, int, int, int]:
    """Return fps, frame_count, orig_h, orig_w without loading frames."""
    cap = open_video_capture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count <= 0:
        frame_count = 0
        while cap.grab():
            frame_count += 1
        _forget_capture(cap)
        cap = open_video_capture(video_path)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _forget_capture(cap)
    if frame_count <= 0:
        raise ValueError(f"No frames found in {video_path}")
    return fps, frame_count, orig_h, orig_w


def inference_hw(orig_h: int, orig_w: int, target_h: int, target_w: int) -> tuple[int, int]:
    """Match DVD resize_for_training_scale geometry, aligned to 16."""
    ratio = max(target_h / orig_h, target_w / orig_w)
    new_h = int(np.ceil(orig_h * ratio))
    new_w = int(np.ceil(orig_w * ratio))
    new_h = (new_h + 15) // 16 * 16
    new_w = (new_w + 15) // 16 * 16
    return new_h, new_w


def _preprocess_bgr_frame(frame_bgr, out_h: int, out_w: int, dtype: torch.dtype) -> torch.Tensor:
    """BGR uint8 -> RGB float CHW at inference resolution (no full-res float copy)."""
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if frame.shape[0] != out_h or frame.shape[1] != out_w:
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).to(
        dtype=dtype
    ) / 255.0
    return tensor


class InferResBgrStore:
    """Shared infer-resolution BGR frames for prep → stabilize reuse."""

    def __init__(self, out_h: int, out_w: int):
        self.out_h = int(out_h)
        self.out_w = int(out_w)
        self._frames: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()

    def put(self, idx: int, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape[0] != self.out_h or frame_bgr.shape[1] != self.out_w:
            frame_bgr = cv2.resize(
                frame_bgr, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA
            )
        with self._lock:
            if idx not in self._frames:
                self._frames[idx] = np.ascontiguousarray(frame_bgr)

    def get_range(self, start: int, end: int) -> list[np.ndarray]:
        with self._lock:
            missing = [i for i in range(start, end) if i not in self._frames]
            if missing:
                raise KeyError(
                    f"InferResBgrStore missing frames {missing[:8]}"
                    f"{'...' if len(missing) > 8 else ''}"
                )
            return [self._frames[i] for i in range(start, end)]

    def try_get_pair(self, idx: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(frame[idx], frame[idx+1])`` if both present."""
        with self._lock:
            a = self._frames.get(idx)
            b = self._frames.get(idx + 1)
            if a is None or b is None:
                return None
            return a, b

    def available_indices(self) -> list[int]:
        with self._lock:
            return sorted(self._frames.keys())

    def release_before(self, idx: int) -> None:
        with self._lock:
            for i in list(self._frames):
                if i < idx:
                    del self._frames[i]


class StreamingWindowReader:
    """Sequential decode with a small frame cache for overlapping windows."""

    def __init__(
        self,
        video_path: str | Path,
        out_h: int,
        out_w: int,
        dtype: torch.dtype,
        bgr_store: InferResBgrStore | None = None,
    ):
        self.path = str(video_path)
        self.out_h = out_h
        self.out_w = out_w
        self.dtype = dtype
        self.bgr_store = bgr_store
        self.cap = open_video_capture(self.path)
        self.next_idx = 0
        self.cache: dict[int, torch.Tensor] = {}

    def close(self) -> None:
        _forget_capture(self.cap)
        self.cap = None
        self.cache.clear()

    def read_window(self, start: int, end: int) -> torch.Tensor:
        # Drop frames that fall behind the sliding window.
        for idx in list(self.cache):
            if idx < start:
                del self.cache[idx]

        if start < self.next_idx and start not in self.cache:
            # Overlap rewind: reopen and decode forward (rare with our window schedule).
            print(
                f"  Rewinding decoder to frame {start} (was at {self.next_idx})...",
                flush=True,
            )
            self.close()
            self.cap = open_video_capture(self.path)
            self.next_idx = 0
            self.cache.clear()

        while self.next_idx < end:
            assert self.cap is not None
            ret, frame = self.cap.read()
            if not ret:
                break
            if self.next_idx >= start:
                if self.bgr_store is not None:
                    self.bgr_store.put(self.next_idx, frame)
                self.cache[self.next_idx] = _preprocess_bgr_frame(
                    frame, self.out_h, self.out_w, self.dtype
                )
            self.next_idx += 1
            if (self.next_idx - start) % 4 == 0 or self.next_idx == end:
                print(
                    f"  decoded frame {self.next_idx - 1}/{end - 1}",
                    flush=True,
                )

        missing = [i for i in range(start, end) if i not in self.cache]
        if missing:
            # Pad with last available frame if the file ended early.
            if not self.cache:
                raise RuntimeError(
                    f"Failed to read frames [{start}:{end}] from {self.path}"
                )
            last = self.cache[max(self.cache)]
            for i in missing:
                self.cache[i] = last.clone()
                if self.bgr_store is not None:
                    # Best-effort: reuse last BGR if store already has max key.
                    try:
                        last_bgr = self.bgr_store.get_range(max(self.cache), max(self.cache) + 1)[0]
                        self.bgr_store.put(i, last_bgr)
                    except Exception:
                        pass

        frames = [self.cache[i] for i in range(start, end)]
        return torch.stack(frames, dim=0).unsqueeze(0)


def system_memory_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
    except (AttributeError, ValueError, OSError):
        return 0.0


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    raise ValueError(f"Unknown device: {requested}")


def free_memory(device: torch.device | None = None) -> None:
    gc.collect()
    if device is not None and device.type == "mps" and hasattr(torch, "mps"):
        if hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except RuntimeError:
                pass
        if hasattr(torch.mps, "synchronize"):
            try:
                torch.mps.synchronize()
            except RuntimeError:
                pass
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def patch_cuda_helpers_for_mps() -> None:
    _orig_empty_cache = getattr(torch.cuda, "empty_cache", None)

    def _safe_empty_cache(*args, **kwargs):
        if torch.cuda.is_available() and _orig_empty_cache is not None:
            return _orig_empty_cache(*args, **kwargs)
        free_memory(torch.device("mps"))
        return None

    torch.cuda.empty_cache = _safe_empty_cache  # type: ignore[method-assign]

    def _safe_mem_get_info(device=None):
        if torch.cuda.is_available():
            return torch.cuda.mem_get_info(device)
        total = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        return int(total * 0.25), total

    torch.cuda.mem_get_info = _safe_mem_get_info  # type: ignore[method-assign]


def bake_and_unload_lora(root: torch.nn.Module) -> int:
    replaced = 0
    for parent in list(root.modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, LoraLinear):
                continue
            if not getattr(child, "merged", False):
                child.merge()
            base = child.get_base_layer()
            linear = torch.nn.Linear(
                base.in_features,
                base.out_features,
                bias=base.bias is not None,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
            linear.weight = torch.nn.Parameter(base.weight.detach())
            if base.bias is not None:
                linear.bias = torch.nn.Parameter(base.bias.detach())
            setattr(parent, child_name, linear)
            replaced += 1
    return replaced


def cast_pipe_dtype_low_mem(
    pipe, dtype: torch.dtype, only: tuple[str, ...] | None = None
) -> None:
    """Cast parameters one submodule at a time to avoid a 2× RAM spike."""
    for name, child in list(pipe.named_children()):
        if child is None:
            continue
        if only is not None and name not in only:
            continue
        print(f"  Casting pipe.{name} -> {dtype}", flush=True)
        if name == "dit" and hasattr(child, "blocks"):
            for sub_name, sub in child.named_children():
                if sub_name == "blocks":
                    for i, block in enumerate(sub):
                        block.to(dtype=dtype)
                        if i % 4 == 0:
                            free_memory()
                else:
                    sub.to(dtype=dtype)
                    free_memory()
        else:
            child.to(dtype=dtype)
            free_memory()
    pipe.torch_dtype = dtype


def drop_unused_pipe_modules(pipe) -> None:
    for attr in ("text_encoder", "image_encoder", "motion_controller", "vace", "prompter"):
        if getattr(pipe, attr, None) is not None:
            setattr(pipe, attr, None)
    free_memory()


def _move_tensors(obj, device):
    if torch.is_tensor(obj):
        t = obj
        # MPS rejects float64 / complex128 (e.g. RoPE freqs).
        if device.type == "mps":
            if t.dtype == torch.float64:
                t = t.to(dtype=torch.float32)
            elif t.dtype == torch.complex128:
                t = t.to(dtype=torch.complex64)
        return t.to(device, non_blocking=False)
    if isinstance(obj, (list, tuple)):
        t = [_move_tensors(x, device) for x in obj]
        return type(obj)(t)
    if isinstance(obj, dict):
        return {k: _move_tensors(v, device) for k, v in obj.items()}
    return obj


def cast_dit_freqs_for_mps(dit: torch.nn.Module) -> None:
    """Ensure RoPE frequency buffers are complex64 (MPS-safe)."""
    freqs = getattr(dit, "freqs", None)
    if freqs is None:
        return
    dit.freqs = tuple(
        f.to(dtype=torch.complex64) if torch.is_tensor(f) else f for f in freqs
    )


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    needles = (
        "out of memory",
        "not enough memory",
        "mps backend out of memory",
        "failed to allocate",
        "oom",
    )
    return any(n in msg for n in needles)


@contextmanager
def cpu_heavy_threads(n: int | None = None):
    """Use more CPU threads during VAE / decode; restore afterward.

    Defaults to ``max_cpu_threads()`` (≤95% of cores) so GEMM does not pin
    the machine at 100% while prep/decode/OS still need cycles.
    """
    n = max(1, n if n is not None else max_cpu_threads())
    prev_torch = torch.get_num_threads()
    prev_interop = torch.get_num_interop_threads() if hasattr(torch, "get_num_interop_threads") else None
    prev_cv = None
    try:
        try:
            prev_cv = cv2.getNumThreads()
            cv2.setNumThreads(n)
        except Exception:
            prev_cv = None
        torch.set_num_threads(n)
        if prev_interop is not None:
            try:
                torch.set_num_interop_threads(max(1, min(n, 4)))
            except RuntimeError:
                pass
        yield n
    finally:
        try:
            torch.set_num_threads(prev_torch)
        except Exception:
            pass
        if prev_cv is not None:
            try:
                cv2.setNumThreads(prev_cv)
            except Exception:
                pass


def _mps_budget_frac() -> float:
    try:
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            return 0.0
        alloc = torch.mps.current_allocated_memory()
        rec = torch.mps.recommended_max_memory()
        if rec <= 0:
            return 0.0
        return float(alloc) / float(rec)
    except Exception:
        return 0.0


class DitDiskCache:
    """Float16 mmap blob for OOM fallback park/unpark (not used every window)."""

    def __init__(self, cache_dir: Path, dtype: torch.dtype):
        self.cache_dir = Path(cache_dir)
        self.dtype = dtype
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mmap: np.memmap | None = None
        self._index: dict | None = None

    @property
    def index_path(self) -> Path:
        return self.cache_dir / "index.json"

    @property
    def blob_path(self) -> Path:
        return self.cache_dir / DIT_BLOB_NAME

    @property
    def ready(self) -> bool:
        if not self.index_path.exists() or not self.blob_path.exists():
            return False
        try:
            meta = json.loads(self.index_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        return int(meta.get("version", 0)) == DIT_CACHE_VERSION and "tensors" in meta

    def _is_meta(self, module: torch.nn.Module) -> bool:
        for p in module.parameters():
            return bool(p.is_meta)
        return True

    def _free_disk_gb(self) -> float:
        st = os.statvfs(self.cache_dir)
        return (st.f_bavail * st.f_frsize) / (1024**3)

    def clear(self) -> None:
        self._mmap = None
        self._index = None
        if not self.cache_dir.exists():
            return
        for p in self.cache_dir.iterdir():
            try:
                p.unlink()
            except OSError:
                pass

    def _load_index(self) -> dict:
        if self._index is not None:
            return self._index
        self._index = json.loads(self.index_path.read_text())
        return self._index

    def save_sharded(self, module: torch.nn.Module) -> None:
        if self.ready:
            print(f"Using existing DiT mmap cache at {self.cache_dir}", flush=True)
            return

        # Drop legacy per-tensor .npy caches and partial writes.
        self.clear()
        free_gb = self._free_disk_gb()
        print(
            f"Writing DiT float16 mmap blob to {self.cache_dir} "
            f"(free disk ≈ {free_gb:.1f}GB)...",
            flush=True,
        )
        if free_gb < 3.0:
            raise RuntimeError(
                f"Only {free_gb:.1f}GB free on disk (need ~3GB+ for DiT mmap). "
                "Free space, then retry. You can also delete old dit_cache_* folders."
            )

        # First pass: sizes only (cast one tensor at a time when writing).
        state_items = list(module.state_dict().items())
        total_nbytes = 0
        for _name, tensor in state_items:
            total_nbytes += int(tensor.numel()) * 2  # stored as float16

        tensors_meta: dict[str, dict] = {}
        offset = 0
        try:
            with open(self.blob_path, "wb") as fh:
                fh.truncate(total_nbytes)
            mm = np.memmap(
                self.blob_path, dtype=np.float16, mode="r+", shape=(total_nbytes // 2,)
            )
            for i, (name, tensor) in enumerate(state_items):
                t = tensor.detach().to(device="cpu").contiguous()
                if t.dtype != torch.float16:
                    t = t.to(torch.float16)
                arr = t.numpy().reshape(-1)
                n = int(arr.size)
                mm[offset // 2 : offset // 2 + n] = arr
                tensors_meta[name] = {
                    "offset": offset,
                    "nbytes": n * 2,
                    "shape": list(t.shape),
                    "dtype": "float16",
                }
                offset += n * 2
                del arr, t
                if i % 50 == 0:
                    print(f"  packed {i} tensors...", flush=True)
                    mm.flush()
                    free_memory()
            mm.flush()
            del mm
            del state_items
            index = {
                "version": DIT_CACHE_VERSION,
                "blob": DIT_BLOB_NAME,
                "dtype": "float16",
                "tensors": tensors_meta,
            }
            self.index_path.write_text(json.dumps(index))
            self._index = index
        except Exception:
            print("DiT mmap write failed — cleaning partial cache.", flush=True)
            self.clear()
            raise

        print(
            f"DiT mmap complete ({len(tensors_meta)} tensors, "
            f"{total_nbytes / (1024**3):.2f}GB).",
            flush=True,
        )
        free_memory()

    def park(self, module: torch.nn.Module) -> None:
        self.save_sharded(module)
        if self._is_meta(module):
            return
        print("Parking DiT on meta device to free RAM (OOM fallback)...", flush=True)
        self._mmap = None
        module.to("meta")
        free_memory()

    def unpark(self, module: torch.nn.Module) -> None:
        if not self._is_meta(module):
            return
        if not self.ready:
            raise RuntimeError(f"DiT mmap cache missing at {self.cache_dir}")

        print("Reloading DiT from float16 mmap into RAM...", flush=True)
        t0 = time.time()
        index = self._load_index()
        tensors_meta = index["tensors"]
        blob = self.blob_path
        flat = np.memmap(blob, dtype=np.float16, mode="r")
        self._mmap = flat
        module.to_empty(device="cpu")
        with torch.no_grad():
            params = dict(module.named_parameters())
            buffers = dict(module.named_buffers())
            for i, (name, meta) in enumerate(tensors_meta.items()):
                off = int(meta["offset"]) // 2
                n = int(meta["nbytes"]) // 2
                shape = tuple(meta["shape"])
                arr = np.array(flat[off : off + n], dtype=np.float16, copy=True).reshape(
                    shape
                )
                tensor = torch.from_numpy(arr)
                if name in params:
                    target = params[name]
                    if target.dtype != tensor.dtype:
                        tensor = tensor.to(dtype=target.dtype)
                    target.copy_(tensor)
                elif name in buffers:
                    target = buffers[name]
                    if target.dtype != tensor.dtype:
                        tensor = tensor.to(dtype=target.dtype)
                    target.copy_(tensor)
                del arr, tensor
                if i % 80 == 0:
                    free_memory()
        module.eval()
        free_memory()
        print(f"DiT reload done in {time.time() - t0:.1f}s.", flush=True)


_PINNABLE_DIT_CHILDREN = (
    "patch_embedding",
    "text_embedding",
    "time_embedding",
    "time_projection",
    "head",
    "img_emb",
    "ref_conv",
    "control_adapter",
)


def install_dit_block_swap(
    dit: torch.nn.Module,
    compute_device: torch.device,
    dit_cache: DitDiskCache | None = None,
    verbose: bool = False,
    max_resident_blocks: int = 3,
    release_vae=None,
) -> None:
    """Keep DiT on CPU; keep a few blocks on MPS to amortize transfers.

    Prefetches the next block into a spare resident slot, pins small non-block
    modules on the compute device, and recovers from OOM by shrinking residency.
    """
    if dit_cache is None or not dit_cache._is_meta(dit):
        dit.cpu()
    free_memory(compute_device)
    state: dict = {
        "resident": [],
        "pinned": set(),
        "max_resident": max(1, int(max_resident_blocks)),
        "cap": max(1, int(max_resident_blocks)),
    }
    blocks_mod = getattr(dit, "blocks", None)
    n_blocks = len(blocks_mod) if blocks_mod is not None else 0

    def _release_vae_for_dit() -> None:
        if callable(release_vae):
            try:
                release_vae()
            except Exception:
                pass

    def _try_to_device(module: torch.nn.Module) -> bool:
        try:
            module.to(compute_device)
            return True
        except Exception as exc:
            if not _is_oom_error(exc):
                raise
            return False

    def _pin_small_modules() -> None:
        if compute_device.type == "cpu":
            return
        for name in _PINNABLE_DIT_CHILDREN:
            child = getattr(dit, name, None)
            if child is None or not isinstance(child, torch.nn.Module):
                continue
            if child in state["pinned"]:
                continue
            if _try_to_device(child):
                state["pinned"].add(child)
            else:
                free_memory(compute_device)
                try:
                    child.cpu()
                except Exception:
                    pass

    def _adapt_max_resident() -> None:
        if compute_device.type != "mps":
            return
        frac = _mps_budget_frac()
        if frac > 0.90:
            state["max_resident"] = 1
        elif frac < 0.55:
            state["max_resident"] = min(state["cap"], max(state["max_resident"], 2))

    def _evict_oldest() -> None:
        while len(state["resident"]) >= state["max_resident"]:
            old = state["resident"].pop(0)
            try:
                old.cpu()
            except Exception:
                pass

    def _ensure_block(module: torch.nn.Module) -> None:
        if module in state["resident"]:
            state["resident"] = [m for m in state["resident"] if m is not module]
            state["resident"].append(module)
            return

        _adapt_max_resident()
        _evict_oldest()
        if not _try_to_device(module):
            # OOM: shrink to 1, flush, retry once.
            state["max_resident"] = 1
            for old in list(state["resident"]):
                try:
                    old.cpu()
                except Exception:
                    pass
            state["resident"].clear()
            free_memory(compute_device)
            module.to(compute_device)
        state["resident"].append(module)

    def _prefetch_next(block_idx: int) -> None:
        if (
            blocks_mod is None
            or block_idx + 1 >= n_blocks
            or state["max_resident"] <= 1
            or len(state["resident"]) >= state["max_resident"]
        ):
            return
        if compute_device.type == "mps" and _mps_budget_frac() >= 0.80:
            return
        nxt = blocks_mod[block_idx + 1]
        if nxt in state["resident"]:
            return
        if not _try_to_device(nxt):
            free_memory(compute_device)
            state["max_resident"] = max(1, state["max_resident"] - 1)
            return
        state["resident"].append(nxt)

    def _ensure_non_block(module: torch.nn.Module) -> None:
        # Never flush DiT block residents for small modules.
        if module in state["pinned"]:
            # Re-pin after flush_resident may have moved them to CPU.
            try:
                p = next(module.parameters(), None)
                if p is not None and p.device.type != compute_device.type:
                    if not _try_to_device(module):
                        state["pinned"].discard(module)
                        module.to(compute_device)
            except StopIteration:
                pass
            return
        if not _try_to_device(module):
            free_memory(compute_device)
            module.to(compute_device)

    def wrap_block(module: torch.nn.Module, block_idx: int, label: str):
        orig_forward = module.forward

        def wrapped_forward(*args, **kwargs):
            _release_vae_for_dit()
            if dit_cache is not None:
                dit_cache.unpark(dit)
            _pin_small_modules()
            if verbose:
                print(f"    DiT {label} -> {compute_device}", flush=True)
            _ensure_block(module)
            _prefetch_next(block_idx)
            args = _move_tensors(args, compute_device)
            kwargs = _move_tensors(kwargs, compute_device)
            return orig_forward(*args, **kwargs)

        module.forward = wrapped_forward  # type: ignore[method-assign]

    def wrap_non_block(module: torch.nn.Module, label: str):
        orig_forward = module.forward

        def wrapped_forward(*args, **kwargs):
            _release_vae_for_dit()
            if dit_cache is not None:
                dit_cache.unpark(dit)
            _pin_small_modules()
            if verbose:
                print(f"    DiT {label} -> {compute_device}", flush=True)
            _ensure_non_block(module)
            args = _move_tensors(args, compute_device)
            kwargs = _move_tensors(kwargs, compute_device)
            out = orig_forward(*args, **kwargs)
            # Pinned modules stay on compute device; unpinned bounce back.
            if module not in state["pinned"]:
                try:
                    module.cpu()
                except Exception:
                    pass
            return out

        module.forward = wrapped_forward  # type: ignore[method-assign]

    for name, child in dit.named_children():
        if name == "blocks":
            for i, block in enumerate(child):
                wrap_block(block, i, f"blocks[{i}/{n_blocks - 1}]")
        else:
            wrap_non_block(child, name)

    _pin_small_modules()

    def flush_resident() -> None:
        for m in list(state["resident"]):
            try:
                m.cpu()
            except Exception:
                pass
        state["resident"].clear()
        for m in list(state["pinned"]):
            try:
                m.cpu()
            except Exception:
                pass
        # Keep pinned set so we re-pin lazily on next DiT forward.

    dit._dvd_flush_resident = flush_resident  # type: ignore[attr-defined]

    print(
        f"Installed DiT multi-block swap -> {compute_device} "
        f"({n_blocks} blocks, max_resident={max_resident_blocks}, "
        f"pinned={len(state['pinned'])}).",
        flush=True,
    )


def configure_vae_on_cpu(
    pipe,
    dtype: torch.dtype,
    dit_cache: DitDiskCache | None = None,
    *,
    dit_resident: bool = True,
    accel_device: torch.device | None = None,
) -> None:
    """VAE float16 with DiT-aware device selection.

    While DiT is parked (meta), VAE prefers MPS and stays sticky across
    consecutive VAE calls. VAE is forced back to CPU before DiT unparks.
    DiT is only parked to mmap when RAM is truly tight (not always on 8GB).
    """
    if pipe.vae is None:
        return
    from resources import available_ram_bytes, max_cpu_threads

    vae_dtype = torch.float16 if dtype in (torch.float16, torch.bfloat16) else dtype
    pipe.vae.to(device="cpu", dtype=vae_dtype)
    pipe.vae.eval()
    orig_encode = pipe.vae.encode
    orig_decode = pipe.vae.decode
    low_mem_machine = system_memory_gb() <= 8.5
    mps_ok = (
        accel_device is not None
        and accel_device.type == "mps"
        and bool(getattr(torch.backends, "mps", None))
        and torch.backends.mps.is_available()
    )
    # Comfortable headroom to keep DiT CPU-resident across VAE (avoid mmap reload).
    _park_ram_floor = int((1.8 if low_mem_machine else 2.5) * 1024**3)
    state = {"vae_on_mps": False}

    def _to_vae_dtype(videos, device: torch.device):
        if torch.is_tensor(videos):
            return videos.to(device=device, dtype=vae_dtype)
        if isinstance(videos, (list, tuple)):
            return [
                v.to(device=device, dtype=vae_dtype) if torch.is_tensor(v) else v
                for v in videos
            ]
        return videos

    def _dit_parked() -> bool:
        return (
            dit_cache is not None
            and pipe.dit is not None
            and dit_cache._is_meta(pipe.dit)
        )

    def _move_vae(dev: torch.device) -> None:
        pipe.vae.to(device=dev, dtype=vae_dtype)
        pipe.vae.eval()
        state["vae_on_mps"] = bool(mps_ok and dev.type == "mps")

    def release_vae_for_dit() -> None:
        """Call before DiT uses MPS so VAE weights do not fight block swap."""
        if pipe.vae is None:
            return
        if state["vae_on_mps"] or (
            mps_ok
            and any(p.device.type == "mps" for p in pipe.vae.parameters())
        ):
            _move_vae(torch.device("cpu"))
            if mps_ok:
                free_memory(accel_device)
            state["vae_on_mps"] = False

    def _maybe_park_for_vae(reason: str) -> bool:
        if pipe.dit is None or dit_cache is None:
            return False
        flush = getattr(pipe.dit, "_dvd_flush_resident", None)
        if callable(flush):
            flush()
        if dit_cache._is_meta(pipe.dit):
            return False
        avail = available_ram_bytes()
        # Prefer keeping DiT CPU-resident across VAE when RAM allows — avoids
        # expensive mmap unpark on the next DiT forward.
        if dit_resident and avail >= _park_ram_floor:
            return False
        if not dit_resident or avail < _park_ram_floor:
            print(
                f"    Parking DiT before VAE ({reason}; "
                f"avail_ram={avail / (1024**3):.2f} GiB)...",
                flush=True,
            )
            dit_cache.park(pipe.dit)
            free_memory(accel_device if mps_ok else None)
            return True
        return False

    def _vae_device(_op: str = "encode") -> torch.device:
        # Prefer MPS when DiT is parked (or was just flushed and weights are
        # still on CPU only — MPS free for VAE). If DiT stays resident in RAM,
        # still try MPS when budget is low-pressure; else CPU.
        if not mps_ok:
            return torch.device("cpu")
        if _dit_parked():
            return accel_device  # type: ignore[return-value]
        # DiT resident on CPU: MPS is free for VAE weights/activations.
        return accel_device  # type: ignore[return-value]

    def _vae_thread_budget() -> int:
        n = max_cpu_threads()
        if _dit_parked():
            return n
        avail = available_ram_bytes()
        if low_mem_machine or avail < _park_ram_floor:
            return max(2, min(n, 4))
        return n

    def _finish_vae_call(*, force_cpu: bool) -> None:
        """Keep VAE sticky on MPS while DiT is parked; otherwise free MPS."""
        if pipe.vae is None:
            return
        if force_cpu or not _dit_parked() or not state["vae_on_mps"]:
            if state["vae_on_mps"] or force_cpu:
                _move_vae(torch.device("cpu"))
            if mps_ok:
                free_memory(accel_device)
            state["vae_on_mps"] = False

    def encode(videos, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        _maybe_park_for_vae("encode")
        if torch.is_tensor(videos):
            shape = tuple(videos.shape)
        elif isinstance(videos, (list, tuple)) and videos:
            shape = ("list", len(videos), tuple(videos[0].shape))
        else:
            shape = type(videos)
        vae_dev = _vae_device("encode")
        print(
            f"    VAE encode on {vae_dev.type} float16 "
            f"(resident_dit={dit_resident}, dit_parked={_dit_parked()}, "
            f"sticky={state['vae_on_mps']}) shape={shape} ...",
            flush=True,
        )

        def _run(dev: torch.device):
            _move_vae(dev)
            vids = _to_vae_dtype(videos, dev)
            return orig_encode(
                vids,
                device=dev,
                tiled=False,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(device="cpu", dtype=pipe.torch_dtype)

        t0 = time.time()
        used = vae_dev.type
        try:
            if vae_dev.type == "mps":
                try:
                    out = _run(vae_dev)
                except Exception as exc:
                    if not _is_oom_error(exc):
                        raise
                    print(
                        f"    VAE encode MPS OOM ({exc}); falling back to CPU...",
                        flush=True,
                    )
                    _move_vae(torch.device("cpu"))
                    free_memory(vae_dev)
                    used = "cpu_fallback"
                    with cpu_heavy_threads(_vae_thread_budget()):
                        out = _run(torch.device("cpu"))
            else:
                with cpu_heavy_threads(_vae_thread_budget()):
                    out = _run(torch.device("cpu"))
        finally:
            # Sticky while DiT parked (next VAE op can reuse MPS residency).
            # DiT forward will call release_vae_for_dit before unpark.
            _finish_vae_call(force_cpu=used == "cpu_fallback")

        enc_s = time.time() - t0
        print(f"    VAE encode done in {enc_s:.1f}s ({used}).", flush=True)
        return out

    def decode(hidden_states, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        _maybe_park_for_vae("decode")
        vae_dev = _vae_device("decode")
        print(
            f"    VAE decode on {vae_dev.type} float16 "
            f"(dit_parked={_dit_parked()}, sticky={state['vae_on_mps']})...",
            flush=True,
        )

        def _run(dev: torch.device):
            _move_vae(dev)
            hs = _to_vae_dtype(hidden_states, dev)
            return orig_decode(
                hs,
                device=dev,
                tiled=False,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )

        t0 = time.time()
        used = vae_dev.type
        try:
            if vae_dev.type == "mps":
                try:
                    out = _run(vae_dev)
                except Exception as exc:
                    if not _is_oom_error(exc):
                        raise
                    print(
                        f"    VAE decode MPS OOM ({exc}); falling back to CPU...",
                        flush=True,
                    )
                    _move_vae(torch.device("cpu"))
                    free_memory(vae_dev)
                    used = "cpu_fallback"
                    with cpu_heavy_threads(_vae_thread_budget()):
                        out = _run(torch.device("cpu"))
            else:
                with cpu_heavy_threads(_vae_thread_budget()):
                    out = _run(torch.device("cpu"))
        finally:
            # End of VAE phase for the window — free MPS for next DiT/prep.
            _finish_vae_call(force_cpu=True)

        # Keep decode outputs on CPU for post.
        if torch.is_tensor(out):
            out = out.to(device="cpu")
        elif isinstance(out, (list, tuple)):
            out = [o.to(device="cpu") if torch.is_tensor(o) else o for o in out]

        dec_s = time.time() - t0
        print(f"    VAE decode done in {dec_s:.1f}s ({used}).", flush=True)
        return out

    pipe.vae.encode = encode  # type: ignore[method-assign]
    pipe.vae.decode = decode  # type: ignore[method-assign]
    pipe._dvd_release_vae_for_dit = release_vae_for_dit  # type: ignore[attr-defined]
    print(
        f"VAE: sticky MPS while DiT parked (encode+decode; CPU fallback on OOM); "
        f"accel={accel_device}, park_ram_floor={_park_ram_floor / (1024**3):.1f}GiB.",
        flush=True,
    )


def load_model(
    ckpt_dir: Path,
    yaml_args,
    device: torch.device,
    dtype: torch.dtype,
    pure_cpu: bool,
    cache_root: Path,
    *,
    dit_resident: bool = True,
    max_resident_blocks: int = 3,
):
    print("Initializing DVD / Wan backbone (CPU)...", flush=True)
    accelerator = Accelerator()
    model = WanTrainingModule(
        accelerator=accelerator,
        model_id_with_origin_paths=yaml_args.model_id_with_origin_paths,
        trainable_models=None,
        use_gradient_checkpointing=False,
        lora_rank=yaml_args.lora_rank,
        lora_base_model=yaml_args.lora_base_model,
        args=yaml_args,
    )

    ckpt_path = ckpt_dir / "dvd_1.1.safetensors"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint at {ckpt_path}. "
            "Run: python scripts/download_weights.py"
        )

    print(f"Loading checkpoint weights from {ckpt_path} ...", flush=True)
    state_dict = load_file(str(ckpt_path), device="cpu")
    dit_state_dict = {
        k.replace("pipe.dit.", ""): v
        for k, v in state_dict.items()
        if "pipe.dit." in k
    }
    del state_dict
    free_memory()

    model.pipe.dit.load_state_dict(dit_state_dict, strict=True)
    del dit_state_dict
    free_memory()

    print("Merging LoRA layers...", flush=True)
    model.merge_lora_layer()
    print("Baking LoRA into base Linear layers...", flush=True)
    replaced = bake_and_unload_lora(model.pipe.dit)
    print(f"Replaced {replaced} LoRA modules with plain Linear.", flush=True)
    free_memory()

    print(f"Casting DiT weights to {dtype} on CPU (low-mem)...", flush=True)
    cast_pipe_dtype_low_mem(model.pipe, dtype, only=("dit",))
    cast_dit_freqs_for_mps(model.pipe.dit)
    free_memory()

    print("Dropping unused pipeline modules...", flush=True)
    drop_unused_pipe_modules(model.pipe)

    dit_dir = resolve_dit_cache_dir(
        cache_root,
        str(dtype).replace("torch.", ""),
        project_root=REPO_ROOT,
    )
    dit_cache = DitDiskCache(dit_dir, dtype=dtype)
    # Build mmap fallback once; keep DiT resident in RAM for the hot path.
    dit_cache.save_sharded(model.pipe.dit)
    if not dit_resident:
        dit_cache.park(model.pipe.dit)
    else:
        print("DiT kept resident in RAM (mmap is OOM fallback only).", flush=True)

    model.pipe.device = torch.device("cpu")
    model.pipe.torch_dtype = dtype
    compute_device = torch.device("cpu") if (pure_cpu or device.type == "cpu") else device
    configure_vae_on_cpu(
        model.pipe,
        dtype=dtype,
        dit_cache=dit_cache,
        dit_resident=dit_resident,
        accel_device=compute_device if compute_device.type != "cpu" else None,
    )
    install_dit_block_swap(
        model.pipe.dit,
        compute_device=compute_device,
        dit_cache=dit_cache,
        verbose=False,
        max_resident_blocks=1 if pure_cpu else max_resident_blocks,
        release_vae=getattr(model.pipe, "_dvd_release_vae_for_dit", None),
    )

    model.eval()
    free_memory(device if device.type == "mps" else None)
    print(
        f"Model ready (DiT resident={dit_resident}; compute={compute_device}; "
        f"VAE on CPU float16; mmap={dit_dir}).",
        flush=True,
    )
    return model


def generate_depth_from_video(
    model,
    video_path: str | Path,
    out_h: int,
    out_w: int,
    dtype: torch.dtype,
    window_size: int = 9,
    overlap: int = 4,
    scale_only: bool = False,
    device: torch.device | None = None,
    max_frames: int | None = None,
    *,
    upsample: bool = True,
    upsample_params: UpsampleParams | None = None,
    upsample_workers: str | int = "auto",
    pipeline_parallel: bool = True,
    cache_root: Path,
    keep_upsample_cache: bool = False,
    probe: tuple[float, int, int, int] | None = None,
    shot_ranges: tuple[tuple[int, int], ...] | None = None,
) -> tuple[np.ndarray | None, float, tuple[int, int], ParallelUpscaler | None]:
    """Windowed inference with optional prep|infer|post resource-aware pipeline.

    When ``upsample`` is True, returns ``(None, fps, orig_size, upscaler)`` so the
    caller can free the DVD model before ``upscaler.finish()`` + streaming export
    (full-res frames stay on the float16 memmap — never a ~20 GiB float32 stack).
    """
    if probe is None:
        fps, total_frames, orig_h, orig_w = probe_video(video_path)
    else:
        fps, total_frames, orig_h, orig_w = probe
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    print(
        f"Video probe: using {total_frames} frames @ {fps:.2f}fps, "
        f"src={orig_w}x{orig_h} -> infer={out_w}x{out_h}",
        flush=True,
    )

    governor = ResourceGovernor(
        upsample_workers=upsample_workers,
        frame_w=orig_w,
        frame_h=orig_h,
        infer_w=out_w,
        infer_h=out_h,
        window_frames=window_size,
    )
    if shot_ranges is None:
        shot_ranges = tuple(
            (s0, min(s1, total_frames))
            for s0, s1 in probe_shot_ranges(video_path)
            if s0 < total_frames and min(s1, total_frames) > s0
        ) or ((0, total_frames),)
    else:
        shot_ranges = tuple(
            (s0, min(s1, total_frames))
            for s0, s1 in shot_ranges
            if s0 < total_frames and min(s1, total_frames) > s0
        ) or ((0, total_frames),)
    stabilize_params = derive_stabilize_params(
        fps=fps,
        process_w=out_w,
        process_h=out_h,
        frame_count=total_frames,
    )
    stream_chunk_frames = min(
        stabilize_params.chunk_frames,
        max(window_size * 2, 12),
    )
    stream_chunk_overlap = min(
        stabilize_params.chunk_overlap,
        max(4, stream_chunk_frames // 4),
    )
    streaming_stabilize_params = StabilizeParams(
        flow_long_side=stabilize_params.flow_long_side,
        farneback_levels=stabilize_params.farneback_levels,
        farneback_winsize=stabilize_params.farneback_winsize,
        lock_percentile=stabilize_params.lock_percentile,
        temporal=stabilize_params.temporal,
        chunk_frames=stream_chunk_frames,
        chunk_overlap=stream_chunk_overlap,
    )

    bgr_store = InferResBgrStore(out_h, out_w)
    run_upsample = bool(upsample)
    if pipeline_parallel:
        print(
            f"[stabilize] streaming chunks={streaming_stabilize_params.chunk_frames} "
            f"overlap={streaming_stabilize_params.chunk_overlap} "
            f"(base chunks={stabilize_params.chunk_frames})",
            flush=True,
        )
        if keep_upsample_cache:
            up_dir = up_cache_dir_for_video(
                cache_root,
                video_path,
                total_frames=total_frames,
                out_w=orig_w,
                out_h=orig_h,
            )
        else:
            up_dir = up_cache_dir(cache_root)
        depth, upscaler = run_window_pipeline(
            model,
            video_path=video_path,
            out_h=out_h,
            out_w=out_w,
            dtype=dtype,
            window_size=window_size,
            overlap=overlap,
            total_frames=total_frames,
            orig_h=orig_h,
            orig_w=orig_w,
            device=device,
            free_memory=free_memory,
            streaming_reader_cls=StreamingWindowReader,
            governor=governor,
            upsample=run_upsample,
            upsample_params=upsample_params,
            scale_only=scale_only,
            pipeline_parallel=True,
            shot_ranges=shot_ranges,
            stabilize_params=streaming_stabilize_params,
            cache_dir=up_dir if run_upsample else None,
            keep_upsample_cache=keep_upsample_cache,
            bgr_store=bgr_store,
        )
        return depth, fps, (orig_h, orig_w), upscaler

    depth, upscaler = run_window_pipeline(
        model,
        video_path=video_path,
        out_h=out_h,
        out_w=out_w,
        dtype=dtype,
        window_size=window_size,
        overlap=overlap,
        total_frames=total_frames,
        orig_h=orig_h,
        orig_w=orig_w,
        device=device,
        free_memory=free_memory,
        streaming_reader_cls=StreamingWindowReader,
        governor=governor,
        upsample=False,
        upsample_params=upsample_params,
        scale_only=scale_only,
        pipeline_parallel=False,
        cache_dir=None,
        keep_upsample_cache=keep_upsample_cache,
        bgr_store=bgr_store,
    )
    if depth is None:
        raise RuntimeError("Expected in-memory depth stack for stabilization.")
    depth = depth[0]
    print("[stabilize] applying DA3-style band lock + temporal median...", flush=True)
    depth = stabilize_depth_video(
        video_path,
        depth,
        fps=fps,
        process_h=out_h,
        process_w=out_w,
        shot_ranges=shot_ranges,
        bgr_store=bgr_store,
    )
    if not run_upsample:
        return depth[None], fps, (orig_h, orig_w), None

    if keep_upsample_cache:
        up_dir = up_cache_dir_for_video(
            cache_root,
            video_path,
            total_frames=total_frames,
            out_w=orig_w,
            out_h=orig_h,
        )
    else:
        up_dir = up_cache_dir(cache_root)
    plan0 = governor.plan()
    params = upsample_params or default_upsample_params(out_h, out_w, orig_h, orig_w)
    upscaler = ParallelUpscaler(
        video_path,
        out_size=(orig_w, orig_h),
        params=params,
        total_frames=total_frames,
        workers=plan0.upsample_workers,
        cache_dir=up_dir,
        keep_cache=keep_upsample_cache,
    )
    upscaler.submit_range(depth[None], 0, total_frames)
    return None, fps, (orig_h, orig_w), upscaler


def _depth_stack_to_disparity_frames(depth: np.ndarray) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for i in range(depth.shape[0]):
        frame = np.asarray(depth[i], dtype=np.float32)
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[..., 0]
        elif frame.ndim == 3:
            frame = frame.mean(axis=-1)
        frames.append(frame.astype(np.float32, copy=False))
    return frames


def _replace_depth_stack_from_disparities(depth: np.ndarray, disparities: list[np.ndarray]) -> np.ndarray:
    out = np.asarray(depth, dtype=np.float32).copy()
    for i, disp in enumerate(disparities):
        if out[i].ndim == 3:
            out[i, ..., :] = disp[..., None]
        else:
            out[i] = disp
    return out


def stabilize_depth_video(
    video_path: str | Path,
    depth: np.ndarray,
    *,
    fps: float,
    process_h: int,
    process_w: int,
    shot_ranges: tuple[tuple[int, int], ...] | None = None,
    bgr_store=None,
) -> np.ndarray:
    """Apply DA3-style shot band-lock + flow temporal median."""
    total_frames = depth.shape[0]
    if shot_ranges is None:
        ranges = probe_shot_ranges(video_path)
    else:
        ranges = list(shot_ranges)
    disparities = _depth_stack_to_disparity_frames(depth)
    params = derive_stabilize_params(
        fps=fps,
        process_w=process_w,
        process_h=process_h,
        frame_count=total_frames,
    )
    stabilized = [d.copy() for d in disparities]
    for shot_idx, (s0, s1) in enumerate(ranges):
        if s0 >= total_frames:
            break
        s1 = min(s1, total_frames)
        if s1 <= s0:
            continue
        print(f"[stabilize] shot {shot_idx + 1}/{len(ranges)} frames [{s0}:{s1})", flush=True)
        shot = [np.asarray(d, dtype=np.float32) for d in stabilized[s0:s1]]
        print(
            f"[stabilize] band-lock + Farneback temporal median on {s1 - s0} frames "
            f"(CPU — can take several minutes on Colab; progress prints below)...",
            flush=True,
        )
        shot = apply_shot_band_lock(shot)
        shot = apply_flow_temporal_median(
            video_path,
            shot,
            process_w=process_w,
            process_h=process_h,
            params=params,
            global_start=s0,
            bgr_store=bgr_store,
        )
        stabilized[s0:s1] = shot
    return _replace_depth_stack_from_disparities(depth, stabilized)


def _depth_output_path(args) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(args.input_video).stem
    return output_dir / f"{base_name}_depth_gray.mp4"


def save_grayscale_depth(depth: np.ndarray, origin_fps: float, args) -> Path:
    """Normalize + encode an in-memory depth stack (inference resolution)."""
    output_path = _depth_output_path(args)
    frames = _depth_stack_to_disparity_frames(depth)
    lo, hi = da3_lock_range(frames, percentile=2.0)
    print(f"Saving grayscale depth video -> {output_path}", flush=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), fourcc, float(origin_fps), (w, h), isColor=True)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {output_path}")
    try:
        for i, frame in enumerate(frames):
            writer.write(normalize_disparity_to_bgr_u8(frame, lo, hi))
            done = i + 1
            if done % 32 == 0 or done == len(frames):
                print(f"  [encode] {done}/{len(frames)}", flush=True)
    finally:
        writer.release()
    return output_path


def save_grayscale_depth_from_upscaler(
    upscaler: ParallelUpscaler, origin_fps: float, args
) -> Path:
    """Stream-encode native-res depth from the upsample memmap (8GB-safe)."""
    return upscaler.export_grayscale_mp4(
        _depth_output_path(args), origin_fps, quality=6
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a grayscale depth-map video with DVD (MPS-safe on 8GB)"
    )
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--ckpt", type=Path, default=REPO_ROOT / "ckpt")
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Root for runtime caches (DiT shards, upsample memmap). "
            "Default: $DVD_CACHE_DIR, else auto-pick a mounted FAT32 volume "
            "under /Volumes (…/dvd_cache). Hugging Face models stay in the project."
        ),
    )
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu", "auto"])
    parser.add_argument(
        "--dtype", default="float16", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force full CPU inference (most reliable on 8GB Macs)",
    )
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Only process the first N frames (recommended smoke test: 45)",
    )
    parser.add_argument(
        "--no-upsample",
        action="store_true",
        help="Skip RGB-guided JBU upscale (keep inference resolution)",
    )
    parser.add_argument(
        "--upsample-workers",
        default="auto",
        help="CPU threads for JBU upscale (auto|N). Default: auto from RAM/CPU",
    )
    parser.add_argument(
        "--keep-upsample-cache",
        action="store_true",
        help=(
            "Keep the float16 upsample memmap for this video under a stable "
            "up_cache_v_* folder and reuse it on later runs (skips slow FAT32 "
            "re-allocation). Delete the folder manually to force a fresh cache."
        ),
    )
    parser.add_argument(
        "--no-pipeline-parallel",
        action="store_true",
        help="Disable prep|infer|post overlap (run windows sequentially)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_cuda_helpers_for_mps()

    device = resolve_device("cpu" if args.cpu else args.device)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    mem_gb = system_memory_gb()
    low_mem = mem_gb <= 8.5
    dit_resident = True
    max_resident_blocks = 2 if low_mem else 3
    height_was_auto = args.height is None
    width_was_auto = args.width is None
    window_was_auto = args.window_size is None
    overlap_was_auto = args.overlap is None

    # Resident DiT + VAE f16: slightly larger 8GB defaults (still 4n+1 friendly).
    if height_was_auto:
        args.height = 256 if low_mem else (320 if device.type == "mps" else 480)
    if width_was_auto:
        args.width = 448 if low_mem else (576 if device.type == "mps" else 640)
    if window_was_auto:
        args.window_size = 9 if low_mem else (17 if device.type == "mps" else 81)
    if overlap_was_auto:
        args.overlap = 3 if low_mem else (5 if device.type == "mps" else 21)

    # window must satisfy 4n+1 after padding helpers; 9 and 17 already do.
    model_config = args.model_config or (args.ckpt / "model_config.yaml")
    if not model_config.exists():
        model_config = REPO_ROOT / "configs" / "model_config.yaml"
    if not model_config.exists():
        model_config = DVD_ROOT / "ckpt" / "model_config.yaml"
    if not model_config.exists():
        raise FileNotFoundError(
            "Missing model_config.yaml under ckpt/, configs/, or vendor/DVD/ckpt/. "
            "Run: python scripts/download_weights.py"
        )
    yaml_args = OmegaConf.load(str(model_config))
    print(f"Model config: {model_config}", flush=True)

    cache_root = resolve_cache_root(args.cache_dir)
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
    os.chdir(REPO_ROOT)

    do_upsample = not args.no_upsample
    pipeline_parallel = not args.no_pipeline_parallel
    cleanup_stale_up_caches(cache_root, keep_pid=os.getpid())
    # Also sweep any leftover caches under the old project outputs/ path.
    cleanup_stale_up_caches(Path(args.output_dir), keep_pid=os.getpid())
    print(
        f"Device: {device} | dtype: {dtype} | RAM: {mem_gb:.1f}GB | "
        f"size={args.height}x{args.width} | "
        f"window={args.window_size} overlap={args.overlap} | "
        f"dit_resident={dit_resident} | max_blocks={max_resident_blocks} | "
        f"upsample={'JBU ' + str(args.upsample_workers) if do_upsample else 'off'}"
        f"{' keep-cache' if getattr(args, 'keep_upsample_cache', False) else ''} | "
        f"pipeline={'parallel' if pipeline_parallel else 'sequential'}",
        flush=True,
    )
    if low_mem and device.type == "mps":
        print(
            "8GB Mac: VAE CPU float16 + resident DiT + multi-block MPS swap. "
            "If killed, retry with --height 192 --width 320 --window-size 5 "
            "--overlap 2, or --cpu.",
            flush=True,
        )

    model = load_model(
        args.ckpt,
        yaml_args,
        device=device,
        dtype=dtype,
        pure_cpu=args.cpu or device.type == "cpu",
        cache_root=cache_root,
        dit_resident=dit_resident,
        max_resident_blocks=max_resident_blocks,
    )

    # Probe once (geometry + shots); reuse inside generate_depth_from_video.
    print("Probing input video (no full decode)...", flush=True)
    fps_probe, total_probe, orig_h, orig_w = probe_video(args.input_video)
    if args.max_frames is not None:
        total_probe = min(total_probe, int(args.max_frames))
    shot_ranges = tuple(
        (s0, min(s1, total_probe))
        for s0, s1 in probe_shot_ranges(args.input_video)
        if s0 < total_probe and min(s1, total_probe) > s0
    ) or ((0, total_probe),)
    if low_mem and not args.no_upsample and height_was_auto and width_was_auto:
        preview_h, preview_w = inference_hw(orig_h, orig_w, args.height, args.width)
        upscale = max(orig_h / max(preview_h, 1), orig_w / max(preview_w, 1))
        if upscale > 6.5:
            args.height = 384
            args.width = 672
            if window_was_auto:
                args.window_size = min(args.window_size, 7)
            if overlap_was_auto:
                args.overlap = min(args.overlap, 3)
            print(
                f"Large native upscale detected ({upscale:.2f}x). "
                f"Raising low-mem inference size to {args.width}x{args.height} "
                f"to reduce blocky depth edges.",
                flush=True,
            )
    out_h, out_w = inference_hw(orig_h, orig_w, args.height, args.width)
    print(f"Inference geometry: {out_w}x{out_h} (from {orig_w}x{orig_h})", flush=True)
    free_memory(device if device.type == "mps" else None)

    upsample_params = None
    if do_upsample:
        upsample_params = default_upsample_params(out_h, out_w, orig_h, orig_w)
        print(
            f"JBU upsample: edge_radius={upsample_params.edge_radius} "
            f"sigma_range={upsample_params.sigma_range} "
            f"(prep|infer|post overlaps; workers from resource governor)",
            flush=True,
        )

    with torch.inference_mode():
        depth, origin_fps, orig_size, upscaler = generate_depth_from_video(
            model,
            args.input_video,
            out_h=out_h,
            out_w=out_w,
            dtype=dtype,
            window_size=args.window_size,
            overlap=args.overlap,
            device=device if device.type in ("mps", "cuda") else None,
            max_frames=args.max_frames,
            upsample=do_upsample,
            upsample_params=upsample_params,
            upsample_workers=args.upsample_workers,
            pipeline_parallel=pipeline_parallel,
            cache_root=cache_root,
            keep_upsample_cache=bool(args.keep_upsample_cache),
            probe=(fps_probe, total_probe, orig_h, orig_w),
            shot_ranges=shot_ranges,
        )
        free_memory(device if device.type == "mps" else None)

    del model
    free_memory()

    if upscaler is not None:
        print("Gathering upsampled frames (model freed)...", flush=True)
        upscaler.finish()
        print(
            f"Depth at native {orig_size[1]}x{orig_size[0]} via guided JBU.",
            flush=True,
        )
        output_path = save_grayscale_depth_from_upscaler(
            upscaler, origin_fps, args
        )
    else:
        depth = depth[0]
        print(
            f"Keeping inference resolution {depth.shape[2]}x{depth.shape[1]} "
            f"(omit --no-upsample for full-res JBU).",
            flush=True,
        )
        output_path = save_grayscale_depth(depth, origin_fps, args)
    print(f"Done: {output_path}", flush=True)


def _format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60.0)
    if minutes < 60:
        return f"{int(minutes)}m {sec:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes}m {sec:04.1f}s"


if __name__ == "__main__":
    t0 = time.perf_counter()
    try:
        main()
    finally:
        elapsed = time.perf_counter() - t0
        print(f"Total time: {_format_elapsed(elapsed)} ({elapsed:.2f}s)", flush=True)
