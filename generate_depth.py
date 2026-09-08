#!/usr/bin/env python3
"""Generate a grayscale depth-map video with Tencent DepthCrafter (CUDA).

DepthCrafter (SVD UNet) runs on NVIDIA GPUs. Defaults target Colab T4 (~15 GiB
VRAM): max-res 512 + model/sequential CPU offload. Post-process reuses the
existing RGB-guided JBU, denoise, and DA3-style shot stabilize stack.
"""

from __future__ import annotations

import argparse
import atexit
import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
DC_ROOT = REPO_ROOT / "vendor" / "DepthCrafter"
if str(DC_ROOT) not in sys.path:
    sys.path.insert(0, str(DC_ROOT))

from da3_stabilize import (  # noqa: E402
    apply_flow_temporal_median,
    apply_shot_band_lock,
    derive_stabilize_params,
    lock_range as da3_lock_range,
    normalize_disparity_to_bgr_u8,
    probe_shot_ranges,
)
from denoise import (  # noqa: E402
    ParallelDenoiser,
    default_denoise_params,
    denoise_depth_stack,
)
from pipeline import (  # noqa: E402
    load_depthcrafter_pipeline,
    release_pipeline,
    run_depthcrafter,
)
from resources import is_low_system_ram, max_cpu_threads, total_ram_bytes  # noqa: E402
from upsample import (  # noqa: E402
    ParallelUpscaler,
    cleanup_stale_up_caches,
    default_upsample_params,
)
from cache_root import up_cache_dir, up_cache_dir_for_video  # noqa: E402

try:
    cv2.setNumThreads(max_cpu_threads())
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass
try:
    torch.set_num_threads(max_cpu_threads())
except Exception:
    pass

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
    for backend in (
        getattr(cv2, "CAP_FFMPEG", 0),
        getattr(cv2, "CAP_AVFOUNDATION", 0),
        0,
    ):
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
    """Return fps, frame_count, orig_h, orig_w (count frames with grab())."""
    cap = open_video_capture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = 0
    while cap.grab():
        frame_count += 1
    _forget_capture(cap)
    if frame_count <= 0:
        raise ValueError(f"No frames found in {video_path}")
    if reported > 0 and frame_count != reported:
        print(
            f"Video probe: readable frames={frame_count} "
            f"(container metadata said {reported})",
            flush=True,
        )
    return fps, frame_count, orig_h, orig_w


def inference_hw(orig_h: int, orig_w: int, max_res: int) -> tuple[int, int]:
    """Long-side ``max_res``, aligned to 64 (SVD / DepthCrafter)."""
    max_res = max(64, int(max_res))
    height = round(orig_h / 64) * 64
    width = round(orig_w / 64) * 64
    if max(height, width) > max_res:
        scale = max_res / max(orig_h, orig_w)
        height = round(orig_h * scale / 64) * 64
        width = round(orig_w * scale / 64) * 64
    height = max(64, int(height))
    width = max(64, int(width))
    return height, width


def free_memory(device: torch.device | None = None) -> None:
    gc.collect()
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_local_cache_root(explicit: Path | None) -> Path:
    """Local project/Colab cache only (no FAT32 auto-pick for CUDA path)."""
    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
    else:
        env = (
            os.environ.get("DEPTHCRAFTER_CACHE_DIR", "").strip()
            or os.environ.get("DVD_CACHE_DIR", "").strip()
        )
        root = Path(env).expanduser().resolve() if env else (REPO_ROOT / ".cache")
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_model_path(
    explicit: Path | str | None,
    *,
    local_candidates: list[Path],
    hf_id: str,
) -> str:
    """Prefer an explicit path, then local ckpt dirs, else Hugging Face id."""
    if explicit is not None:
        path = Path(explicit)
        if path.exists():
            return str(path.resolve())
        # Allow passing a HF id via the flag.
        return str(explicit)
    for cand in local_candidates:
        if cand.exists() and any(cand.iterdir()):
            return str(cand.resolve())
    return hf_id


def cuda_device_info() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    props = torch.cuda.get_device_properties(idx)
    total_gb = props.total_memory / (1024**3)
    free, total = torch.cuda.mem_get_info(idx)
    return (
        f"{name} (cuda:{idx}) | VRAM {total_gb:.1f} GiB total, "
        f"{free / (1024**3):.1f}/{total / (1024**3):.1f} GiB free | "
        f"CUDA {torch.version.cuda}"
    )


def _is_colab() -> bool:
    return bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"))


def _cuda_vram_gib() -> float:
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.total_memory / (1024**3)


def apply_t4_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Colab/T4-safe defaults: max-res 512 + CPU offload + shorter windows."""
    mem_gb = total_ram_bytes() / (1024**3)
    low = is_low_system_ram() or _is_colab()
    vram = _cuda_vram_gib()
    t4_like = vram > 0 and vram <= 16.5

    if args.max_res is None:
        args.max_res = 512 if (low or t4_like) else 1024
    if args.window_size is None:
        if low:
            args.window_size = 40
        elif t4_like:
            args.window_size = 75
        else:
            args.window_size = 110
    if args.overlap is None:
        args.overlap = min(25, max(1, args.window_size // 4))
    if args.cpu_offload == "auto":
        if low or t4_like or _is_colab():
            args.cpu_offload = "sequential" if (low or _is_colab()) else "model"
        elif vram >= 24:
            args.cpu_offload = "none"
        else:
            args.cpu_offload = "model"

    print(
        f"DepthCrafter profile ({mem_gb:.1f} GiB host"
        f"{', Colab' if _is_colab() else ''}; VRAM {vram:.1f} GiB): "
        f"max_res={args.max_res} window={args.window_size} "
        f"overlap={args.overlap} cpu_offload={args.cpu_offload}",
        flush=True,
    )
    return args


def load_infer_frames(
    video_path: str | Path,
    *,
    out_h: int,
    out_w: int,
    total_frames: int,
) -> np.ndarray:
    """Decode frames to float32 RGB [T,H,W,3] in [0,1] at inference resolution."""
    cap = open_video_capture(video_path)
    frames: list[np.ndarray] = []
    try:
        while len(frames) < total_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            if bgr.shape[0] != out_h or bgr.shape[1] != out_w:
                bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frames.append(np.ascontiguousarray(rgb))
            if (len(frames) % 64 == 0) or len(frames) == total_frames:
                print(f"  [decode] {len(frames)}/{total_frames}", flush=True)
    finally:
        _forget_capture(cap)

    if not frames:
        raise ValueError(f"Failed to decode frames from {video_path}")
    if len(frames) < total_frames:
        print(
            f"Decoded {len(frames)} frames (expected {total_frames}); "
            "continuing with readable frames.",
            flush=True,
        )
    return np.stack(frames, axis=0)


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


def _replace_depth_stack_from_disparities(
    depth: np.ndarray, disparities: list[np.ndarray]
) -> np.ndarray:
    out = np.asarray(depth, dtype=np.float32).copy()
    for i, disp in enumerate(disparities):
        if out.ndim == 4 and out[i].ndim == 3:
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
        print(
            f"[stabilize] shot {shot_idx + 1}/{len(ranges)} frames [{s0}:{s1})",
            flush=True,
        )
        shot = [np.asarray(d, dtype=np.float32) for d in stabilized[s0:s1]]
        print(
            f"[stabilize] band-lock + Farneback temporal median on {s1 - s0} frames "
            f"(CPU — can take several minutes on Colab)...",
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
            bgr_store=None,
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
    writer = cv2.VideoWriter(
        str(output_path), fourcc, float(origin_fps), (w, h), isColor=True
    )
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
    """Stream-encode native-res depth from the upsample memmap."""
    return upscaler.export_grayscale_mp4(
        _depth_output_path(args), origin_fps, quality=6
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a grayscale depth-map video with DepthCrafter on CUDA "
            "(T4-oriented defaults)"
        )
    )
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Local root for upsample memmaps "
            "(default: ./.cache or $DEPTHCRAFTER_CACHE_DIR / $DVD_CACHE_DIR)"
        ),
    )
    parser.add_argument(
        "--unet-path",
        type=Path,
        default=None,
        help="Local DepthCrafter UNet dir or HF id (default: ckpt/DepthCrafter)",
    )
    parser.add_argument(
        "--svd-path",
        type=Path,
        default=None,
        help=(
            "Local SVD-XT dir or HF id "
            "(default: ckpt/stable-video-diffusion-img2vid-xt)"
        ),
    )
    parser.add_argument(
        "--max-res",
        type=int,
        default=None,
        help="Long-side inference resolution (default: 512 on T4/Colab, else 1024)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Temporal window (default: 40–110 from host/VRAM profile)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Window overlap (default: min(25, window/4))",
    )
    parser.add_argument(
        "--cpu-offload",
        default="auto",
        choices=["auto", "none", "model", "sequential"],
        help="CPU offload strategy (default: auto from VRAM / Colab)",
    )
    parser.add_argument("--num-inference-steps", type=int, default=5)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Only process the first N frames (smoke test)",
    )
    parser.add_argument(
        "--no-upsample",
        action="store_true",
        help="Skip RGB-guided JBU upscale (keep inference resolution)",
    )
    parser.add_argument(
        "--upsample-workers",
        default="auto",
        help="CPU threads for JBU when --upsample-device cpu (auto|N)",
    )
    parser.add_argument(
        "--upsample-device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Run joint-bilateral upsample on CUDA (default) or CPU",
    )
    parser.add_argument(
        "--jbu-sigma-range",
        type=float,
        default=None,
        help="JBU RGB range sigma (default 0.14; higher = less RGB texture bleed)",
    )
    parser.add_argument(
        "--jbu-edge-strength",
        type=float,
        default=None,
        help="JBU blend strength at edges (default 0.55; lower = less hair, softer edges)",
    )
    parser.add_argument(
        "--jbu-depth-gate",
        type=float,
        default=None,
        help="How strongly JBU is limited to depth edges (default 0.5; lower = more Lanczos)",
    )
    parser.add_argument(
        "--jbu-edge-radius",
        type=int,
        default=None,
        help="JBU neighborhood radius override (default: ~0.75 * upscale)",
    )
    parser.add_argument(
        "--keep-upsample-cache",
        action="store_true",
        help="Reuse stable up_cache_v_* memmap under --cache-dir",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Skip per-frame kernel noise scan + bilateral removal",
    )
    parser.add_argument(
        "--denoise-device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Run noise reduction on CUDA (default) or CPU",
    )
    parser.add_argument(
        "--denoise-workers",
        default="1",
        help="CPU threads for denoise when --denoise-device cpu (auto|N)",
    )
    parser.add_argument(
        "--no-stabilize",
        action="store_true",
        help="Skip DA3-style shot band-lock + temporal median",
    )
    return parser.parse_args()


def _upsample_workers(arg: str | int) -> int:
    if isinstance(arg, int):
        return max(1, arg)
    text = str(arg).strip().lower()
    if text == "auto":
        try:
            return max(1, max_cpu_threads())
        except Exception:
            return max(1, os.cpu_count() or 2)
    return max(1, int(text))


def _denoise_workers(arg: str | int) -> int:
    if isinstance(arg, int):
        return max(1, arg)
    text = str(arg).strip().lower()
    if text == "auto":
        return 1
    return max(1, int(text))


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for DepthCrafter video depth. "
            "Enable a CUDA GPU runtime (e.g. Colab T4)."
        )

    args = apply_t4_defaults(args)
    device = torch.device("cuda")

    cache_root = resolve_local_cache_root(args.cache_dir)
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
    os.chdir(REPO_ROOT)

    unet_path = resolve_model_path(
        args.unet_path,
        local_candidates=[
            REPO_ROOT / "ckpt" / "DepthCrafter",
            REPO_ROOT / "ckpt" / "tencent_DepthCrafter",
        ],
        hf_id="tencent/DepthCrafter",
    )
    svd_path = resolve_model_path(
        args.svd_path,
        local_candidates=[
            REPO_ROOT / "ckpt" / "stable-video-diffusion-img2vid-xt",
            REPO_ROOT / "ckpt" / "stabilityai_stable-video-diffusion-img2vid-xt",
        ],
        hf_id="stabilityai/stable-video-diffusion-img2vid-xt",
    )

    do_upsample = not args.no_upsample
    do_denoise = not args.no_denoise
    do_stabilize = not args.no_stabilize
    cleanup_stale_up_caches(cache_root, keep_pid=os.getpid())
    cleanup_stale_up_caches(Path(args.output_dir), keep_pid=os.getpid())

    offload = None if args.cpu_offload == "none" else args.cpu_offload
    print(
        f"Device: {device} | {cuda_device_info()}\n"
        f"unet={unet_path} | svd={svd_path}\n"
        f"max_res={args.max_res} | window={args.window_size} "
        f"overlap={args.overlap} | steps={args.num_inference_steps} "
        f"guidance={args.guidance_scale} | offload={args.cpu_offload}\n"
        f"upsample={'JBU ' + str(args.upsample_device) if do_upsample else 'off'} | "
        f"denoise={args.denoise_device if do_denoise else 'off'} | "
        f"stabilize={'on' if do_stabilize else 'off'} | "
        f"cache={cache_root}",
        flush=True,
    )

    print("Probing input video...", flush=True)
    fps_probe, total_probe, orig_h, orig_w = probe_video(args.input_video)
    if args.max_frames is not None:
        total_probe = min(total_probe, int(args.max_frames))
    out_h, out_w = inference_hw(orig_h, orig_w, args.max_res)
    print(
        f"Video: {total_probe} frames @ {fps_probe:.2f}fps, "
        f"src={orig_w}x{orig_h} -> infer={out_w}x{out_h}",
        flush=True,
    )

    shot_ranges = tuple(
        (s0, min(s1, total_probe))
        for s0, s1 in probe_shot_ranges(args.input_video)
        if s0 < total_probe and min(s1, total_probe) > s0
    ) or ((0, total_probe),)

    from diffusers.training_utils import set_seed

    set_seed(int(args.seed))

    frames = load_infer_frames(
        args.input_video,
        out_h=out_h,
        out_w=out_w,
        total_frames=total_probe,
    )
    total_frames = int(frames.shape[0])
    free_memory(device)

    pipe = load_depthcrafter_pipeline(
        unet_path=unet_path,
        svd_path=svd_path,
        cpu_offload=offload,
        device="cuda",
    )
    try:
        depth = run_depthcrafter(
            pipe,
            frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            window_size=args.window_size,
            overlap=min(args.overlap, max(0, args.window_size - 1)),
            track_time=False,
        )
    finally:
        release_pipeline(pipe)
        free_memory(device)

    del frames
    free_memory(device)

    if do_stabilize:
        print("[stabilize] applying DA3-style band lock + temporal median...", flush=True)
        depth = stabilize_depth_video(
            args.input_video,
            depth,
            fps=fps_probe,
            process_h=out_h,
            process_w=out_w,
            shot_ranges=shot_ranges,
        )
        free_memory(device)

    upsample_params = None
    if do_upsample:
        upsample_params = default_upsample_params(
            out_h,
            out_w,
            orig_h,
            orig_w,
            edge_radius=args.jbu_edge_radius,
            sigma_range=args.jbu_sigma_range,
            edge_strength=args.jbu_edge_strength,
            depth_gate=args.jbu_depth_gate,
        )
        print(
            f"JBU upsample: device={args.upsample_device} "
            f"edge_radius={upsample_params.edge_radius} "
            f"sigma_range={upsample_params.sigma_range} "
            f"edge_strength={upsample_params.edge_strength} "
            f"depth_gate={upsample_params.depth_gate}",
            flush=True,
        )

    denoise_params = None
    if do_denoise:
        dn_h = orig_h if do_upsample else out_h
        dn_w = orig_w if do_upsample else out_w
        denoise_params = default_denoise_params(dn_h, dn_w)
        print(
            f"Noise reduction: device={args.denoise_device} "
            f"kernel={denoise_params.kernel_size} "
            f"(per-frame scan @ {dn_w}x{dn_h})",
            flush=True,
        )

    if do_upsample and upsample_params is not None:
        if args.keep_upsample_cache:
            up_dir = up_cache_dir_for_video(
                cache_root,
                args.input_video,
                total_frames=total_frames,
                out_w=orig_w,
                out_h=orig_h,
            )
        else:
            up_dir = up_cache_dir(cache_root)
        workers = 1 if args.upsample_device == "cuda" else _upsample_workers(args.upsample_workers)
        upscaler = ParallelUpscaler(
            args.input_video,
            out_size=(orig_w, orig_h),
            params=upsample_params,
            total_frames=total_frames,
            workers=workers,
            cache_dir=up_dir,
            keep_cache=bool(args.keep_upsample_cache),
            device=args.upsample_device,
        )
        # ParallelUpscaler expects [1,T,H,W,C]
        depth_bthwc = depth[..., None][None]
        upscaler.submit_range(depth_bthwc, 0, total_frames)
        del depth, depth_bthwc
        free_memory(device)
        print("Gathering upsampled frames...", flush=True)
        upscaler.finish()
        print(
            f"Depth at native {orig_h}x{orig_w} via guided JBU.",
            flush=True,
        )
        if do_denoise and denoise_params is not None:
            ParallelDenoiser(
                upscaler.frame_store,
                denoise_params,
                workers=_denoise_workers(args.denoise_workers),
                device=args.denoise_device,
            ).run()
        output_path = save_grayscale_depth_from_upscaler(upscaler, fps_probe, args)
    else:
        print(
            f"Keeping inference resolution {out_w}x{out_h} "
            f"(omit --no-upsample for full-res JBU).",
            flush=True,
        )
        if do_denoise and denoise_params is not None:
            print("Applying per-frame noise reduction (inference res)...", flush=True)
            depth = denoise_depth_stack(
                depth,
                denoise_params,
                device=args.denoise_device,
                workers=_denoise_workers(args.denoise_workers),
            )
        output_path = save_grayscale_depth(depth, fps_probe, args)

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
