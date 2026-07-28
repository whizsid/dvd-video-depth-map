#!/usr/bin/env python3
"""Generate a grayscale depth-map video with DVD on full CUDA (T4-oriented).

Unlike ``generate_depth.py`` (Mac MPS: DiT in RAM + block swap + CPU VAE), this
keeps DiT and VAE resident on the GPU. Runtime caches stay on local disk
(``.cache/`` or ``--cache-dir``) — never FAT32 external volumes. Google Drive
(or any remote mount) is only for input/output videos.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parent
DVD_ROOT = REPO_ROOT / "vendor" / "DVD"
sys.path.insert(0, str(DVD_ROOT))
sys.path.insert(0, str(DVD_ROOT / "test_script"))

from examples.wanvideo.model_training.WanTrainingModule import (  # noqa: E402
    WanTrainingModule,
)

import generate_depth as gd  # noqa: E402
from da3_stabilize import probe_shot_ranges  # noqa: E402
from upsample import (  # noqa: E402
    cleanup_stale_up_caches,
    default_upsample_params,
)


def resolve_local_cache_root(explicit: Path | None) -> Path:
    """Local project/Colab cache only — never auto-pick FAT32 volumes."""
    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
    else:
        env = os.environ.get("DVD_CACHE_DIR", "").strip()
        root = Path(env).expanduser().resolve() if env else (REPO_ROOT / ".cache")
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_model_config(explicit: Path | None, ckpt_dir: Path) -> Path:
    """Locate model_config.yaml (repo configs/, downloaded ckpt/, or vendor)."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            Path(ckpt_dir) / "model_config.yaml",
            REPO_ROOT / "configs" / "model_config.yaml",
            DVD_ROOT / "ckpt" / "model_config.yaml",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    checked = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Missing model_config.yaml. Checked:\n  "
        f"{checked}\n"
        "Run: python scripts/download_weights.py  (or pull the latest repo with "
        "configs/model_config.yaml)."
    )


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


def apply_low_host_ram_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Shrink windows / disable prep overlap on Colab-class ~12 GiB hosts.

    Upstream defaults (480×640, window 81) keep DiT+VAE on the T4 fine, but
    host-side prep queues + float32 VAE dumps exhaust Colab system RAM.
    """
    from resources import is_low_system_ram, total_ram_bytes

    mem_gb = total_ram_bytes() / (1024**3)
    low = is_low_system_ram() or _is_colab()

    # Resolve None → profile defaults (explicit CLI values always win).
    if low:
        if args.height is None:
            args.height = 320
        if args.width is None:
            args.width = 576
        if args.window_size is None:
            args.window_size = 17
        if args.overlap is None:
            args.overlap = 5
        if not args.no_pipeline_parallel:
            args.no_pipeline_parallel = True
        if str(args.upsample_workers) == "auto":
            args.upsample_workers = "1"
        print(
            f"Low host RAM profile ({mem_gb:.1f} GiB"
            f"{', Colab' if _is_colab() else ''}): "
            f"{args.height}x{args.width} window={args.window_size} "
            f"overlap={args.overlap} sequential prep|infer. "
            "Pass explicit flags to override.",
            flush=True,
        )
    else:
        if args.height is None:
            args.height = 480
        if args.width is None:
            args.width = 640
        if args.window_size is None:
            args.window_size = 81
        if args.overlap is None:
            args.overlap = 21

    return args


def load_model_full_gpu(
    ckpt_dir: Path,
    yaml_args,
    device: torch.device,
    dtype: torch.dtype,
):
    """Load DVD/Wan with DiT + VAE fully resident on ``device`` (no block swap)."""
    # peft probes torchao and hard-raises on Colab's stock 0.10.x; DVD only needs LoRA.
    try:
        import importlib.metadata as importlib_metadata
        import importlib.util

        if importlib.util.find_spec("torchao") is not None:
            ver = importlib_metadata.version("torchao")
            from packaging.version import Version

            if Version(ver) < Version("0.16.0"):
                raise ImportError(
                    f"Incompatible torchao {ver} (peft needs >=0.16). "
                    "On Colab run: !pip uninstall -y torchao  then Restart session. "
                    "DVD does not use torchao."
                )
    except ImportError:
        raise
    except Exception:
        pass

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
    gd.free_memory()

    model.pipe.dit.load_state_dict(dit_state_dict, strict=True)
    del dit_state_dict
    gd.free_memory()

    print("Merging LoRA layers...", flush=True)
    model.merge_lora_layer()
    print("Baking LoRA into base Linear layers...", flush=True)
    replaced = gd.bake_and_unload_lora(model.pipe.dit)
    print(f"Replaced {replaced} LoRA modules with plain Linear.", flush=True)
    gd.free_memory()

    print(f"Casting DiT + VAE to {dtype} on CPU...", flush=True)
    gd.cast_pipe_dtype_low_mem(model.pipe, dtype, only=("dit", "vae"))
    gd.cast_dit_freqs_for_mps(model.pipe.dit)
    gd.free_memory()

    print("Dropping unused pipeline modules...", flush=True)
    gd.drop_unused_pipe_modules(model.pipe)

    print(f"Moving DiT + VAE to {device} (full GPU residency)...", flush=True)
    model.pipe.dit.to(device=device, dtype=dtype)
    model.pipe.vae.to(device=device, dtype=dtype)
    model.pipe.device = device
    model.pipe.torch_dtype = dtype
    model.pipe.vram_management_enabled = False
    # Drop any lingering host copies from load/cast/LoRA bake (Colab RAM killer).
    import gc

    gc.collect()
    gd.free_memory(device)
    gc.collect()
    gd.free_memory(device)

    model.eval()
    print(
        f"Model ready (full GPU: DiT+VAE on {device}; dtype={dtype}).",
        flush=True,
    )
    print(f"  {cuda_device_info()}", flush=True)
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a grayscale depth-map video with DVD on full CUDA "
            "(T4: DiT+VAE resident on GPU, local disk caches only)"
        )
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
            "Local root for upsample memmaps (default: ./.cache or $DVD_CACHE_DIR). "
            "Never uses FAT32 external volumes."
        ),
    )
    parser.add_argument(
        "--dtype", default="float16", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Inference height (default: 480, or 320 on Colab/≤14GiB RAM)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Inference width (default: 640, or 576 on Colab/≤14GiB RAM)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Temporal window (default: 81, or 17 on Colab/≤14GiB RAM; must be 4n+1)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Window overlap (default: 21, or 5 on Colab/≤14GiB RAM)",
    )
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
        help="CPU threads for JBU upscale (auto|N)",
    )
    parser.add_argument(
        "--keep-upsample-cache",
        action="store_true",
        help="Reuse stable up_cache_v_* memmap under --cache-dir",
    )
    parser.add_argument(
        "--no-pipeline-parallel",
        action="store_true",
        help="Disable prep|infer|post overlap",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args = apply_low_host_ram_defaults(args)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for generate_depth_cuda.py. "
            "Use generate_depth.py for MPS/CPU, or enable a GPU runtime."
        )
    device = torch.device("cuda")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    model_config = resolve_model_config(args.model_config, args.ckpt)
    yaml_args = OmegaConf.load(str(model_config))
    print(f"Model config: {model_config}", flush=True)

    cache_root = resolve_local_cache_root(args.cache_dir)
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))
    os.chdir(REPO_ROOT)

    do_upsample = not args.no_upsample
    pipeline_parallel = not args.no_pipeline_parallel
    cleanup_stale_up_caches(cache_root, keep_pid=os.getpid())
    cleanup_stale_up_caches(Path(args.output_dir), keep_pid=os.getpid())

    print(
        f"Device: {device} | {cuda_device_info()}\n"
        f"dtype={dtype} | size={args.height}x{args.width} | "
        f"window={args.window_size} overlap={args.overlap} | "
        f"upsample={'JBU ' + str(args.upsample_workers) if do_upsample else 'off'} | "
        f"pipeline={'parallel' if pipeline_parallel else 'sequential'} | "
        f"cache={cache_root}",
        flush=True,
    )

    model = load_model_full_gpu(
        args.ckpt,
        yaml_args,
        device=device,
        dtype=dtype,
    )

    print("Probing input video (no full decode)...", flush=True)
    fps_probe, total_probe, orig_h, orig_w = gd.probe_video(args.input_video)
    if args.max_frames is not None:
        total_probe = min(total_probe, int(args.max_frames))
    shot_ranges = tuple(
        (s0, min(s1, total_probe))
        for s0, s1 in probe_shot_ranges(args.input_video)
        if s0 < total_probe and min(s1, total_probe) > s0
    ) or ((0, total_probe),)

    out_h, out_w = gd.inference_hw(orig_h, orig_w, args.height, args.width)
    print(f"Inference geometry: {out_w}x{out_h} (from {orig_w}x{orig_h})", flush=True)
    gd.free_memory(device)

    upsample_params = None
    if do_upsample:
        upsample_params = default_upsample_params(out_h, out_w, orig_h, orig_w)
        print(
            f"JBU upsample: edge_radius={upsample_params.edge_radius} "
            f"sigma_range={upsample_params.sigma_range}",
            flush=True,
        )

    with torch.inference_mode():
        depth, origin_fps, orig_size, upscaler = gd.generate_depth_from_video(
            model,
            args.input_video,
            out_h=out_h,
            out_w=out_w,
            dtype=dtype,
            window_size=args.window_size,
            overlap=args.overlap,
            device=device,
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
        gd.free_memory(device)

    del model
    gd.free_memory(device)

    if upscaler is not None:
        print("Gathering upsampled frames (model freed)...", flush=True)
        upscaler.finish()
        print(
            f"Depth at native {orig_size[1]}x{orig_size[0]} via guided JBU.",
            flush=True,
        )
        output_path = gd.save_grayscale_depth_from_upscaler(
            upscaler, origin_fps, args
        )
    else:
        depth = depth[0]
        print(
            f"Keeping inference resolution {depth.shape[2]}x{depth.shape[1]} "
            f"(omit --no-upsample for full-res JBU).",
            flush=True,
        )
        output_path = gd.save_grayscale_depth(depth, origin_fps, args)
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
