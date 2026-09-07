#!/usr/bin/env python3
"""Generate depth PFMs from CR3/photo folders with Depth Anything 3 (MPS)."""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.95")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.70")

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parent

from denoise import default_denoise_params, denoise_frame  # noqa: E402
from edge_smooth import default_edge_smooth_params, smooth_depth_edges  # noqa: E402
from pfm import write_pfm  # noqa: E402
from upsample import default_upsample_params, sharp_upsample  # noqa: E402
from resources import total_ram_bytes  # noqa: E402

IMAGE_SUFFIXES = {
    ".cr3",
    ".cr2",
    ".nef",
    ".arw",
    ".dng",
    ".raf",
    ".orf",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

PFM_NAMES = (
    "depth.pfm",
    "inverse.pfm",
    "min.pfm",
    "min_inverse.pfm",
    "max.pfm",
    "max_inverse.pfm",
)

MODEL_PRESETS: dict[str, str] = {
    "da3-small": "da3-small",
    "da3-base": "da3-base",
    "da3-large": "da3-large",
    "da3mono-large": "da3mono-large",
    "da3metric-large": "da3metric-large",
    "da3-giant": "da3-giant",
    "da3nested-giant-large": "da3nested-giant-large",
}

# ViT patch alignment used by DA3 resize paths.
PROCESS_RES_ALIGN = 14
PROCESS_RES_ABSOLUTE_MAX = 1680


def _model_size_class(model_name: str) -> str:
    key = model_name.lower()
    if "nested" in key or "giant" in key:
        return "giant"
    if "large" in key or "metric" in key or "mono" in key:
        return "large"
    if "base" in key:
        return "base"
    return "small"


def _align_process_res(value: int) -> int:
    n = int(value)
    n = min(n, PROCESS_RES_ABSOLUTE_MAX)
    n = max(PROCESS_RES_ALIGN, (n // PROCESS_RES_ALIGN) * PROCESS_RES_ALIGN)
    return n


def resolve_process_res(
    requested: str | int,
    *,
    model_name: str,
    device: torch.device,
) -> int:
    """
    Pick DA3 ``process_res`` (long side target) as high as RAM/model allow.

    Default ``auto`` pushes toward MPS/VRAM limits on Apple Silicon unified memory.
    """
    if str(requested).lower() not in ("auto", ""):
        return _align_process_res(int(requested))

    mem_gb = total_ram_bytes() / (1024**3)
    tier = _model_size_class(model_name)

    # Aggressive caps — tuned for still-image DA3 on unified memory.
    if mem_gb <= 8.5:
        caps = {"small": 896, "base": 840, "large": 756, "giant": 588}
    elif mem_gb <= 14.0:
        caps = {"small": 1260, "base": 1176, "large": 1008, "giant": 840}
    elif mem_gb <= 24.0:
        caps = {"small": 1512, "base": 1428, "large": 1260, "giant": 1008}
    else:
        caps = {"small": 1680, "base": 1596, "large": 1512, "giant": 1260}

    res = caps[tier]
    if device.type == "cpu":
        res = min(res, 756)
    elif device.type == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb <= 8.0:
                    res = min(res, 840 if tier in ("small", "base") else 756)
                elif vram_gb <= 12.0:
                    res = min(res, 1008)
        except Exception:
            pass

    res = _align_process_res(res)
    print(
        f"process_res=auto -> {res} "
        f"(RAM {mem_gb:.1f} GiB, model tier={tier}, device={device.type})",
        flush=True,
    )
    return res


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


def resolve_denoise_device(requested: str, infer_device: torch.device) -> str:
    req = (requested or "auto").lower()
    if req == "cpu":
        return "cpu"
    if req == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        print("Denoise: MPS unavailable, falling back to CPU.", flush=True)
        return "cpu"
    if req == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("Denoise: CUDA unavailable, falling back to CPU.", flush=True)
        return "cpu"
    if infer_device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    if infer_device.type == "cuda" and torch.cuda.is_available():
        return "cuda"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def free_memory(device: torch.device | None = None) -> None:
    gc.collect()
    if device is not None and device.type == "mps" and hasattr(torch, "mps"):
        if hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except RuntimeError:
                pass
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_da3_model(model_name: str, device: torch.device):
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:
        raise SystemExit(
            "depth_anything_3 is not installed. Run:\n"
            "  pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"
        ) from exc

    preset = MODEL_PRESETS.get(model_name.lower(), model_name.lower())
    hub_name = preset.upper().replace("_", "-")
    if not hub_name.startswith("DA3"):
        hub_name = f"DA3-{hub_name}"
    repo_id = f"depth-anything/{hub_name}"
    print(f"Loading DA3 {repo_id} on {device} …", flush=True)
    t0 = time.perf_counter()
    try:
        model = DepthAnything3.from_pretrained(repo_id)
    except Exception:
        model = DepthAnything3(model_name=preset)
    model = model.to(device)
    model.eval()
    print(f"Model ready in {time.perf_counter() - t0:.1f}s", flush=True)
    return model


RAW_SUFFIXES = {".cr3", ".cr2", ".nef", ".arw", ".dng", ".raf", ".orf"}


def _read_raw_rgb(path: Path) -> np.ndarray:
    try:
        import rawpy
    except ImportError as exc:
        raise SystemExit(
            "rawpy is required for CR3/RAW files. Run: pip install rawpy\n"
            "On macOS you may also need: brew install libraw"
        ) from exc

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=False,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=8,
            # Apply camera/EXIF orientation so depth matches viewers (darktable, etc.).
            user_flip=0,
        )
    return np.ascontiguousarray(rgb)


def load_rgb(path: Path, *, apply_exif: bool = True) -> tuple[np.ndarray, bool]:
    """Load RGB uint8; return (array, exif_was_applied)."""
    suffix = path.suffix.lower()
    if suffix in RAW_SUFFIXES:
        return _read_raw_rgb(path), False
    with Image.open(path) as im:
        exif = im.getexif()
        ori = int(exif.get(274, 1)) if exif else 1
        if apply_exif:
            im = ImageOps.exif_transpose(im)
        exif_applied = apply_exif and ori not in (1,)
        rgb = im.convert("RGB")
    return np.ascontiguousarray(np.array(rgb, dtype=np.uint8)), exif_applied


def rgb_to_guide_gray(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return np.clip(gray, 0.0, 1.0)


def exif_focal_px(path: Path, width: int) -> float | None:
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            if not exif:
                return None
            focal_mm = exif.get(37386)  # FocalLength
            if focal_mm is None:
                return None
            focal_mm = float(focal_mm)
            fl35 = exif.get(41989)  # FocalLengthIn35mmFilm
            if fl35:
                return float(fl35) * width / 36.0
            # Rough fallback: assume APS-C if no 35mm equivalent
            return focal_mm * width / 24.0
    except Exception:
        return None


def is_metric_model(model_name: str) -> bool:
    key = model_name.lower()
    return "metric" in key or "nested" in key


def infer_depth(
    model,
    rgb: np.ndarray,
    *,
    process_res: int,
    process_res_method: str,
    metric_scale: bool,
    focal_px: float | None,
) -> np.ndarray:
    """Run DA3 on one image; return float32 depth at infer resolution."""
    with torch.inference_mode():
        prediction = model.inference(
            image=[rgb],
            process_res=int(process_res),
            process_res_method=process_res_method,
        )
    depth = np.asarray(prediction.depth[0], dtype=np.float32)
    finite = np.isfinite(depth)
    if int(finite.sum()) < 64:
        raise RuntimeError("DA3 returned invalid depth")

    if metric_scale and focal_px is not None and focal_px > 0:
        depth = depth * float(focal_px) / 300.0

    return depth


def normalize_01(arr: np.ndarray, percentile: float = 2.0) -> np.ndarray:
    finite = np.isfinite(arr)
    if int(finite.sum()) < 64:
        return np.zeros_like(arr, dtype=np.float32)
    vals = arr[finite]
    lo = float(np.percentile(vals, percentile))
    hi = float(np.percentile(vals, 100.0 - percentile))
    if hi <= lo:
        hi = lo + 1e-6
    span = hi - lo
    out = np.clip((arr.astype(np.float32) - lo) / span, 0.0, 1.0)
    return np.where(finite, out, 0.0).astype(np.float32)


def derive_pfm_maps(depth: np.ndarray, *, eps: float = 1e-6) -> dict[str, np.ndarray]:
    d = np.asarray(depth, dtype=np.float32)
    inv = 1.0 / np.maximum(d, eps)
    d01 = normalize_01(d)
    i01 = normalize_01(inv)
    mn = np.minimum(d01, i01)
    mx = np.maximum(d01, i01)
    return {
        "depth.pfm": d,
        "inverse.pfm": inv.astype(np.float32),
        "min.pfm": mn.astype(np.float32),
        "min_inverse.pfm": (1.0 - mn).astype(np.float32),
        "max.pfm": mx.astype(np.float32),
        "max_inverse.pfm": (1.0 - mx).astype(np.float32),
    }


def output_complete(out_dir: Path) -> bool:
    return out_dir.is_dir() and all((out_dir / name).is_file() for name in PFM_NAMES)


def process_image(
    model,
    path: Path,
    out_dir: Path,
    *,
    device: torch.device,
    denoise_device: str,
    process_res: int,
    process_res_method: str,
    model_name: str,
    do_upsample: bool,
    do_denoise: bool,
    do_edge_smooth: bool,
    apply_exif: bool = True,
) -> None:
    t0 = time.perf_counter()
    print(f"\n[{path.name}] loading …", flush=True)
    rgb, exif_applied = load_rgb(path, apply_exif=apply_exif)
    if exif_applied:
        print("  applied EXIF orientation (matches darktable display)", flush=True)
    orig_h, orig_w = rgb.shape[:2]
    guide = rgb_to_guide_gray(rgb)

    focal_px = exif_focal_px(path, orig_w) if is_metric_model(model_name) else None
    depth_low = infer_depth(
        model,
        rgb,
        process_res=process_res,
        process_res_method=process_res_method,
        metric_scale=is_metric_model(model_name),
        focal_px=focal_px,
    )
    infer_h, infer_w = depth_low.shape[:2]
    print(f"  infer {infer_w}x{infer_h} -> native {orig_w}x{orig_h}", flush=True)

    upsample_dev = device.type if device.type in ("mps", "cuda") else "cpu"
    if do_upsample and (infer_h, infer_w) != (orig_h, orig_w):
        params = default_upsample_params(infer_h, infer_w, orig_h, orig_w)
        depth = sharp_upsample(
            depth_low,
            guide,
            (orig_w, orig_h),
            params,
            device=upsample_dev,
        )
    else:
        depth = cv2.resize(depth_low, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR).astype(
            np.float32
        )

    if do_denoise:
        dn_params = default_denoise_params(orig_h, orig_w)
        depth = denoise_frame(depth, dn_params, device=denoise_device)

    if do_edge_smooth:
        es_params = default_edge_smooth_params(orig_h, orig_w)
        depth = smooth_depth_edges(depth, guide, es_params, device=denoise_device)

    maps = derive_pfm_maps(depth)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in maps.items():
        write_pfm(out_dir / name, arr)
        print(f"  wrote {out_dir / name}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s", flush=True)
    free_memory(device)


def iter_images(input_dir: Path) -> list[Path]:
    files = [
        p
        for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate depth PFMs from CR3/photo folders with DA3 (MPS)"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder containing CR3/JPEG/PNG images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root for per-image output folders (default: same as --input-dir)",
    )
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu", "auto"])
    parser.add_argument(
        "--model",
        default="da3-small",
        help="DA3 preset (default: da3-small)",
    )
    parser.add_argument(
        "--process-res",
        default="auto",
        help=(
            "DA3 infer long-side target (default: auto — max safe for RAM/model/device). "
            f"Manual values are aligned to {PROCESS_RES_ALIGN}px, max {PROCESS_RES_ABSOLUTE_MAX}."
        ),
    )
    parser.add_argument(
        "--process-res-method",
        default="upper_bound_resize",
        choices=["upper_bound_resize", "lower_bound_resize"],
    )
    parser.add_argument("--no-upsample", action="store_true")
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--no-edge-smooth", action="store_true")
    parser.add_argument(
        "--denoise-device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-exif-orient",
        action="store_true",
        help="Keep sensor/file storage orientation (skip EXIF transpose on JPEG/PNG)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    output_root = (args.output_dir or args.input_dir).resolve()
    device = resolve_device(args.device)
    denoise_device = resolve_denoise_device(args.denoise_device, device)
    process_res = resolve_process_res(
        args.process_res, model_name=args.model, device=device
    )

    images = iter_images(input_dir)
    if not images:
        raise SystemExit(f"No images found in {input_dir}")

    os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))

    print(
        f"Photo depth PFMs | device={device} | model={args.model} | "
        f"process_res={process_res} | images={len(images)} | "
        f"upsample={'on' if not args.no_upsample else 'off'} | "
        f"denoise={denoise_device if not args.no_denoise else 'off'} | "
        f"edge_smooth={'on' if not args.no_edge_smooth else 'off'}",
        flush=True,
    )

    model = load_da3_model(args.model, device)

    processed = 0
    skipped = 0
    try:
        with torch.inference_mode():
            for path in images:
                out_dir = output_root / path.stem
                if not args.overwrite and output_complete(out_dir):
                    print(f"\n[{path.name}] skip (complete)", flush=True)
                    skipped += 1
                    continue
                process_image(
                    model,
                    path,
                    out_dir,
                    device=device,
                    denoise_device=denoise_device,
                    process_res=process_res,
                    process_res_method=args.process_res_method,
                    model_name=args.model,
                    do_upsample=not args.no_upsample,
                    do_denoise=not args.no_denoise,
                    do_edge_smooth=not args.no_edge_smooth,
                    apply_exif=not args.no_exif_orient,
                )
                processed += 1
    finally:
        del model
        free_memory(device)

    print(
        f"\nFinished: processed={processed} skipped={skipped} output={output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
