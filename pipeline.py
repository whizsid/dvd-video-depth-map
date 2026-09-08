"""DepthCrafter CUDA runner: load SVD UNet pipeline and estimate video depth."""

from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent
_DC = _REPO / "vendor" / "DepthCrafter"
if str(_DC) not in sys.path:
    sys.path.insert(0, str(_DC))

from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline  # noqa: E402
from depthcrafter.unet import (  # noqa: E402
    DiffusersUNetSpatioTemporalConditionModelDepthCrafter,
)

logger = logging.getLogger(__name__)


def load_depthcrafter_pipeline(
    *,
    unet_path: str | Path,
    svd_path: str | Path,
    cpu_offload: Optional[str] = "model",
    device: str = "cuda",
) -> DepthCrafterPipeline:
    """Load DepthCrafter UNet + SVD-XT pipeline with optional CPU offload."""
    unet_path = str(unet_path)
    svd_path = str(svd_path)
    print(f"Loading DepthCrafter UNet from {unet_path} ...", flush=True)
    unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
        unet_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    print(f"Loading SVD backbone from {svd_path} ...", flush=True)
    pipe = DepthCrafterPipeline.from_pretrained(
        svd_path,
        unet=unet,
        torch_dtype=torch.float16,
        variant="fp16",
    )

    if cpu_offload is None or cpu_offload == "none":
        pipe.to(device)
        print(f"DepthCrafter on {device} (no CPU offload).", flush=True)
    elif cpu_offload == "sequential":
        pipe.enable_sequential_cpu_offload()
        print("DepthCrafter: sequential CPU offload enabled.", flush=True)
    elif cpu_offload == "model":
        pipe.enable_model_cpu_offload()
        print("DepthCrafter: model CPU offload enabled.", flush=True)
    else:
        raise ValueError(
            f"Unknown cpu_offload={cpu_offload!r}; use none|model|sequential"
        )

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xformers memory-efficient attention enabled.", flush=True)
    except Exception as exc:
        print(f"xformers not enabled ({exc}); continuing with attention slicing.", flush=True)
    pipe.enable_attention_slicing()
    return pipe


def run_depthcrafter(
    pipe: DepthCrafterPipeline,
    frames: np.ndarray,
    *,
    num_inference_steps: int = 5,
    guidance_scale: float = 1.0,
    window_size: int = 110,
    overlap: int = 25,
    track_time: bool = False,
) -> np.ndarray:
    """Run DepthCrafter on ``frames`` [T,H,W,3] float32 in [0,1].

    Returns single-channel depth ``[T,H,W]`` normalized to [0, 1].
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames [T,H,W,3], got {frames.shape}")
    height, width = int(frames.shape[1]), int(frames.shape[2])
    overlap = max(0, min(int(overlap), max(0, int(window_size) - 1)))
    print(
        f"DepthCrafter infer: T={frames.shape[0]} {width}x{height} "
        f"steps={num_inference_steps} guidance={guidance_scale} "
        f"window={window_size} overlap={overlap}",
        flush=True,
    )
    with torch.inference_mode():
        res = pipe(
            frames,
            height=height,
            width=width,
            output_type="np",
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(num_inference_steps),
            window_size=int(window_size),
            overlap=overlap,
            track_time=bool(track_time),
        ).frames[0]
    # RGB channels → single channel, then normalize across the clip.
    depth = res.sum(-1) / max(res.shape[-1], 1)
    d_min = float(np.nanmin(depth))
    d_max = float(np.nanmax(depth))
    denom = max(d_max - d_min, 1e-8)
    depth = ((depth - d_min) / denom).astype(np.float32, copy=False)
    return depth


def release_pipeline(pipe) -> None:
    """Drop the pipeline and free CUDA memory before post-process (JBU)."""
    try:
        del pipe
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
