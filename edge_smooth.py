"""RGB-guided edge smoothing for single-frame depth maps.

Softens depth discontinuities that do not align with the source RGB edges
while preserving sharp, image-consistent boundaries (post-JBU polish).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class EdgeSmoothParams:
    radius: int = 3
    sigma_spatial: float = 2.0
    sigma_range: float = 0.06
    mismatch_strength: float = 0.75
    feather: float = 1.25


def default_edge_smooth_params(height: int, width: int) -> EdgeSmoothParams:
    short = max(1, min(int(height), int(width)))
    r = int(round(short * 0.0015))
    r = max(2, min(5, r))
    return EdgeSmoothParams(
        radius=r,
        sigma_spatial=max(1.5, r * 0.65),
        sigma_range=0.06,
        mismatch_strength=0.75,
        feather=1.25,
    )


def _resolve_torch_device(device: str) -> str | None:
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


def _sobel_mag_cpu(arr: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy).astype(np.float32)


def _sobel_mag_torch(arr: np.ndarray, *, device: str) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    dev = torch.device(device)
    x = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(dev)
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=dev
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=dev
    ).view(1, 1, 3, 3)
    xb = x.view(1, 1, *x.shape)
    gx = F.conv2d(xb, kx, padding=1)
    gy = F.conv2d(xb, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy).view(*x.shape)
    return mag.detach().float().cpu().numpy().astype(np.float32, copy=False)


def _guided_bilateral_cpu(
    depth: np.ndarray,
    guide: np.ndarray,
    radius: int,
    sigma_spatial: float,
    sigma_range: float,
) -> np.ndarray:
    h, w = depth.shape[:2]
    depth = depth.astype(np.float32, copy=False)
    guide = guide.astype(np.float32, copy=False)
    acc = np.zeros((h, w), dtype=np.float32)
    acc_w = np.zeros((h, w), dtype=np.float32)
    inv_2ss = 1.0 / (2.0 * sigma_spatial * sigma_spatial)
    inv_2sr = 1.0 / (2.0 * sigma_range * sigma_range)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            d_shift = np.roll(depth, shift=(dy, dx), axis=(0, 1))
            g_shift = np.roll(guide, shift=(dy, dx), axis=(0, 1))
            w_spatial = math.exp(-(dx * dx + dy * dy) * inv_2ss)
            diff = guide - g_shift
            w = w_spatial * np.exp(-(diff * diff) * inv_2sr)
            acc += w * d_shift
            acc_w += w
    return (acc / np.maximum(acc_w, 1e-8)).astype(np.float32)


def _guided_bilateral_torch(
    depth: np.ndarray,
    guide: np.ndarray,
    radius: int,
    sigma_spatial: float,
    sigma_range: float,
    *,
    device: str,
) -> np.ndarray:
    import torch

    dev = torch.device(device)
    d = torch.from_numpy(np.ascontiguousarray(depth, dtype=np.float32)).to(dev)
    g = torch.from_numpy(np.ascontiguousarray(guide, dtype=np.float32)).to(dev)
    h, w = d.shape
    acc = torch.zeros((h, w), device=dev, dtype=torch.float32)
    acc_w = torch.zeros((h, w), device=dev, dtype=torch.float32)
    inv_2ss = 1.0 / (2.0 * sigma_spatial * sigma_spatial)
    inv_2sr = 1.0 / (2.0 * sigma_range * sigma_range)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            d_shift = torch.roll(d, shifts=(dy, dx), dims=(0, 1))
            g_shift = torch.roll(g, shifts=(dy, dx), dims=(0, 1))
            w_spatial = math.exp(-(dx * dx + dy * dy) * inv_2ss)
            diff = g - g_shift
            weight = w_spatial * torch.exp(-(diff * diff) * inv_2sr)
            acc = acc + weight * d_shift
            acc_w = acc_w + weight
    out = acc / torch.clamp(acc_w, min=1e-8)
    return out.detach().float().cpu().numpy().astype(np.float32, copy=False)


def _edge_mismatch_mask(
    depth: np.ndarray,
    guide_gray: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    """High where depth edges disagree with RGB edges (likely hallucinated)."""
    torch_dev = _resolve_torch_device(device)
    if torch_dev is not None:
        d_mag = _sobel_mag_torch(depth, device=torch_dev)
        g_mag = _sobel_mag_torch(guide_gray, device=torch_dev)
    else:
        d_mag = _sobel_mag_cpu(depth)
        g_mag = _sobel_mag_cpu(guide_gray)

    d_n = d_mag / max(float(np.percentile(d_mag, 98.0)), 1e-6)
    g_n = g_mag / max(float(np.percentile(g_mag, 98.0)), 1e-6)
    # Agreement: both strong edges; disagreement: depth edge without RGB edge.
    agree = np.minimum(d_n, g_n)
    disagree = np.clip(d_n - 0.35 * g_n, 0.0, 1.0)
    mask = np.clip(disagree - 0.45 * agree, 0.0, 1.0)
    mask = mask * mask * (3.0 - 2.0 * mask)
    return mask.astype(np.float32)


def smooth_depth_edges(
    depth: np.ndarray,
    guide_gray: np.ndarray,
    params: EdgeSmoothParams | None = None,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """
    Blend depth toward an RGB-guided bilateral smooth at mismatched edges.

    ``guide_gray`` is full-res float32 luminance in [0, 1].
    """
    if depth.ndim != 2:
        raise ValueError(f"smooth_depth_edges expects [H,W], got {depth.shape}")
    src = np.asarray(depth, dtype=np.float32)
    guide = np.clip(np.asarray(guide_gray, dtype=np.float32), 0.0, 1.0)
    if guide.shape != src.shape:
        guide = cv2.resize(guide, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_AREA)

    if params is None:
        params = default_edge_smooth_params(src.shape[0], src.shape[1])

    finite = np.isfinite(src)
    if int(finite.sum()) < 64:
        return src

    work = np.where(finite, src, 0.0).astype(np.float32, copy=False)
    depth_rng = float(
        np.percentile(work[finite], 98.0) - np.percentile(work[finite], 2.0)
    )
    sigma_range = max(1e-6, float(params.sigma_range) * max(depth_rng, 1e-6))
    radius = max(1, int(params.radius))

    torch_dev = _resolve_torch_device(device)
    if torch_dev is not None:
        smooth = _guided_bilateral_torch(
            work,
            guide,
            radius,
            float(params.sigma_spatial),
            sigma_range,
            device=torch_dev,
        )
    else:
        smooth = _guided_bilateral_cpu(
            work,
            guide,
            radius,
            float(params.sigma_spatial),
            sigma_range,
        )

    mismatch = _edge_mismatch_mask(work, guide, device=device)
    if params.feather > 0:
        mismatch = cv2.GaussianBlur(mismatch, (0, 0), float(params.feather))

    strength = float(np.clip(params.mismatch_strength, 0.0, 1.0))
    blend = np.clip(mismatch * strength, 0.0, 1.0)
    out = work * (1.0 - blend) + smooth * blend
    out = np.where(finite, out, src)
    return out.astype(np.float32, copy=False)
