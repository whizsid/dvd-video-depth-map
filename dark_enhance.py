"""Low-cost dark-scene contrast / edge pre-pass for DepthCrafter video encoding.

Applied only to inference RGB frames so the model still sees silhouettes
(heads, limbs) when people walk into shadow. Optical-flow, shot stabilize,
and JBU keep the original frames.

OpenCV only: CLAHE + optional gamma lift + Scharr overlay + chroma boost.
Cost is negligible next to DepthCrafter inference (infer-resolution uint8).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DarkEnhanceParams:
    strength: float = 1.0
    clahe_tiles: tuple[int, int] = (8, 8)
    min_clip: float = 1.2
    max_clip: float = 3.8
    sat_boost: float = 0.38
    edge_boost: float = 0.30
    gamma_boost: float = 0.55
    ema: float = 0.22
    skip_threshold: float = 0.04


def default_dark_enhance_params(strength: float = 1.0) -> DarkEnhanceParams:
    return DarkEnhanceParams(strength=float(max(0.0, strength)))


def _scene_darkness(luma: np.ndarray) -> float:
    """0 = well-lit, 1 = crushed shadows (mean + dark-tail percentile)."""
    mean_l = float(cv2.mean(luma)[0])
    # 15th percentile catches a person in shadow even if a lamp lifts the mean.
    p15 = float(np.percentile(luma, 15))
    from_mean = np.clip((108.0 - mean_l) / 78.0, 0.0, 1.0)
    from_tail = np.clip((72.0 - p15) / 58.0, 0.0, 1.0)
    return float(max(from_mean, from_tail))


def _gamma_lut(gamma: float) -> np.ndarray:
    inv = 1.0 / max(float(gamma), 1.0)
    x = np.arange(256, dtype=np.float32) / 255.0
    return np.clip(np.power(x, inv) * 255.0, 0.0, 255.0).astype(np.uint8)


class DarkEdgePrepass:
    """Stateful per-frame enhancer with EMA-smoothed darkness (flicker control)."""

    def __init__(self, params: DarkEnhanceParams | None = None):
        self.params = params or default_dark_enhance_params()
        self._dark_ema: float | None = None
        self._clahe: cv2.CLAHE | None = None
        self._clip: float | None = None
        self._lut: np.ndarray | None = None
        self._lut_gamma: float | None = None
        self._logged = False

    def reset(self) -> None:
        self._dark_ema = None

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        return self.apply(bgr)

    def apply(self, bgr: np.ndarray) -> np.ndarray:
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            return bgr
        strength = float(self.params.strength)
        if strength <= 0.0:
            return bgr

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        luma = lab[:, :, 0]
        raw_dark = _scene_darkness(luma)
        if self._dark_ema is None:
            dark = raw_dark
        else:
            a = float(np.clip(self.params.ema, 0.05, 0.9))
            dark = (1.0 - a) * self._dark_ema + a * raw_dark
        self._dark_ema = dark

        amount = float(np.clip(dark * strength, 0.0, 1.5))
        if amount < float(self.params.skip_threshold):
            return bgr

        if not self._logged:
            print(
                f"Dark-scene pre-pass: CLAHE+Scharr+sat "
                f"(darkness={dark:.2f}, amount={amount:.2f}, strength={strength:.2f})",
                flush=True,
            )
            self._logged = True

        L = luma
        gamma = 1.0 + float(self.params.gamma_boost) * min(amount, 1.0)
        if gamma > 1.02:
            if self._lut is None or abs((self._lut_gamma or 0.0) - gamma) > 0.04:
                self._lut = _gamma_lut(gamma)
                self._lut_gamma = gamma
            L = cv2.LUT(L, self._lut)

        # Soft denoise before CLAHE so ISO grain in shadows is not amplified.
        if amount > 0.45:
            L = cv2.GaussianBlur(L, (3, 3), 0.55)

        clip = float(self.params.min_clip) + (
            float(self.params.max_clip) - float(self.params.min_clip)
        ) * min(amount, 1.0)
        clip = float(np.clip(clip, 1.0, 8.0))
        if self._clahe is None or self._clip is None or abs(self._clip - clip) >= 0.25:
            tiles = self.params.clahe_tiles
            self._clahe = cv2.createCLAHE(
                clipLimit=clip, tileGridSize=(int(tiles[0]), int(tiles[1]))
            )
            self._clip = clip
        L = self._clahe.apply(L)

        # Cheap Scharr magnitude overlaid on L — silhouette ridges without a CNN.
        edge_amt = float(self.params.edge_boost) * min(amount, 1.0)
        if edge_amt > 0.02:
            gx = cv2.Scharr(L, cv2.CV_16S, 1, 0)
            gy = cv2.Scharr(L, cv2.CV_16S, 0, 1)
            edge = cv2.addWeighted(
                cv2.convertScaleAbs(gx), 0.5, cv2.convertScaleAbs(gy), 0.5, 0
            )
            L = cv2.addWeighted(L, 1.0, edge, edge_amt, 0)

        lab[:, :, 0] = L
        sat = 1.0 + float(self.params.sat_boost) * min(amount, 1.0)
        if sat > 1.02:
            # Push a/b away from neutral 128 (LAB chroma ≈ saturation).
            mid = np.full_like(lab[:, :, 1], 128)
            lab[:, :, 1] = cv2.addWeighted(lab[:, :, 1], sat, mid, 1.0 - sat, 0)
            lab[:, :, 2] = cv2.addWeighted(lab[:, :, 2], sat, mid, 1.0 - sat, 0)

        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
