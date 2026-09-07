"""Middlebury Portable Float Map (PFM) read/write helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_pfm(path: str | Path, data: np.ndarray, *, scale: float = -1.0) -> None:
    """
    Write a single-channel float32 PFM (Pf, little-endian).

    ``data`` is stored bottom-up per the PFM convention.
    Negative ``scale`` selects little-endian byte order.
    """
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"write_pfm expects [H,W], got {arr.shape}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = arr.shape
    flipped = np.flipud(arr).astype(np.float32, copy=False)
    header = f"Pf\n{w} {h}\n{scale}\n"
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        flipped.tofile(f)
