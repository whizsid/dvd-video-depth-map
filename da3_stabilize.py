"""DA3-style temporal stabilization helpers for DVD depth video."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FlowGraph:
    """Forward/backward flow and grayscale frames on one process grid."""

    h: int
    w: int
    flow_fwd: list[np.ndarray]
    flow_bwd: list[np.ndarray]
    gray: list[np.ndarray]


@dataclass(frozen=True)
class TemporalMedianParams:
    """Controls the flow-warped temporal median post-pass."""

    enabled: bool = True
    radius: int = 2
    photo_sigma: float = 0.04
    consistency_thresh: float = 2.0


@dataclass(frozen=True)
class StabilizeParams:
    """Shot-level stabilization config derived from video stats."""

    flow_long_side: int = 480
    farneback_levels: int = 3
    farneback_winsize: int = 21
    lock_percentile: float = 2.0
    temporal: TemporalMedianParams = TemporalMedianParams()
    chunk_frames: int = 96
    chunk_overlap: int = 8


class SharedPairCache:
    """Thread-safe Farneback pair cache: abs frame ``i`` → (fwd i→i+1, bwd)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._data

    def __getitem__(self, key: int) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self._data[key]

    def __setitem__(self, key: int, value: tuple[np.ndarray, np.ndarray]) -> None:
        with self._lock:
            self._data[key] = value

    def get(self, key: int, default=None):
        with self._lock:
            return self._data.get(key, default)

    def keys(self):
        with self._lock:
            return list(self._data.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def release_before(self, idx: int) -> None:
        with self._lock:
            for k in list(self._data):
                if k < idx:
                    del self._data[k]


def compute_flow_pair(
    frame_a_bgr: np.ndarray,
    frame_b_bgr: np.ndarray,
    process_w: int,
    process_h: int,
    *,
    flow_long_side: int = 480,
    farneback_levels: int = 3,
    farneback_winsize: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """Bidirectional Farneback between two BGR frames on the process grid."""

    def _gray(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape != (process_h, process_w):
            gray = cv2.resize(gray, (process_w, process_h), interpolation=cv2.INTER_AREA)
        return gray

    g0 = _gray(frame_a_bgr)
    g1 = _gray(frame_b_bgr)
    long_side = max(process_w, process_h)
    scale = min(1.0, float(flow_long_side) / max(long_side, 1))
    fw = max(16, int(round(process_w * scale)))
    fh = max(16, int(round(process_h * scale)))
    sx = process_w / float(fw)
    sy = process_h / float(fh)
    gf0 = g0 if (g0.shape[1], g0.shape[0]) == (fw, fh) else cv2.resize(g0, (fw, fh), interpolation=cv2.INTER_AREA)
    gf1 = g1 if (g1.shape[1], g1.shape[0]) == (fw, fh) else cv2.resize(g1, (fw, fh), interpolation=cv2.INTER_AREA)
    f = cv2.calcOpticalFlowFarneback(
        gf0, gf1, None, 0.5, farneback_levels, farneback_winsize, 3, 5, 1.1, 0
    )
    b = cv2.calcOpticalFlowFarneback(
        gf1, gf0, None, 0.5, farneback_levels, farneback_winsize, 3, 5, 1.1, 0
    )
    if (fw, fh) != (process_w, process_h):
        fx = cv2.resize(f[..., 0], (process_w, process_h), interpolation=cv2.INTER_LINEAR) * sx
        fy = cv2.resize(f[..., 1], (process_w, process_h), interpolation=cv2.INTER_LINEAR) * sy
        bx = cv2.resize(b[..., 0], (process_w, process_h), interpolation=cv2.INTER_LINEAR) * sx
        by = cv2.resize(b[..., 1], (process_w, process_h), interpolation=cv2.INTER_LINEAR) * sy
        f = np.stack([fx, fy], axis=-1).astype(np.float32)
        b = np.stack([bx, by], axis=-1).astype(np.float32)
    else:
        f = f.astype(np.float32)
        b = b.astype(np.float32)
    return f, b


class FlowPrefetchPool:
    """Background Farneback over InferResBgrStore pairs while DiT/VAE use MPS."""

    def __init__(
        self,
        bgr_store,
        *,
        process_w: int,
        process_h: int,
        params: StabilizeParams,
        pair_cache: SharedPairCache,
        workers: int = 1,
    ):
        self.bgr_store = bgr_store
        self.process_w = int(process_w)
        self.process_h = int(process_h)
        self.params = params
        self.pair_cache = pair_cache
        self._workers = max(1, int(workers))
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._lock = threading.Lock()
        self._inflight: set[int] = set()
        self._cached = 0
        self._threads: list[threading.Thread] = []
        for i in range(self._workers):
            t = threading.Thread(
                target=self._loop,
                name=f"dvd-flow-prefetch-{i}",
                daemon=True,
            )
            self._threads.append(t)
            t.start()
        print(
            f"[flow-prefetch] started workers={self._workers} "
            f"grid={process_w}x{process_h}",
            flush=True,
        )

    def set_workers(self, n: int) -> None:
        # Soft control: pause when n==0; otherwise keep existing threads.
        n = max(0, int(n))
        if n <= 0:
            self._pause.set()
        else:
            self._pause.clear()

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def stop(self) -> None:
        self._stop.set()
        self._pause.clear()
        for t in self._threads:
            t.join(timeout=2.0)
        print(
            f"[flow-prefetch] stopped (cached_pairs={self._cached})",
            flush=True,
        )

    def release_before(self, idx: int) -> None:
        self.pair_cache.release_before(idx)

    def _claim_work(self) -> int | None:
        """Pick and claim lowest abs index with both frames present and no cache."""
        try:
            indices = self.bgr_store.available_indices()
        except Exception:
            return None
        if len(indices) < 2:
            return None
        idx_set = set(indices)
        with self._lock:
            for i in indices:
                if (
                    (i + 1) in idx_set
                    and i not in self.pair_cache
                    and i not in self._inflight
                ):
                    self._inflight.add(i)
                    return i
        return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self._pause.is_set():
                time.sleep(0.05)
                continue
            # Shed if RAM is critically low.
            try:
                from resources import CRITICAL_FRAC, available_ram_bytes, total_ram_bytes

                total = total_ram_bytes()
                avail = available_ram_bytes()
                used_frac = 1.0 - (avail / max(total, 1))
                if used_frac >= CRITICAL_FRAC:
                    time.sleep(0.2)
                    continue
            except Exception:
                pass

            abs_i = self._claim_work()
            if abs_i is None:
                time.sleep(0.05)
                continue
            try:
                if abs_i in self.pair_cache:
                    continue
                pair = self.bgr_store.try_get_pair(abs_i)
                if pair is None:
                    continue
                frame_a, frame_b = pair
                try:
                    f, b = compute_flow_pair(
                        frame_a,
                        frame_b,
                        self.process_w,
                        self.process_h,
                        flow_long_side=self.params.flow_long_side,
                        farneback_levels=self.params.farneback_levels,
                        farneback_winsize=self.params.farneback_winsize,
                    )
                except Exception:
                    continue
                if abs_i not in self.pair_cache:
                    self.pair_cache[abs_i] = (f, b)
                    with self._lock:
                        self._cached += 1
                        n = self._cached
                    if n == 1 or n % 32 == 0:
                        print(f"[flow-prefetch] cached pairs={n}", flush=True)
            finally:
                with self._lock:
                    self._inflight.discard(abs_i)


def _robust_affine_samples(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    scale_clamp: tuple[float, float] = (0.5, 2.0),
    max_samples: int = 50000,
) -> tuple[float, float]:
    """Moment-matching affine ``dst ~= a * src + b`` on finite samples."""
    s = np.asarray(src, dtype=np.float64).reshape(-1)
    d = np.asarray(dst, dtype=np.float64).reshape(-1)
    m = np.isfinite(s) & np.isfinite(d)
    s, d = s[m], d[m]
    if s.size < 64:
        return 1.0, 0.0
    if s.size > max_samples:
        idx = np.linspace(0, s.size - 1, max_samples).astype(np.int64)
        s, d = s[idx], d[idx]
    ms, md = np.median(s), np.median(d)
    ss = 1.4826 * np.median(np.abs(s - ms))
    sd = 1.4826 * np.median(np.abs(d - md))
    if ss < 1e-8 or sd < 1e-8:
        return 1.0, float(md - ms)
    a = float(np.clip(sd / ss, scale_clamp[0], scale_clamp[1]))
    b = float(md - a * ms)
    if s.std() > 1e-9 and d.std() > 1e-9:
        corr = float(np.mean((s - s.mean()) * (d - d.mean())) / (s.std() * d.std()))
        if not np.isfinite(corr) or corr < 0.3:
            return 1.0, 0.0
    return a, b


def _align_disp_global(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    a, b = _robust_affine_samples(src, ref)
    return (a * src + b).astype(np.float32)


def lock_window_to_ref(
    disparities: list[np.ndarray],
    strategy: str = "middle",
) -> list[np.ndarray]:
    """Match every frame in a window onto one reference metric."""
    if not disparities:
        return disparities
    n = len(disparities)
    if strategy == "first":
        ri = 0
    elif strategy == "last":
        ri = n - 1
    else:
        ri = n // 2
    ref = np.asarray(disparities[ri], dtype=np.float32)
    out: list[np.ndarray] = []
    for i, d in enumerate(disparities):
        arr = np.asarray(d, dtype=np.float32)
        out.append(arr.copy() if i == ri else _align_disp_global(arr, ref))
    return out


def _finite_values(disp: np.ndarray) -> np.ndarray:
    v = np.asarray(disp, dtype=np.float64).reshape(-1)
    return v[np.isfinite(v)]


def _knot_values(disp: np.ndarray, pcts: tuple[float, ...]) -> np.ndarray | None:
    vals = _finite_values(disp)
    if vals.size < 64:
        return None
    knots = np.asarray([float(np.percentile(vals, p)) for p in pcts], dtype=np.float64)
    span = max(float(knots[-1] - knots[0]), 1e-3)
    eps = max(1e-6, 1e-4 * span)
    for i in range(1, knots.size):
        if knots[i] <= knots[i - 1]:
            knots[i] = knots[i - 1] + eps
    return knots


def _piecewise_linear_map(
    disp: np.ndarray,
    src_knots: np.ndarray,
    dst_knots: np.ndarray,
) -> np.ndarray:
    d = np.asarray(disp, dtype=np.float32)
    out = np.empty_like(d)
    finite = np.isfinite(d)
    if not np.any(finite):
        return d.copy()
    x = d[finite].astype(np.float64)
    y = np.empty_like(x)
    y[:] = np.interp(x, src_knots, dst_knots)
    s0 = (dst_knots[1] - dst_knots[0]) / max(src_knots[1] - src_knots[0], 1e-12)
    s1 = (dst_knots[-1] - dst_knots[-2]) / max(src_knots[-1] - src_knots[-2], 1e-12)
    left = x < src_knots[0]
    right = x > src_knots[-1]
    y[left] = dst_knots[0] + s0 * (x[left] - src_knots[0])
    y[right] = dst_knots[-1] + s1 * (x[right] - src_knots[-1])
    out[finite] = y.astype(np.float32)
    out[~finite] = d[~finite]
    return out


def apply_shot_band_lock(
    disparities: list[np.ndarray],
    *,
    knot_pcts: tuple[float, ...] = (10.0, 30.0, 50.0, 70.0, 90.0),
    adapt: float = 0.0,
) -> list[np.ndarray]:
    """Lock each frame onto a shot-canonical far/mid/near mapping."""
    if not disparities:
        return disparities
    canons: np.ndarray | None = None
    out: list[np.ndarray] = []
    n = len(disparities)
    t0 = time.perf_counter()
    for i, disp in enumerate(disparities):
        arr = np.asarray(disp, dtype=np.float32)
        knots = _knot_values(arr, knot_pcts)
        if knots is None:
            out.append(arr.copy())
        elif canons is None:
            canons = knots.copy()
            out.append(arr.copy())
        else:
            out.append(_piecewise_linear_map(arr, knots, canons))
            if adapt > 0.0:
                a = float(adapt)
                canons = (1.0 - a) * canons + a * knots
        done = i + 1
        if done == 1 or done % 32 == 0 or done == n:
            print(
                f"  [stabilize] band-lock {done}/{n} "
                f"({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
    return out


def _first_canonical_knots(
    disparities: list[np.ndarray],
    *,
    knot_pcts: tuple[float, ...] = (10.0, 30.0, 50.0, 70.0, 90.0),
) -> np.ndarray | None:
    for disp in disparities:
        knots = _knot_values(np.asarray(disp, dtype=np.float32), knot_pcts)
        if knots is not None:
            return knots.copy()
    return None


def apply_shot_band_lock_to_canonical(
    disparities: list[np.ndarray],
    canonical_knots: np.ndarray | None,
    *,
    knot_pcts: tuple[float, ...] = (10.0, 30.0, 50.0, 70.0, 90.0),
) -> list[np.ndarray]:
    """Map each frame onto a previously chosen shot canonical mapping."""
    if not disparities:
        return disparities
    if canonical_knots is None:
        return [np.asarray(d, dtype=np.float32).copy() for d in disparities]
    out: list[np.ndarray] = []
    for disp in disparities:
        arr = np.asarray(disp, dtype=np.float32)
        knots = _knot_values(arr, knot_pcts)
        if knots is None:
            out.append(arr.copy())
        else:
            out.append(_piecewise_linear_map(arr, knots, canonical_knots))
    return out


def _mid_stats(disp: np.ndarray, far_pct: float = 20.0, near_pct: float = 80.0) -> dict[str, float]:
    d = np.asarray(disp, dtype=np.float32)
    finite = d[np.isfinite(d)]
    if finite.size < 64:
        return {"mid_med": 0.0}
    lo = float(np.percentile(finite, far_pct))
    hi = float(np.percentile(finite, near_pct))
    mid = finite[(finite > lo) & (finite < hi)]
    if mid.size < 16:
        mid = finite
    return {"mid_med": float(np.median(mid))}


def _mid_mask(disp: np.ndarray, far_pct: float = 20.0, near_pct: float = 80.0) -> np.ndarray:
    d = np.asarray(disp, dtype=np.float32)
    finite = np.isfinite(d)
    if int(finite.sum()) < 64:
        return np.zeros(d.shape, dtype=bool)
    lo = float(np.percentile(d[finite], far_pct))
    hi = float(np.percentile(d[finite], near_pct))
    return finite & (d > lo) & (d < hi)


def _weighted_median_stack(vals: np.ndarray, wts: np.ndarray, chunk_rows: int = 128) -> np.ndarray:
    _, h, w = vals.shape
    out = np.empty((h, w), dtype=np.float32)
    for y0 in range(0, h, chunk_rows):
        y1 = min(h, y0 + chunk_rows)
        v = vals[:, y0:y1, :]
        wt = wts[:, y0:y1, :]
        order = np.argsort(v, axis=0)
        sv = np.take_along_axis(v, order, axis=0)
        sw = np.take_along_axis(wt, order, axis=0)
        cum = np.cumsum(sw, axis=0)
        target = 0.5 * cum[-1]
        idx = (cum >= target[None]).argmax(axis=0)
        out[y0:y1] = np.take_along_axis(sv, idx[None], axis=0)[0]
    return out


def compute_shot_flow(
    frames_bgr: list[np.ndarray],
    process_w: int,
    process_h: int,
    *,
    flow_long_side: int = 480,
    farneback_levels: int = 3,
    farneback_winsize: int = 21,
    pair_cache: dict | SharedPairCache | None = None,
    abs_start: int = 0,
) -> FlowGraph:
    """Compute bidirectional Farneback flow on the stabilization grid.

    ``pair_cache`` maps absolute frame index ``i`` → ``(flow_fwd[i→i+1], flow_bwd)``
    so overlapping chunks / background prefetch can reuse already-computed pairs.
    """
    n = len(frames_bgr)
    if n == 0:
        return FlowGraph(process_h, process_w, [], [], [])
    grays = []
    for frame in frames_bgr:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape != (process_h, process_w):
            gray = cv2.resize(gray, (process_w, process_h), interpolation=cv2.INTER_AREA)
        grays.append(gray)

    flow_fwd: list[np.ndarray] = []
    flow_bwd: list[np.ndarray] = []
    n_pairs = max(0, n - 1)
    t0 = time.perf_counter()
    for i in range(n_pairs):
        abs_i = int(abs_start) + i
        if pair_cache is not None and abs_i in pair_cache:
            f, b = pair_cache[abs_i]
            flow_fwd.append(f)
            flow_bwd.append(b)
            continue
        f, b = compute_flow_pair(
            frames_bgr[i],
            frames_bgr[i + 1],
            process_w,
            process_h,
            flow_long_side=flow_long_side,
            farneback_levels=farneback_levels,
            farneback_winsize=farneback_winsize,
        )
        if pair_cache is not None:
            pair_cache[abs_i] = (f, b)
        flow_fwd.append(f)
        flow_bwd.append(b)
        done = i + 1
        if done == 1 or done % 8 == 0 or done == n_pairs:
            elapsed = time.perf_counter() - t0
            rate = done / max(elapsed, 1e-3)
            eta = (n_pairs - done) / max(rate, 1e-3)
            print(
                f"  [stabilize] Farneback pairs {done}/{n_pairs} "
                f"({elapsed:.1f}s, ~{eta:.0f}s left)",
                flush=True,
            )
    return FlowGraph(process_h, process_w, flow_fwd, flow_bwd, grays)


def accumulate_flow(
    graph: FlowGraph,
    src: int,
    dst: int,
    *,
    remap_cache: dict[tuple[int, int], np.ndarray] | None = None,
) -> np.ndarray | None:
    """Compose flows to warp from frame ``src`` onto frame ``dst``."""
    key = (src, dst)
    if remap_cache is not None and key in remap_cache:
        return remap_cache[key]
    h, w = graph.h, graph.w
    if src == dst:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        out = np.stack([xx, yy], axis=-1)
        if remap_cache is not None:
            remap_cache[key] = out
        return out
    if src < 0 or dst < 0 or src >= len(graph.gray) or dst >= len(graph.gray):
        return None
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if dst > src:
        map_x, map_y = xx.copy(), yy.copy()
        for i in range(dst - 1, src - 1, -1):
            flow = graph.flow_bwd[i]
            fx = cv2.remap(flow[..., 0], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            fy = cv2.remap(flow[..., 1], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            map_x = map_x + fx
            map_y = map_y + fy
        out = np.stack([map_x, map_y], axis=-1).astype(np.float32)
        if remap_cache is not None:
            remap_cache[key] = out
        return out
    map_x, map_y = xx.copy(), yy.copy()
    for i in range(dst, src):
        flow = graph.flow_fwd[i]
        fx = cv2.remap(flow[..., 0], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        fy = cv2.remap(flow[..., 1], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        map_x = map_x + fx
        map_y = map_y + fy
    out = np.stack([map_x, map_y], axis=-1).astype(np.float32)
    if remap_cache is not None:
        remap_cache[key] = out
    return out


def warp_with_map(image: np.ndarray, remap: np.ndarray, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    return cv2.remap(image, remap[..., 0], remap[..., 1], interpolation, borderMode=cv2.BORDER_REFLECT)


def flow_consistency_mask(flow_fwd: np.ndarray, flow_bwd: np.ndarray, thresh: float) -> np.ndarray:
    h, w = flow_fwd.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x1 = xx + flow_fwd[..., 0]
    y1 = yy + flow_fwd[..., 1]
    bx = cv2.remap(flow_bwd[..., 0], x1, y1, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    by = cv2.remap(flow_bwd[..., 1], x1, y1, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    rx = flow_fwd[..., 0] + bx
    ry = flow_fwd[..., 1] + by
    err = np.sqrt(rx * rx + ry * ry)
    t = max(float(thresh), 0.5)
    return np.clip(1.0 - err / (2.0 * t), 0.0, 1.0).astype(np.float32)


def pairwise_consistency(
    graph: FlowGraph,
    src: int,
    dst: int,
    thresh: float,
    *,
    remap_cache: dict[tuple[int, int], np.ndarray] | None = None,
) -> np.ndarray | None:
    if src == dst:
        return np.ones((graph.h, graph.w), dtype=np.float32)
    if src < 0 or dst < 0 or src >= len(graph.gray) or dst >= len(graph.gray):
        return None
    if abs(src - dst) == 1:
        i = min(src, dst)
        mask_i = flow_consistency_mask(graph.flow_fwd[i], graph.flow_bwd[i], thresh)
        if dst == i:
            return mask_i
        remap = accumulate_flow(graph, i, dst, remap_cache=remap_cache)
        if remap is None:
            return None
        return warp_with_map(mask_i, remap).astype(np.float32)
    acc = np.ones((graph.h, graph.w), dtype=np.float32)
    step = 1 if dst > src else -1
    for a in range(src, dst, step):
        b = a + step
        m = pairwise_consistency(graph, a, b, thresh, remap_cache=remap_cache)
        if m is None:
            return None
        if b != dst:
            remap = accumulate_flow(graph, b, dst, remap_cache=remap_cache)
            if remap is None:
                return None
            m = warp_with_map(m, remap)
        acc = acc * m.astype(np.float32)
    return acc.astype(np.float32)


def apply_temporal_median(
    disparities: list[np.ndarray],
    flow_graph: FlowGraph,
    params: TemporalMedianParams,
) -> list[np.ndarray]:
    """Flow-warped temporal weighted median with dark-sample rejection."""
    if not params.enabled or params.radius <= 0 or not disparities:
        return disparities
    n = len(disparities)
    h, w = flow_graph.h, flow_graph.w
    inv_2sp = 1.0 / (2.0 * params.photo_sigma * params.photo_sigma)
    remap_cache: dict[tuple[int, int], np.ndarray] = {}
    out: list[np.ndarray] = []
    for t in range(n):
        ref = np.asarray(disparities[t], dtype=np.float32)
        if ref.shape[:2] != (h, w):
            ref = cv2.resize(ref, (w, h), interpolation=cv2.INTER_LINEAR)
        ref_gray = flow_graph.gray[t].astype(np.float32) / 255.0
        vals: list[np.ndarray] = [ref]
        wts: list[np.ndarray] = [np.full((h, w), 1.25, dtype=np.float32)]
        for dt in range(-params.radius, params.radius + 1):
            if dt == 0:
                continue
            src = t + dt
            if src < 0 or src >= n:
                continue
            remap = accumulate_flow(flow_graph, src, t, remap_cache=remap_cache)
            if remap is None:
                continue
            cons = pairwise_consistency(
                flow_graph, src, t, params.consistency_thresh, remap_cache=remap_cache
            )
            if cons is None:
                continue
            src_disp = np.asarray(disparities[src], dtype=np.float32)
            if src_disp.shape[:2] != (h, w):
                src_disp = cv2.resize(src_disp, (w, h), interpolation=cv2.INTER_LINEAR)
            warped = warp_with_map(src_disp, remap)
            src_g = flow_graph.gray[src].astype(np.float32) / 255.0
            warped_g = warp_with_map(src_g, remap)
            diff = ref_gray - warped_g
            photo_w = np.exp(-(diff * diff) * inv_2sp).astype(np.float32)
            gate = np.where((cons * photo_w) >= 0.30, (cons * photo_w), 0.0).astype(np.float32)
            if float(gate.max()) < 0.30:
                continue
            dist_w = 1.0 / (1.0 + abs(dt))
            aligned = _align_disp_global(warped, ref)
            dark = aligned < ref
            sample_w = (0.95 * dist_w * gate).astype(np.float32)
            sample_w = np.where(dark, sample_w * 0.12, sample_w).astype(np.float32)
            vals.append(aligned)
            wts.append(sample_w)
        if len(vals) == 1:
            frame = ref.copy()
        else:
            V = np.stack(vals, axis=0)
            W = np.maximum(np.stack(wts, axis=0), 1e-6)
            frame = _weighted_median_stack(V, W)
        mid = _mid_mask(ref)
        frame = np.where(mid & (frame < ref), ref, frame).astype(np.float32)
        before = _mid_stats(ref)["mid_med"]
        after = _mid_stats(frame)["mid_med"]
        shift = float(before - after)
        if abs(shift) > 1e-6:
            frame = (frame + shift).astype(np.float32)
        out.append(frame)
    return out


def chunk_ranges(total: int, chunk_frames: int, overlap: int) -> list[tuple[int, int]]:
    """Build overlapping ranges; each chunk keeps a central keep-region."""
    total = max(0, int(total))
    chunk_frames = max(1, int(chunk_frames))
    overlap = max(0, min(int(overlap), chunk_frames - 1))
    if total <= chunk_frames:
        return [(0, total)]
    step = max(1, chunk_frames - overlap)
    out: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(total, start + chunk_frames)
        out.append((start, end))
        if end >= total:
            break
        start += step
        if start + chunk_frames > total:
            start = max(0, total - chunk_frames)
    uniq: list[tuple[int, int]] = []
    seen = set()
    for item in out:
        if item not in seen:
            seen.add(item)
            uniq.append(item)
    return uniq


def derive_stabilize_params(
    *,
    fps: float,
    process_w: int,
    process_h: int,
    frame_count: int,
    flow_long_side: int | None = None,
) -> StabilizeParams:
    """Small DA3-inspired heuristic set for DVD post-stabilization."""
    long_side = max(process_w, process_h)
    flow_side = flow_long_side or min(long_side, 480)
    # Colab ~12GiB hosts: Farneback at 480 on 600+ frames looks "hung" for many minutes.
    low_host = False
    try:
        import psutil

        low_host = psutil.virtual_memory().total <= int(14 * 1024**3)
    except Exception:
        pass
    if os.environ.get("COLAB_GPU") or os.environ.get("COLAB_RELEASE_TAG"):
        low_host = True
    if low_host and flow_long_side is None:
        flow_side = min(flow_side, 256)
    motion_radius = max(2, int(round(0.10 * fps)))
    if low_host:
        motion_radius = min(motion_radius, 2)
    chunk_frames = int(max(24, min(frame_count, 96)))
    if low_host:
        chunk_frames = int(max(24, min(frame_count, 48)))
    chunk_overlap = max(4, min(chunk_frames // 4, int(round(max(2.0, 0.25 * fps)))))
    return StabilizeParams(
        flow_long_side=flow_side,
        farneback_levels=3 if low_host or max(process_w, process_h) < 640 else 4,
        farneback_winsize=max(11, int(round(21 * flow_side / 480.0)) | 1),
        temporal=TemporalMedianParams(
            enabled=True,
            radius=motion_radius,
            photo_sigma=0.04,
            consistency_thresh=float(np.clip(1.0 + 0.5 * (480.0 / max(process_w, 1)), 1.0, 4.0)),
        ),
        chunk_frames=chunk_frames,
        chunk_overlap=chunk_overlap,
    )


def probe_shot_ranges(
    video_path: str | Path,
    *,
    max_samples: int = 30,
    probe_long_side: int = 320,
    cut_thresh: float = 0.45,
) -> tuple[tuple[int, int], ...]:
    """Return DA3-like shot ranges from the input video."""
    path = str(video_path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        frames = 0
        while cap.grab():
            frames += 1
        cap.release()
        cap = cv2.VideoCapture(path)
        frame_count = frames
    cut_step = max(1, int(round(fps * 0.25)))
    n_samp = min(max_samples, max(2, frame_count))
    sample_indices = {int(i) for i in np.linspace(0, max(frame_count - 1, 0), n_samp, dtype=np.int64)}
    needed = set(range(0, frame_count, cut_step)) | sample_indices
    tw, th = 64, max(36, int(round(64 * height / max(width, 1))))
    prev_cut_gray: np.ndarray | None = None
    cuts: list[int] = []
    i = 0
    while i < frame_count:
        if i not in needed:
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g_cut = cv2.resize(gray_full, (tw, th), interpolation=cv2.INTER_AREA)
        if prev_cut_gray is not None:
            ha = cv2.calcHist([prev_cut_gray], [0], None, [64], [0, 256])
            hb = cv2.calcHist([g_cut], [0], None, [64], [0, 256])
            cv2.normalize(ha, ha)
            cv2.normalize(hb, hb)
            if float(cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA)) >= cut_thresh:
                cuts.append(i)
        prev_cut_gray = g_cut
        i += 1
    cap.release()
    n = max(1, frame_count)
    starts = [0] + sorted(j for j in cuts if 0 < j < n)
    out: list[tuple[int, int]] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else n
        if end > start:
            out.append((start, end))
    return tuple(out or [(0, n)])


def lock_range(
    disparities: list[np.ndarray] | np.ndarray,
    percentile: float = 2.0,
    sample_per_frame: int = 20000,
) -> tuple[float, float]:
    """Sample-based percentile display range, matching DA3 behavior."""
    if isinstance(disparities, np.ndarray):
        frames = [np.asarray(disparities[i], dtype=np.float32) for i in range(disparities.shape[0])]
    else:
        frames = [np.asarray(d, dtype=np.float32) for d in disparities]
    samples: list[np.ndarray] = []
    for d in frames:
        flat = d.reshape(-1)
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            continue
        if flat.size > sample_per_frame:
            idx = np.random.randint(0, flat.size, size=sample_per_frame)
            flat = flat[idx]
        samples.append(flat)
    if not samples:
        return 0.0, 1.0
    all_s = np.concatenate(samples)
    lo = float(np.percentile(all_s, float(percentile)))
    hi = float(np.percentile(all_s, 100.0 - float(percentile)))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def normalize_disparity_to_bgr_u8(
    disp: np.ndarray,
    lo: float,
    hi: float,
    *,
    invert: bool = False,
) -> np.ndarray:
    """Map a disparity frame to DA3-style BGR grayscale."""
    span = max(1e-6, float(hi) - float(lo))
    norm = np.clip((np.asarray(disp, dtype=np.float32) - lo) / span, 0.0, 1.0)
    if invert:
        norm = 1.0 - norm
    gray_u8 = (norm * 255.0).astype(np.uint8)
    return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)


def _resize_bgr(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    if frame.shape[0] != out_h or frame.shape[1] != out_w:
        return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return frame


def _open_stabilize_capture(video_path: str | Path) -> cv2.VideoCapture:
    path = str(video_path)
    for backend in (getattr(cv2, "CAP_FFMPEG", 0), 0):
        cap = cv2.VideoCapture(path, backend) if backend else cv2.VideoCapture(path)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            return cap
    raise RuntimeError(f"Cannot open video for stabilization: {video_path}")


def _decode_video_range(
    video_path: str | Path,
    start: int,
    end: int,
    out_w: int,
    out_h: int,
) -> list[np.ndarray]:
    """Decode ``[start, end)`` with seek, falling back to linear grab on failure.

    OpenCV's ``CAP_PROP_POS_FRAMES`` often *reports* success on H.264 while the
    next ``read()`` fails — especially on late chunks. Prefer seek for speed,
    then retry from frame 0 when the result is short.
    """
    need = max(0, int(end) - int(start))
    if need == 0:
        return []
    start = int(start)
    end = int(end)

    def _grab_skip(cap: cv2.VideoCapture, until: int) -> int:
        idx = 0
        while idx < until:
            if not cap.grab():
                break
            idx += 1
        return idx

    def _read_forward(cap: cv2.VideoCapture, idx: int) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        while idx < end:
            ok, frame = cap.read()
            if not ok:
                break
            if idx >= start:
                out.append(_resize_bgr(frame, out_w, out_h))
            idx += 1
        return out

    def _linear_decode() -> list[np.ndarray]:
        cap = _open_stabilize_capture(video_path)
        try:
            idx = _grab_skip(cap, start)
            if idx < start:
                return []
            return _read_forward(cap, idx)
        finally:
            cap.release()

    # Attempt 1: seek (fast path for late chunks / Drive).
    frames: list[np.ndarray] = []
    use_linear = start <= 0
    if start > 0:
        cap = _open_stabilize_capture(video_path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            if abs(pos - start) > 2:
                use_linear = True
            else:
                frames = _read_forward(cap, start)
        finally:
            cap.release()

    if use_linear:
        frames = _linear_decode()
    elif len(frames) < need:
        got = len(frames)
        print(
            f"  [stabilize] seek decode short ({got}/{need} frames for "
            f"[{start}:{end})); retrying linear grab from 0",
            flush=True,
        )
        frames = _linear_decode()

    return frames


def _pad_frame_list(
    frames: list[np.ndarray | None],
    *,
    start: int,
    bgr_store: object | None,
) -> list[np.ndarray]:
    """Fill holes / trailing EOF with nearest available frame (matches infer pad)."""
    n = len(frames)
    if n == 0:
        return []
    last: np.ndarray | None = None
    for i in range(n):
        if frames[i] is not None:
            last = frames[i]
        elif last is not None:
            filled = last.copy()
            frames[i] = filled
            if bgr_store is not None:
                try:
                    bgr_store.put(start + i, filled)  # type: ignore[attr-defined]
                except Exception:
                    pass
    first = next((f for f in frames if f is not None), None)
    if first is None:
        raise RuntimeError(
            f"Failed to read stabilization frames [{start}:{start + n}) "
            f"(no decodable frames)"
        )
    for i in range(n):
        if frames[i] is None:
            filled = first.copy()
            frames[i] = filled
            if bgr_store is not None:
                try:
                    bgr_store.put(start + i, filled)  # type: ignore[attr-defined]
                except Exception:
                    pass
    return [frames[i] for i in range(n)]  # type: ignore[misc]


def _read_video_frames(
    video_path: str | Path,
    start: int,
    end: int,
    out_w: int,
    out_h: int,
    bgr_store: object | None = None,
) -> list[np.ndarray]:
    need = max(0, int(end) - int(start))
    if need == 0:
        return []
    start = int(start)
    end = int(end)
    slots: list[np.ndarray | None] = [None] * need

    if bgr_store is not None:
        try:
            return bgr_store.get_range(start, end)  # type: ignore[attr-defined]
        except KeyError as exc:
            existing: dict[int, np.ndarray] = {}
            getter = getattr(bgr_store, "get_existing", None)
            if callable(getter):
                try:
                    existing = getter(start, end)
                except Exception:
                    existing = {}
            for idx, frame in existing.items():
                li = idx - start
                if 0 <= li < need:
                    slots[li] = frame
            n_hit = sum(1 for f in slots if f is not None)
            print(
                f"  [stabilize] bgr_store miss ({exc}); "
                f"have {n_hit}/{need} cached, decoding gaps "
                f"[{start}:{end})",
                flush=True,
            )

    if all(f is not None for f in slots):
        return [f for f in slots if f is not None]  # type: ignore[misc]

    missing = [start + i for i, f in enumerate(slots) if f is None]
    decode_start = missing[0]
    decode_end = missing[-1] + 1
    decoded = _decode_video_range(
        video_path, decode_start, decode_end, out_w, out_h
    )
    for offset, frame in enumerate(decoded):
        abs_i = decode_start + offset
        li = abs_i - start
        if 0 <= li < need and slots[li] is None:
            slots[li] = frame
            if bgr_store is not None:
                try:
                    bgr_store.put(abs_i, frame)  # type: ignore[attr-defined]
                except Exception:
                    pass

    n_got = sum(1 for f in slots if f is not None)
    if n_got < need:
        print(
            f"  [stabilize] video ended early for [{start}:{end}) "
            f"({n_got}/{need} frames); padding with last frame",
            flush=True,
        )
    return _pad_frame_list(slots, start=start, bgr_store=bgr_store)


def apply_flow_temporal_median(
    video_path: str | Path,
    disparities: list[np.ndarray],
    *,
    process_w: int,
    process_h: int,
    params: StabilizeParams,
    global_start: int = 0,
    bgr_store: object | None = None,
) -> list[np.ndarray]:
    """Chunked DA3-style temporal median over one shot's disparities."""
    n = len(disparities)
    if n == 0 or not params.temporal.enabled:
        return [np.asarray(d, dtype=np.float32).copy() for d in disparities]
    out: list[np.ndarray | None] = [None] * n
    ranges = chunk_ranges(n, params.chunk_frames, params.chunk_overlap)
    half_overlap = params.chunk_overlap // 2
    pair_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    print(
        f"  [stabilize] temporal median: {n} frames in {len(ranges)} chunks "
        f"(chunk={params.chunk_frames}, flow_side={params.flow_long_side})",
        flush=True,
    )
    for chunk_idx, (s0, s1) in enumerate(ranges):
        t_chunk = time.perf_counter()
        print(
            f"  [stabilize] chunk {chunk_idx + 1}/{len(ranges)} "
            f"frames [{global_start + s0}:{global_start + s1})",
            flush=True,
        )
        frames_bgr = _read_video_frames(
            video_path,
            global_start + s0,
            global_start + s1,
            process_w,
            process_h,
            bgr_store=bgr_store,
        )
        graph = compute_shot_flow(
            frames_bgr,
            process_w,
            process_h,
            flow_long_side=params.flow_long_side,
            farneback_levels=params.farneback_levels,
            farneback_winsize=params.farneback_winsize,
            pair_cache=pair_cache,
            abs_start=global_start + s0,
        )
        local = [np.asarray(d, dtype=np.float32) for d in disparities[s0:s1]]
        local_out = apply_temporal_median(local, graph, params.temporal)
        keep0 = 0 if chunk_idx == 0 else half_overlap
        keep1 = len(local_out) if chunk_idx == len(ranges) - 1 else max(keep0, len(local_out) - half_overlap)
        for li in range(keep0, keep1):
            out[s0 + li] = local_out[li]
        # Drop pair flows that fall fully behind the emitted keep-region.
        emit_upto = s0 + keep1
        drop_before = global_start + max(0, emit_upto - params.temporal.radius - 2)
        for key in list(pair_cache):
            if key < drop_before:
                del pair_cache[key]
        if bgr_store is not None:
            try:
                bgr_store.release_before(drop_before)  # type: ignore[attr-defined]
            except Exception:
                pass
        print(
            f"  [stabilize] chunk {chunk_idx + 1}/{len(ranges)} done "
            f"({time.perf_counter() - t_chunk:.1f}s)",
            flush=True,
        )
    return [
        np.asarray(out[i], dtype=np.float32) if out[i] is not None else np.asarray(disparities[i], dtype=np.float32)
        for i in range(n)
    ]


@dataclass
class IncrementalShotStabilizer:
    """Stabilize one shot incrementally while preserving chunk semantics."""

    video_path: str | Path
    process_w: int
    process_h: int
    params: StabilizeParams
    global_start: int
    knot_pcts: tuple[float, ...] = (10.0, 30.0, 50.0, 70.0, 90.0)
    bgr_store: object | None = None
    pair_cache: SharedPairCache | dict | None = None

    def __post_init__(self) -> None:
        self.video_path = str(self.video_path)
        self._raw: list[np.ndarray] = []
        self._canonical_knots: np.ndarray | None = None
        self._next_chunk_start = 0
        self._emitted_upto = 0
        self._step = max(1, self.params.chunk_frames - self.params.chunk_overlap)
        self._half_overlap = self.params.chunk_overlap // 2
        if self.pair_cache is None:
            self.pair_cache = SharedPairCache()

    def _ensure_canonical(self) -> None:
        if self._canonical_knots is None:
            self._canonical_knots = _first_canonical_knots(
                self._raw, knot_pcts=self.knot_pcts
            )

    def _run_chunk(self, start: int, end: int) -> list[np.ndarray]:
        local = [np.asarray(d, dtype=np.float32) for d in self._raw[start:end]]
        local = apply_shot_band_lock_to_canonical(
            local,
            self._canonical_knots,
            knot_pcts=self.knot_pcts,
        )
        frames_bgr = _read_video_frames(
            self.video_path,
            self.global_start + start,
            self.global_start + end,
            self.process_w,
            self.process_h,
            bgr_store=self.bgr_store,
        )
        graph = compute_shot_flow(
            frames_bgr,
            self.process_w,
            self.process_h,
            flow_long_side=self.params.flow_long_side,
            farneback_levels=self.params.farneback_levels,
            farneback_winsize=self.params.farneback_winsize,
            pair_cache=self.pair_cache,
            abs_start=self.global_start + start,
        )
        return apply_temporal_median(local, graph, self.params.temporal)

    def append(
        self, disparities: list[np.ndarray], *, finalize: bool = False
    ) -> list[np.ndarray]:
        self._raw.extend(np.asarray(d, dtype=np.float32) for d in disparities)
        self._ensure_canonical()
        emitted: list[np.ndarray] = []
        total = len(self._raw)
        while self._next_chunk_start < total:
            start = self._next_chunk_start
            if not finalize and start + self.params.chunk_frames > total:
                break
            end = min(total, start + self.params.chunk_frames)
            local_out = self._run_chunk(start, end)
            keep0 = 0 if start == 0 else self._half_overlap
            if finalize and end == total:
                keep1 = len(local_out)
            else:
                keep1 = max(keep0, len(local_out) - self._half_overlap)
            abs_keep0 = start + keep0
            abs_keep1 = start + keep1
            if abs_keep1 > self._emitted_upto:
                trim = max(0, self._emitted_upto - abs_keep0)
                emit0 = keep0 + trim
                emit1 = keep1
                emitted.extend(local_out[emit0:emit1])
                self._emitted_upto = abs_keep1
            # Drop pair flows / BGR behind the emitted frontier (keep temporal halo).
            drop_before = self.global_start + max(
                0, self._emitted_upto - self.params.temporal.radius - 2
            )
            if isinstance(self.pair_cache, SharedPairCache):
                self.pair_cache.release_before(drop_before)
            else:
                assert self.pair_cache is not None
                for key in list(self.pair_cache.keys()):
                    if key < drop_before:
                        del self.pair_cache[key]
            if self.bgr_store is not None:
                try:
                    self.bgr_store.release_before(drop_before)  # type: ignore[attr-defined]
                except Exception:
                    pass
            if end >= total:
                if finalize:
                    self._next_chunk_start = total
                break
            self._next_chunk_start += self._step
        return emitted
