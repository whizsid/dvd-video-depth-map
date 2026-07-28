"""Resource governor: size DVD pipeline concurrency from live RAM / CPU / MPS."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace

try:
    import psutil
except ImportError:  # optional
    psutil = None  # type: ignore[assignment]


# Leave a thin OS headroom on unified memory (Apple Silicon shares RAM/MPS).
RAM_HARD_FRAC = 0.95
TARGET_LO = 0.85
TARGET_HI = 0.93
CRITICAL_FRAC = 0.95
PRESSURE_TARGET = 0.90
# Soft-cap system CPU so VAE / JBU leave headroom for the OS + decode threads.
CPU_SOFT_CAP = 0.95

# How many decoded windows prep may buffer ahead of DVD infer.
DEFAULT_PREP_QUEUE_DEPTH = 2
MAX_PREP_QUEUE_DEPTH = 6


@dataclass
class ResourceSnapshot:
    ram_used_frac_of_budget: float
    cpu_frac: float
    mps_frac_of_budget: float
    available_ram_bytes: int
    total_ram_bytes: int
    mps_allocated: int | None
    mps_recommended: int | None
    upsample_workers: int
    prep_queue: int
    infer_queue: int
    pause_prep: bool
    queue_depth: int
    last_action: str = ""


@dataclass
class ConcurrencyPlan:
    upsample_workers: int
    prep_queue_depth: int = DEFAULT_PREP_QUEUE_DEPTH
    flow_prefetch_workers: int = 1
    pause_prep: bool = False
    last_action: str = "hold"


def cpu_count() -> int:
    return max(1, os.cpu_count() or 4)


def max_cpu_threads(cap_frac: float = CPU_SOFT_CAP) -> int:
    """Torch/OpenMP/OpenCV thread budget so aggregate CPU stays ≤ ``cap_frac``."""
    n = cpu_count()
    capped = int(n * float(cap_frac))
    if capped >= n and n > 1:
        capped = n - 1
    return max(1, capped)


def total_ram_bytes() -> int:
    if psutil is not None:
        try:
            return int(psutil.virtual_memory().total)
        except Exception:
            pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:
        return 16 * 1024**3


def available_ram_bytes() -> int:
    if psutil is not None:
        try:
            return int(psutil.virtual_memory().available)
        except Exception:
            pass
    return max(1, total_ram_bytes() // 2)


def _cpu_frac() -> float:
    if psutil is not None:
        try:
            return float(psutil.cpu_percent(interval=0.0)) / 100.0
        except Exception:
            pass
    return 0.0


def _accelerator_memory() -> tuple[int | None, int | None]:
    """Return (allocated_bytes, recommended/total_bytes) for CUDA or MPS."""
    try:
        import torch

        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            return int(total_b - free_b), int(total_b)
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            return None, None
        rec = getattr(torch.mps, "recommended_max_memory", None)
        alloc = getattr(torch.mps, "current_allocated_memory", None)
        if callable(rec) and callable(alloc):
            return int(alloc()), int(rec())
    except Exception:
        return None, None
    return None, None


def _usable_ram_budget(total: int) -> int:
    return max(1, int(total * RAM_HARD_FRAC))


class ResourceGovernor:
    """
    Size CPU upsample workers + prep queue depth from live RAM / CPU / MPS.

    Targets high RAM/MPS utilization, but keeps system CPU ≤ ``CPU_SOFT_CAP``
    (default 95%) by capping thread budgets and shedding JBU workers.
    """

    def __init__(
        self,
        *,
        upsample_workers: str | int = "auto",
        frame_w: int = 1920,
        frame_h: int = 1080,
        infer_w: int = 320,
        infer_h: int = 192,
        window_frames: int = 5,
    ) -> None:
        self.upsample_arg = upsample_workers
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.infer_w = infer_w
        self.infer_h = infer_h
        self.window_frames = max(1, window_frames)
        self._plan = self._initial_plan()
        self._last_adjust = 0.0
        self._last_cpu_kick = time.time()
        self._underutil_streak = 0
        _cpu_frac()  # warm psutil counter

    def _parse(self, arg: str | int, default: int, cap: int) -> int:
        if isinstance(arg, int):
            return max(1, min(int(arg), cap))
        if isinstance(arg, str) and arg.lower() != "auto":
            try:
                return max(1, min(int(arg), cap))
            except ValueError:
                return default
        return default

    def _worker_cap(self) -> int:
        """Max JBU workers under the CPU soft-cap (leave room for VAE threads)."""
        return max(1, max_cpu_threads())

    def _fullres_frame_bytes(self) -> int:
        # float32 guide + float16 output temp + overhead
        return max(1, self.frame_w * self.frame_h * (4 + 2 + 4))

    def _prep_window_bytes(self) -> int:
        # float16 RGB window buffered in prep queue
        return max(
            1,
            self.window_frames * 3 * self.infer_h * self.infer_w * 2,
        )

    def _max_prep_depth(self) -> int:
        total = total_ram_bytes()
        # Prep is cheap (~0.3s/window); buffering many windows on 8GB steals RAM
        # from VAE and causes swap thrash. Keep the queue shallow.
        if total <= 9 * 1024**3:
            return 2
        return max(MAX_PREP_QUEUE_DEPTH, 8)

    def _initial_plan(self) -> ConcurrencyPlan:
        cpu_cap = self._worker_cap()
        avail = available_ram_bytes()
        total = total_ram_bytes()
        budget = _usable_ram_budget(total)
        used = max(0, total - avail)
        headroom = max(1, budget - used)

        ram_cap_up = max(1, int(headroom * 0.45) // self._fullres_frame_bytes())
        auto_up = max(1, min(cpu_cap, max(ram_cap_up, cpu_cap // 2)))

        # More RAM → allow prep to buffer further ahead.
        ram_cap_prep = max(1, int(headroom * 0.20) // self._prep_window_bytes())
        prep_depth = max(1, min(self._max_prep_depth(), ram_cap_prep))
        if total <= 9 * 1024**3:
            prep_depth = min(prep_depth, 2)
            # Keep JBU quiet until post actually has frames; VAE needs the cores/RAM.
            auto_up = max(1, min(2, cpu_cap))

        return ConcurrencyPlan(
            upsample_workers=self._parse(self.upsample_arg, auto_up, cpu_cap),
            prep_queue_depth=max(1, prep_depth),
            flow_prefetch_workers=1 if total <= 9 * 1024**3 else 2,
            pause_prep=False,
            last_action="init",
        )

    def plan(self) -> ConcurrencyPlan:
        return self._plan

    def snapshot(self, *, prep_queue: int = 0, infer_queue: int = 0) -> ResourceSnapshot:
        total = total_ram_bytes()
        avail = available_ram_bytes()
        used = max(0, total - avail)
        budget = _usable_ram_budget(total)
        ram_frac = used / float(budget)

        now = time.time()
        if now - self._last_cpu_kick >= 0.5 and psutil is not None:
            try:
                cpu = float(psutil.cpu_percent(interval=0.05)) / 100.0
            except Exception:
                cpu = _cpu_frac()
            self._last_cpu_kick = now
        else:
            cpu = _cpu_frac()

        alloc, rec = _accelerator_memory()
        if alloc is not None and rec is not None and rec > 0:
            mps_budget = max(1, int(rec * RAM_HARD_FRAC))
            mps_frac = alloc / float(mps_budget)
        else:
            mps_frac = 0.0

        return ResourceSnapshot(
            ram_used_frac_of_budget=float(ram_frac),
            cpu_frac=float(cpu),
            mps_frac_of_budget=float(mps_frac),
            available_ram_bytes=avail,
            total_ram_bytes=total,
            mps_allocated=alloc,
            mps_recommended=rec,
            upsample_workers=self._plan.upsample_workers,
            prep_queue=prep_queue,
            infer_queue=infer_queue,
            pause_prep=self._plan.pause_prep,
            queue_depth=self._plan.prep_queue_depth,
            last_action=self._plan.last_action,
        )

    def adjust(
        self,
        *,
        prep_queue: int = 0,
        infer_queue: int = 0,
        pending_cpu_work: bool = True,
        pending_gpu_work: bool = True,
        force: bool = False,
    ) -> ConcurrencyPlan:
        now = time.time()
        if not force and now - self._last_adjust < 0.5:
            return self._plan
        self._last_adjust = now

        snap = self.snapshot(prep_queue=prep_queue, infer_queue=infer_queue)
        up = self._plan.upsample_workers
        depth = self._plan.prep_queue_depth
        flow_n = self._plan.flow_prefetch_workers
        pause = False
        action = "hold"
        cap = self._worker_cap()
        max_depth = self._max_prep_depth()
        up_auto = isinstance(self.upsample_arg, str) and self.upsample_arg.lower() == "auto"
        cpu_over = snap.cpu_frac >= CPU_SOFT_CAP
        low_mem = snap.total_ram_bytes <= 9 * 1024**3
        flow_cap = 1 if low_mem else 2

        if snap.ram_used_frac_of_budget >= CRITICAL_FRAC:
            if up_auto:
                up = max(1, up - 1)
            depth = max(1, depth - 1)
            flow_n = 0
            pause = True
            action = "shed"
            self._underutil_streak = 0
        elif cpu_over:
            # Hard CPU soft-cap: shed before anything else boosts concurrency.
            if up_auto and up > 1:
                up = max(1, up - 1)
            if depth > 1:
                depth = max(1, depth - 1)
            flow_n = 0
            if snap.cpu_frac >= CPU_SOFT_CAP + 0.02:
                pause = True
            action = "trim_cpu_cap"
            self._underutil_streak = 0
        elif snap.ram_used_frac_of_budget > TARGET_HI:
            if up_auto and up > 1:
                up -= 1
            if depth > 1:
                depth -= 1
            flow_n = min(flow_n, 1)
            action = "trim_ram"
            self._underutil_streak = 0
        elif pending_gpu_work and snap.mps_frac_of_budget < 0.15:
            # Pre-DiT / VAE phase (MPS idle): do not grow prep or JBU — that was
            # stealing RAM from VAE on 8GB (prep looked "hung", infer was thrashing).
            if up_auto:
                up = min(up, 2 if low_mem else max(1, cap // 2))
            if low_mem:
                depth = min(depth, 2)
            flow_n = 0  # leave CPU quiet so VAE can claim RAM/cores next
            action = "hold_for_vae"
            self._underutil_streak = 0
        else:
            # MPS busy (DiT/VAE on Metal): fill idle CPU with prep + JBU + flow prefetch.
            if pending_gpu_work and snap.mps_frac_of_budget > 0.5:
                if (
                    up_auto
                    and up < cap
                    and snap.ram_used_frac_of_budget < TARGET_HI
                    and snap.cpu_frac < CPU_SOFT_CAP
                ):
                    up = min(cap, up + 1)
                    action = "boost_workers_mps"
                if (
                    snap.ram_used_frac_of_budget < TARGET_HI
                    and snap.cpu_frac < CPU_SOFT_CAP
                    and depth < max_depth
                ):
                    depth = min(max_depth, depth + 1)
                    if action == "hold":
                        action = "boost_prep_mps"
                    elif "prep" not in action:
                        action = f"{action}+prep"
                if (
                    snap.ram_used_frac_of_budget < TARGET_HI
                    and snap.cpu_frac < CPU_SOFT_CAP
                ):
                    flow_n = flow_cap
                    if action == "hold":
                        action = "boost_flow_mps"
            elif pending_gpu_work and snap.mps_frac_of_budget < 0.5:
                # Transition / light MPS: keep some CPU work but don't stampede.
                flow_n = min(flow_n, 1)
                if up_auto and up > max(1, cap // 2):
                    up = max(1, up - 1)
                    action = "yield_cpu_for_infer"
            elif (not pending_gpu_work) and pending_cpu_work:
                if (
                    up_auto
                    and up < cap
                    and snap.ram_used_frac_of_budget < TARGET_HI
                    and snap.cpu_frac < CPU_SOFT_CAP
                ):
                    up = min(cap, up + 1)
                    action = "boost_workers_post"
                flow_n = flow_cap if snap.ram_used_frac_of_budget < TARGET_HI else 0

            under = (
                snap.ram_used_frac_of_budget < PRESSURE_TARGET
                and snap.cpu_frac < PRESSURE_TARGET
            )
            if under and pending_cpu_work and not pending_gpu_work:
                self._underutil_streak += 1
            else:
                self._underutil_streak = 0

            # Pressure toward high utilization, but never past the CPU soft-cap.
            if (
                self._underutil_streak >= 2
                and snap.ram_used_frac_of_budget < CRITICAL_FRAC
                and snap.cpu_frac < CPU_SOFT_CAP
                and not pending_gpu_work
            ):
                if up_auto and up < cap:
                    up = min(cap, up + 1)
                    action = "boost_workers"
                if depth < max_depth and prep_queue <= max(0, depth - 1):
                    depth = min(max_depth, depth + 1)
                    action = "boost_prep" if action == "hold" else f"{action}+prep"
            elif (
                snap.ram_used_frac_of_budget < TARGET_LO
                and pending_cpu_work
                and not pending_gpu_work
            ):
                if snap.cpu_frac < TARGET_LO:
                    if up_auto and up < cap:
                        up = min(cap, up + 1)
                        action = "boost_workers"
                    if depth < max_depth and prep_queue == 0:
                        depth += 1
                        action = "boost_prep" if action == "hold" else f"{action}+prep"

            if pending_cpu_work and snap.ram_used_frac_of_budget < CRITICAL_FRAC:
                if snap.cpu_frac > TARGET_HI:
                    if up_auto and up > 1:
                        up = max(1, up - 1)
                        if action == "hold":
                            action = "trim_cpu"
                    flow_n = 0
                elif (
                    snap.cpu_frac < TARGET_LO
                    and up_auto
                    and up < cap
                    and snap.cpu_frac < CPU_SOFT_CAP
                    and not pending_gpu_work
                ):
                    up = min(cap, up + 1)
                    if action == "hold":
                        action = "boost_workers"

        up = max(1, min(up, cap))
        flow_n = max(0, min(int(flow_n), flow_cap))
        self._plan = ConcurrencyPlan(
            upsample_workers=up,
            prep_queue_depth=depth,
            flow_prefetch_workers=flow_n,
            pause_prep=pause,
            last_action=action,
        )
        return self._plan

    def format_status(self, snap: ResourceSnapshot | None = None) -> str:
        s = snap or self.snapshot()
        return (
            f"[resources] RAM {s.ram_used_frac_of_budget * 100:.0f}% budget | "
            f"CPU {s.cpu_frac * 100:.0f}% | "
            f"MPS {s.mps_frac_of_budget * 100:.0f}% | "
            f"up_workers={s.upsample_workers} "
            f"flow_prefetch={self._plan.flow_prefetch_workers} "
            f"queues={s.prep_queue}/{s.infer_queue} (depth={s.queue_depth}) "
            f"action={s.last_action or self._plan.last_action}"
            f"{' PAUSE_PREP' if s.pause_prep else ''}"
        )

    def emergency_shed(self) -> ConcurrencyPlan:
        self._plan = replace(
            self._plan,
            upsample_workers=1,
            prep_queue_depth=1,
            flow_prefetch_workers=0,
            pause_prep=True,
            last_action="emergency_shed",
        )
        self._underutil_streak = 0
        return self._plan
