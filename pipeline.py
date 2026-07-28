"""Prep | infer | post assembly-line for DVD windowed depth."""

from __future__ import annotations

import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

from da3_stabilize import (
    FlowPrefetchPool,
    IncrementalShotStabilizer,
    SharedPairCache,
    StabilizeParams,
    lock_window_to_ref,
)

_REPO = Path(__file__).resolve().parent
_DVD = _REPO / "vendor" / "DVD"
if str(_DVD) not in sys.path:
    sys.path.insert(0, str(_DVD))
if str(_DVD / "test_script") not in sys.path:
    sys.path.insert(0, str(_DVD / "test_script"))

from resources import ResourceGovernor
from test_single_video import compute_scale_and_shift, get_window_index, pad_time_mod4
from upsample import ParallelUpscaler, UpsampleParams, default_upsample_params


class _Sentinel:
    pass


_STOP = _Sentinel()


@dataclass
class PreparedWindow:
    index: int
    start: int
    end: int
    rgb: torch.Tensor  # [1, T_pad, C, H, W]
    origin_t: int


@dataclass
class InferredWindow:
    index: int
    start: int
    end: int
    depth: np.ndarray  # [1, origin_t, H, W, C]
    elapsed_s: float = 0.0


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60.0)
    if m < 60:
        return f"{int(m)}m{s:04.1f}s"
    h, m = divmod(int(m), 60)
    return f"{h}h{m:02d}m{s:04.1f}s"


def _log_window_timing(
    *,
    index: int,
    n_windows: int,
    start: int,
    end: int,
    elapsed_s: float,
    totals: list[float],
) -> None:
    """Print per-window wall time plus running average and ETA."""
    totals.append(float(elapsed_s))
    done = len(totals)
    avg = sum(totals) / done
    remaining = max(0, int(n_windows) - done)
    eta = avg * remaining
    print(
        f"  [timing] window {index}/{max(0, n_windows - 1)} "
        f"frames [{start}:{end}] "
        f"took {_format_duration(elapsed_s)} "
        f"(avg {_format_duration(avg)} over {done}/{n_windows}; "
        f"ETA {_format_duration(eta)})",
        flush=True,
    )


def _depth_frames_to_disparities(depth_bthwc: np.ndarray) -> list[np.ndarray]:
    if depth_bthwc.ndim != 5 or depth_bthwc.shape[0] != 1:
        raise ValueError(f"Expected depth batch [1,T,H,W,C], got {depth_bthwc.shape}")
    out: list[np.ndarray] = []
    for frame in depth_bthwc[0]:
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.ndim == 3:
            arr = arr.mean(axis=-1)
        out.append(arr.astype(np.float32, copy=False))
    return out


def _disparities_to_depth_batch(disparities: list[np.ndarray]) -> np.ndarray:
    if not disparities:
        raise ValueError("No disparities to convert")
    batch = np.stack(
        [np.asarray(d, dtype=np.float32) for d in disparities],
        axis=0,
    )
    return batch[None, ..., None]


def _align_window(
    depth_aligned: np.ndarray | None,
    depth: np.ndarray,
    prev_end: int | None,
    start: int,
    *,
    scale_only: bool,
) -> tuple[np.ndarray, int]:
    end = start + depth.shape[1]
    if depth_aligned is None:
        return depth, end

    real_overlap = (prev_end or start) - start
    if real_overlap > 0:
        ref_frames = depth_aligned[:, -real_overlap:]
        curr_frames = depth[:, :real_overlap]
        if scale_only:
            scale = np.sum(curr_frames * ref_frames) / (
                np.sum(curr_frames * curr_frames) + 1e-6
            )
            shift = 0.0
        else:
            scale, shift = compute_scale_and_shift(curr_frames, ref_frames)
        scale = float(np.clip(scale, 0.7, 1.5))
        aligned_t = depth * scale + shift
        aligned_t[aligned_t < 0] = 0
        alpha = np.linspace(0, 1, real_overlap, dtype=np.float32).reshape(
            1, real_overlap, 1, 1, 1
        )
        smooth = (1 - alpha) * ref_frames + alpha * aligned_t[:, :real_overlap]
        depth_aligned = np.concatenate(
            [
                depth_aligned[:, :-real_overlap],
                smooth,
                aligned_t[:, real_overlap:],
            ],
            axis=1,
        )
    else:
        depth_aligned = np.concatenate([depth_aligned, depth], axis=1)
    return depth_aligned, end


def _finalize_upto(
    upscaler: ParallelUpscaler | None,
    depth_aligned: np.ndarray,
    *,
    window_index: int,
    n_windows: int,
    depth_windows: list[tuple[int, int]],
    total_frames: int,
    finalized_upto: int,
) -> int:
    if upscaler is None:
        return finalized_upto
    if window_index + 1 < n_windows:
        next_start, _ = depth_windows[window_index + 1]
        can_finalize_to = min(next_start, depth_aligned.shape[1], total_frames)
    else:
        can_finalize_to = min(depth_aligned.shape[1], total_frames)
    if can_finalize_to > finalized_upto:
        upscaler.submit_range(
            depth_aligned[:, finalized_upto:can_finalize_to],
            finalized_upto,
            can_finalize_to,
        )
        return can_finalize_to
    return finalized_upto


def run_window_pipeline(
    model,
    *,
    video_path: str | Path,
    out_h: int,
    out_w: int,
    dtype: torch.dtype,
    window_size: int,
    overlap: int,
    total_frames: int,
    orig_h: int,
    orig_w: int,
    device: torch.device | None,
    free_memory: Callable[..., None],
    streaming_reader_cls,
    governor: ResourceGovernor,
    upsample: bool = True,
    upsample_params: UpsampleParams | None = None,
    scale_only: bool = False,
    pipeline_parallel: bool = True,
    shot_ranges: tuple[tuple[int, int], ...] | None = None,
    stabilize_params: StabilizeParams | None = None,
    cache_dir: Path | None = None,
    keep_upsample_cache: bool = False,
    bgr_store=None,
) -> tuple[np.ndarray | None, ParallelUpscaler | None]:
    """
    Run windowed DVD inference.

    When ``pipeline_parallel``: prep ‖ infer ‖ post(align+upsample).
    Returns ``(depth_aligned_or_None, upscaler_or_None)``.
    """
    depth_windows = get_window_index(total_frames, window_size, overlap)
    n_windows = len(depth_windows)
    print(
        f"{n_windows} windows. "
        f"Pipeline={'prep|infer|post' if pipeline_parallel and n_windows > 1 else 'sequential'}.",
        flush=True,
    )
    print(governor.format_status(), flush=True)

    upscaler: ParallelUpscaler | None = None
    if upsample:
        params = upsample_params or default_upsample_params(
            out_h, out_w, orig_h, orig_w
        )
        plan0 = governor.plan()
        upscaler = ParallelUpscaler(
            video_path,
            out_size=(orig_w, orig_h),
            params=params,
            total_frames=total_frames,
            workers=plan0.upsample_workers,
            cache_dir=cache_dir,
            keep_cache=keep_upsample_cache,
        )

    if not pipeline_parallel or n_windows == 1:
        return _run_sequential(
            model,
            video_path=video_path,
            out_h=out_h,
            out_w=out_w,
            dtype=dtype,
            depth_windows=depth_windows,
            total_frames=total_frames,
            device=device,
            free_memory=free_memory,
            streaming_reader_cls=streaming_reader_cls,
            governor=governor,
            upscaler=upscaler,
            scale_only=scale_only,
            bgr_store=bgr_store,
        )

    return _run_overlapped(
        model,
        video_path=video_path,
        out_h=out_h,
        out_w=out_w,
        dtype=dtype,
        depth_windows=depth_windows,
        total_frames=total_frames,
        device=device,
        free_memory=free_memory,
        streaming_reader_cls=streaming_reader_cls,
        governor=governor,
        upscaler=upscaler,
        scale_only=scale_only,
        shot_ranges=shot_ranges,
        stabilize_params=stabilize_params,
        bgr_store=bgr_store,
    )


def _flush_dit_resident(model) -> None:
    dit = getattr(getattr(model, "pipe", None), "dit", None)
    flush = getattr(dit, "_dvd_flush_resident", None)
    if callable(flush):
        flush()


def _infer_one(model, prepared: PreparedWindow, out_h: int, out_w: int) -> InferredWindow:
    rgb = prepared.rgb
    num_frames = rgb.shape[1]
    # First window (VAE + DiT block-swap) can take quiet minutes on 8GB MPS;
    # heartbeat so a full prep queue does not look like a hang.
    stop_hb = threading.Event()
    t0 = time.perf_counter()

    def _heartbeat() -> None:
        while not stop_hb.wait(30.0):
            print(
                f"  … still running DVD pipeline for window {prepared.index} "
                f"({time.perf_counter() - t0:.0f}s elapsed)",
                flush=True,
            )

    hb = threading.Thread(target=_heartbeat, name="dvd-infer-hb", daemon=True)
    hb.start()
    try:
        outputs = model.pipe(
            prompt=[""],
            negative_prompt=[""],
            mode=model.args.mode,
            height=out_h,
            width=out_w,
            num_frames=num_frames,
            batch_size=1,
            input_image=rgb[:, 0],
            extra_images=rgb,
            extra_image_frame_index=torch.ones([1, num_frames]),
            input_video=rgb,
            cfg_scale=1,
            seed=0,
            tiled=False,
            denoise_step=model.args.denoise_step,
        )
        depth = np.asarray(outputs["depth"][:, : prepared.origin_t])
        if depth.ndim == 5 and depth.shape[0] == 1:
            depth_frames = [
                np.asarray(depth[0, i], dtype=np.float32)[..., 0]
                if depth[0, i].ndim == 3 and depth[0, i].shape[-1] == 1
                else np.asarray(depth[0, i], dtype=np.float32).mean(axis=-1)
                if depth[0, i].ndim == 3
                else np.asarray(depth[0, i], dtype=np.float32)
                for i in range(depth.shape[1])
            ]
            locked = lock_window_to_ref(depth_frames, strategy="middle")
            safe_locked: list[np.ndarray] = []
            for src, dst in zip(depth_frames, locked):
                pre_std = float(np.nanstd(src))
                post_std = float(np.nanstd(dst))
                ratio = post_std / max(pre_std, 1e-6)
                if ratio < 0.85 or ratio > 1.25:
                    safe_locked.append(src)
                else:
                    safe_locked.append(dst)
            for i, disp in enumerate(safe_locked):
                if depth[0, i].ndim == 3:
                    depth[0, i, ..., :] = disp[..., None]
                else:
                    depth[0, i] = disp
        del outputs
    finally:
        stop_hb.set()
        hb.join(timeout=1.0)
    _flush_dit_resident(model)
    elapsed_s = time.perf_counter() - t0
    print(
        f"  DVD pipeline window {prepared.index} "
        f"frames [{prepared.start}:{prepared.end}] "
        f"finished in {_format_duration(elapsed_s)}",
        flush=True,
    )
    return InferredWindow(
        index=prepared.index,
        start=prepared.start,
        end=prepared.end,
        depth=depth,
        elapsed_s=elapsed_s,
    )


def _run_sequential(
    model,
    *,
    video_path,
    out_h,
    out_w,
    dtype,
    depth_windows,
    total_frames,
    device,
    free_memory,
    streaming_reader_cls,
    governor,
    upscaler,
    scale_only,
    bgr_store=None,
) -> tuple[np.ndarray | None, ParallelUpscaler | None]:
    reader = streaming_reader_cls(video_path, out_h, out_w, dtype, bgr_store=bgr_store)
    depth_aligned = None
    prev_end = None
    finalized_upto = 0
    n_windows = len(depth_windows)
    window_timings: list[float] = []
    try:
        for i, (start, end) in enumerate(tqdm(depth_windows, desc="Inferencing Slices")):
            plan = governor.adjust(
                pending_cpu_work=upscaler is not None,
                pending_gpu_work=True,
                force=True,
            )
            if upscaler is not None:
                upscaler.set_workers(plan.upsample_workers)
            print(governor.format_status(), flush=True)
            print(f"Window {i}/{n_windows - 1}: frames [{start}:{end}]", flush=True)
            rgb = reader.read_window(start, end)
            rgb, origin_t = pad_time_mod4(rgb)
            prepared = PreparedWindow(i, start, end, rgb, origin_t)
            print("  running DVD pipeline...", flush=True)
            try:
                inferred = _infer_one(model, prepared, out_h, out_w)
            except Exception:
                governor.emergency_shed()
                _flush_dit_resident(model)
                free_memory(device)
                raise
            del prepared, rgb
            _flush_dit_resident(model)
            free_memory(device)
            _log_window_timing(
                index=inferred.index,
                n_windows=n_windows,
                start=inferred.start,
                end=inferred.end,
                elapsed_s=inferred.elapsed_s,
                totals=window_timings,
            )
            print(f"  pipeline done. depth shape={inferred.depth.shape}", flush=True)
            # Post/align is CPU-bound — pressure upsample workers.
            plan = governor.adjust(
                pending_cpu_work=True,
                pending_gpu_work=False,
                force=True,
            )
            if upscaler is not None:
                upscaler.set_workers(plan.upsample_workers)
            print(governor.format_status(), flush=True)
            depth_aligned, prev_end = _align_window(
                depth_aligned,
                inferred.depth,
                prev_end,
                inferred.start,
                scale_only=scale_only,
            )
            del inferred
            finalized_upto = _finalize_upto(
                upscaler,
                depth_aligned,
                window_index=i,
                n_windows=n_windows,
                depth_windows=depth_windows,
                total_frames=total_frames,
                finalized_upto=finalized_upto,
            )
            free_memory()
        if window_timings:
            total_infer = sum(window_timings)
            print(
                f"[timing] all {len(window_timings)} infer windows: "
                f"total {_format_duration(total_infer)} "
                f"(avg {_format_duration(total_infer / len(window_timings))})",
                flush=True,
            )
    finally:
        reader.close()

    return _finish_align(
        depth_aligned,
        upscaler,
        total_frames=total_frames,
        finalized_upto=finalized_upto,
        free_memory=free_memory,
    )


def _run_overlapped(
    model,
    *,
    video_path,
    out_h,
    out_w,
    dtype,
    depth_windows,
    total_frames,
    device,
    free_memory,
    streaming_reader_cls,
    governor,
    upscaler,
    scale_only,
    shot_ranges,
    stabilize_params,
    bgr_store=None,
) -> tuple[np.ndarray | None, ParallelUpscaler | None]:
    """
    Three-station line:

    Thread A (dvd-prep):  decode+pad windows → prep_q
    Main:                 prep_q → DVD infer → infer_q
    Thread B (dvd-post):  infer_q → align → queue JBU upsample
    """
    n_windows = len(depth_windows)
    plan0 = governor.plan()
    q_depth = max(1, int(plan0.prep_queue_depth))
    prep_q: queue.Queue = queue.Queue(maxsize=q_depth)
    infer_q: queue.Queue = queue.Queue(maxsize=max(2, q_depth))
    errors: list[BaseException] = []
    post_done = threading.Event()
    result_box: dict = {
        "depth_aligned": None,
        "depth_stabilized": None,
        "upscaler": upscaler,
        "finalized_upto": 0,
    }
    if stabilize_params is None:
        raise ValueError("stabilize_params is required for overlapped pipeline")
    if not shot_ranges:
        shot_ranges = ((0, total_frames),)

    pair_cache = SharedPairCache()
    flow_prefetch: FlowPrefetchPool | None = None
    if bgr_store is not None:
        low_mem = False
        try:
            from resources import total_ram_bytes

            low_mem = total_ram_bytes() <= int(8.5 * 1024**3)
        except Exception:
            pass
        prefetch_workers = 1 if low_mem else 2
        flow_prefetch = FlowPrefetchPool(
            bgr_store,
            process_w=out_w,
            process_h=out_h,
            params=stabilize_params,
            pair_cache=pair_cache,
            workers=prefetch_workers,
        )

    def _prep_loop() -> None:
        reader = streaming_reader_cls(
            video_path, out_h, out_w, dtype, bgr_store=bgr_store
        )
        try:
            for i, (start, end) in enumerate(depth_windows):
                while True:
                    if errors:
                        return
                    plan = governor.adjust(
                        prep_queue=prep_q.qsize(),
                        infer_queue=infer_q.qsize(),
                        pending_cpu_work=True,
                        pending_gpu_work=True,
                    )
                    if not plan.pause_prep:
                        # Resize queue depth soft-bound via blocking put timeout.
                        break
                    time.sleep(0.15)
                print(
                    f"Prep window {i}/{n_windows - 1}: frames [{start}:{end}]",
                    flush=True,
                )
                rgb = reader.read_window(start, end)
                rgb, origin_t = pad_time_mod4(rgb)
                prepared = PreparedWindow(i, start, end, rgb, origin_t)
                while True:
                    if errors:
                        return
                    try:
                        prep_q.put(prepared, timeout=0.4)
                        break
                    except queue.Full:
                        governor.adjust(
                            prep_queue=prep_q.qsize(),
                            infer_queue=infer_q.qsize(),
                            pending_cpu_work=True,
                            pending_gpu_work=True,
                        )
                        continue
            while True:
                try:
                    prep_q.put(_STOP, timeout=0.4)
                    break
                except queue.Full:
                    if errors:
                        return
                    continue
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            traceback.print_exc()
            try:
                prep_q.put_nowait(_STOP)
            except Exception:
                pass
        finally:
            reader.close()

    def _post_loop() -> None:
        depth_aligned = None
        prev_end = None
        finalized_upto = 0
        stabilized_upto = 0
        shot_idx = 0
        shot_stabilizer: IncrementalShotStabilizer | None = None
        shot_emit_cursor = int(shot_ranges[0][0])
        stabilized_depth = None

        def _ensure_shot() -> None:
            nonlocal shot_stabilizer, shot_emit_cursor
            if shot_stabilizer is not None or shot_idx >= len(shot_ranges):
                return
            s0, s1 = shot_ranges[shot_idx]
            shot_stabilizer = IncrementalShotStabilizer(
                video_path=video_path,
                process_w=out_w,
                process_h=out_h,
                params=stabilize_params,
                global_start=s0,
                bgr_store=bgr_store,
                pair_cache=pair_cache,
            )
            shot_emit_cursor = s0

        def _commit_emitted(disparities: list[np.ndarray]) -> None:
            nonlocal shot_emit_cursor, stabilized_depth
            if not disparities:
                return
            depth_chunk = _disparities_to_depth_batch(disparities)
            emit_start = shot_emit_cursor
            emit_end = emit_start + depth_chunk.shape[1]
            shot_emit_cursor = emit_end
            if upscaler is not None:
                upscaler.submit_range(depth_chunk, emit_start, emit_end)
                return
            if stabilized_depth is None:
                stabilized_depth = np.empty(
                    (1, total_frames, depth_chunk.shape[2], depth_chunk.shape[3], 1),
                    dtype=np.float32,
                )
            stabilized_depth[:, emit_start:emit_end] = depth_chunk

        def _stabilize_ready(final_upto: int, *, flush: bool = False) -> None:
            nonlocal stabilized_upto, shot_idx, shot_stabilizer, shot_emit_cursor
            while stabilized_upto < final_upto and shot_idx < len(shot_ranges):
                _ensure_shot()
                assert shot_stabilizer is not None
                shot_start, shot_end = shot_ranges[shot_idx]
                chunk_end = min(final_upto, shot_end)
                if chunk_end <= stabilized_upto:
                    break
                disps = _depth_frames_to_disparities(
                    depth_aligned[:, stabilized_upto:chunk_end]
                )
                emitted = shot_stabilizer.append(
                    disps,
                    finalize=flush or chunk_end >= shot_end,
                )
                _commit_emitted(emitted)
                stabilized_upto = chunk_end
                if flush or chunk_end >= shot_end:
                    shot_idx += 1
                    shot_stabilizer = None
                    if shot_idx < len(shot_ranges):
                        shot_emit_cursor = int(shot_ranges[shot_idx][0])

        try:
            while True:
                item = infer_q.get()
                if isinstance(item, _Sentinel):
                    if depth_aligned is not None:
                        finalized_upto = min(depth_aligned.shape[1], total_frames)
                        _stabilize_ready(finalized_upto, flush=True)
                    break
                assert isinstance(item, InferredWindow)
                plan = governor.adjust(
                    prep_queue=prep_q.qsize(),
                    infer_queue=infer_q.qsize(),
                    pending_cpu_work=True,
                    pending_gpu_work=False,
                    force=True,
                )
                if upscaler is not None:
                    upscaler.set_workers(plan.upsample_workers)
                if flow_prefetch is not None:
                    flow_prefetch.set_workers(plan.flow_prefetch_workers)
                print(governor.format_status(), flush=True)

                depth_aligned, prev_end = _align_window(
                    depth_aligned,
                    item.depth,
                    prev_end,
                    item.start,
                    scale_only=scale_only,
                )
                finalized_upto = _finalize_upto(
                    None,
                    depth_aligned,
                    window_index=item.index,
                    n_windows=n_windows,
                    depth_windows=depth_windows,
                    total_frames=total_frames,
                    finalized_upto=finalized_upto,
                )
                _stabilize_ready(finalized_upto, flush=False)
                del item
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            traceback.print_exc()
        finally:
            result_box["depth_aligned"] = depth_aligned
            result_box["depth_stabilized"] = stabilized_depth
            result_box["finalized_upto"] = finalized_upto
            post_done.set()

    prep_thread = threading.Thread(target=_prep_loop, name="dvd-prep", daemon=True)
    post_thread = threading.Thread(target=_post_loop, name="dvd-post", daemon=True)
    prep_thread.start()
    post_thread.start()

    try:
        pbar = tqdm(total=n_windows, desc="Inferencing Slices")
        window_timings: list[float] = []
        while True:
            if errors:
                break
            try:
                item = prep_q.get(timeout=0.4)
            except queue.Empty:
                if not prep_thread.is_alive() and prep_q.empty():
                    break
                continue
            if isinstance(item, _Sentinel):
                break
            assert isinstance(item, PreparedWindow)
            plan = governor.adjust(
                prep_queue=prep_q.qsize(),
                infer_queue=infer_q.qsize(),
                pending_cpu_work=True,
                pending_gpu_work=True,
                force=True,
            )
            if upscaler is not None:
                upscaler.set_workers(plan.upsample_workers)
            if flow_prefetch is not None:
                flow_prefetch.set_workers(plan.flow_prefetch_workers)
            print(governor.format_status(), flush=True)
            print(
                f"Infer window {item.index}/{n_windows - 1}: "
                f"frames [{item.start}:{item.end}]",
                flush=True,
            )
            print("  running DVD pipeline...", flush=True)
            try:
                inferred = _infer_one(model, item, out_h, out_w)
            except Exception:
                governor.emergency_shed()
                _flush_dit_resident(model)
                free_memory(device)
                raise
            del item
            _flush_dit_resident(model)
            free_memory(device)
            _log_window_timing(
                index=inferred.index,
                n_windows=n_windows,
                start=inferred.start,
                end=inferred.end,
                elapsed_s=inferred.elapsed_s,
                totals=window_timings,
            )
            print(f"  pipeline done. depth shape={inferred.depth.shape}", flush=True)
            while True:
                if errors:
                    break
                try:
                    infer_q.put(inferred, timeout=0.4)
                    break
                except queue.Full:
                    continue
            pbar.update(1)
        pbar.close()
        if window_timings:
            total_infer = sum(window_timings)
            print(
                f"[timing] all {len(window_timings)} infer windows: "
                f"total {_format_duration(total_infer)} "
                f"(avg {_format_duration(total_infer / len(window_timings))})",
                flush=True,
            )
    except BaseException as exc:  # noqa: BLE001
        errors.append(exc)
        traceback.print_exc()
    finally:
        try:
            infer_q.put(_STOP, timeout=5.0)
        except Exception:
            try:
                infer_q.put_nowait(_STOP)
            except Exception:
                pass
        # Wait for prep/post to fully drain. A short timeout here races
        # upscaler.finish() against late stabilize→submit of the tail frames
        # (Missing upsampled frames at the end of long runs).
        if prep_thread.is_alive():
            print("  waiting for prep thread to finish...", flush=True)
        prep_thread.join()
        if not post_done.is_set():
            print(
                "  waiting for post (align/stabilize/upsample-submit) to finish...",
                flush=True,
            )
        post_done.wait()
        post_thread.join()
        if flow_prefetch is not None:
            flow_prefetch.stop()
            flow_prefetch = None

    if errors:
        raise errors[0]

    if upscaler is not None:
        # Ensure every frame was queued before the caller drains workers.
        pending = upscaler.pending()
        if pending:
            print(
                f"  waiting for {pending} queued upsample job(s) before return...",
                flush=True,
            )
        free_memory()
        return None, upscaler
    depth_stabilized = result_box["depth_stabilized"]
    if depth_stabilized is None:
        raise RuntimeError("No stabilized depth produced")
    return depth_stabilized[:, :total_frames], None


def _finish_align(
    depth_aligned: np.ndarray | None,
    upscaler: ParallelUpscaler | None,
    *,
    total_frames: int,
    finalized_upto: int,
    free_memory: Callable[..., None],
) -> tuple[np.ndarray | None, ParallelUpscaler | None]:
    if upscaler is not None:
        if depth_aligned is not None and finalized_upto < total_frames:
            end_t = min(depth_aligned.shape[1], total_frames)
            if end_t > finalized_upto:
                upscaler.submit_range(
                    depth_aligned[:, finalized_upto:end_t],
                    finalized_upto,
                    end_t,
                )
        del depth_aligned
        free_memory()
        return None, upscaler

    if depth_aligned is None:
        raise RuntimeError("No depth windows produced")
    return depth_aligned[:, :total_frames], None
