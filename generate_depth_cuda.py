#!/usr/bin/env python3
"""Backward-compatible entry point for CUDA DepthCrafter video depth.

``generate_depth.py`` is now CUDA-only DepthCrafter. This module re-exports
its ``main`` so older Colab / docs commands keep working.
"""

from __future__ import annotations

import time

from generate_depth import _format_elapsed, main

if __name__ == "__main__":
    t0 = time.perf_counter()
    try:
        main()
    finally:
        elapsed = time.perf_counter() - t0
        print(f"Total time: {_format_elapsed(elapsed)} ({elapsed:.2f}s)", flush=True)
