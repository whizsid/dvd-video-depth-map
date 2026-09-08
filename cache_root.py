"""Resolve an external FAT32 (or explicit) root for runtime caches.

Upsample memmaps can be large; keeping them off the project/APFS volume frees
internal disk. Model weights (HF hub, ckpt/) stay in the project. FAT32 caps a
single file at <4 GiB — callers should use ``fat32_max_file_bytes`` when
allocating contiguous caches. The CUDA DepthCrafter path prefers local
``.cache/`` via ``DEPTHCRAFTER_CACHE_DIR`` / ``DVD_CACHE_DIR``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

# FAT32 max file size is 2^32 - 1; leave a small margin for FS metadata.
FAT32_MAX_FILE_BYTES = (4 * 1024**3) - 65536
CACHE_DIRNAME = "dvd_cache"
_MIN_FREE_GB = 5.0
_FAT_FSTYPES = frozenset({"msdos", "fat", "fat32", "ms-dos"})


def _mount_table() -> list[tuple[Path, str]]:
    """Return [(mount_point, fstype), ...] from ``mount`` (macOS/Linux)."""
    try:
        out = subprocess.check_output(["mount"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return []
    rows: list[tuple[Path, str]] = []
    # /dev/disk5s4 on /Volumes/EXTERNAL HA (msdos, local, ...)
    pat = re.compile(r"^.*? on (.+) \(([^,\)]+)")
    for line in out.splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        rows.append((Path(m.group(1)), m.group(2).strip().lower()))
    return rows


def filesystem_type(path: Path) -> str | None:
    """Best-effort fstype for ``path`` (longest matching mount prefix)."""
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    best: str | None = None
    best_len = -1
    for mp, fstype in _mount_table():
        try:
            if resolved == mp or resolved.is_relative_to(mp):
                n = len(str(mp))
                if n > best_len:
                    best = fstype
                    best_len = n
        except (OSError, ValueError):
            continue
    return best


def is_fat32(path: Path) -> bool:
    fstype = filesystem_type(path)
    return fstype is not None and fstype in _FAT_FSTYPES


def fat32_max_file_bytes(path: Path) -> int | None:
    """Return the max safe single-file size on this path, or None if unlimited."""
    return FAT32_MAX_FILE_BYTES if is_fat32(path) else None


def _free_bytes(path: Path) -> int:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0


def list_fat32_volumes(volumes_root: Path | None = None) -> list[Path]:
    """Mounted FAT/FAT32 volumes under ``/Volumes`` (macOS)."""
    root = volumes_root or Path("/Volumes")
    if not root.is_dir():
        return []
    found: list[Path] = []
    mount_fat = {mp for mp, ft in _mount_table() if ft in _FAT_FSTYPES}
    try:
        candidates = sorted(root.iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return []
    for p in candidates:
        if not p.is_dir() or p.name.startswith("."):
            continue
        try:
            resolved = p.resolve()
        except OSError:
            continue
        if resolved in mount_fat or filesystem_type(p) in _FAT_FSTYPES:
            found.append(p)
    return found


def pick_fat32_volume(*, min_free_gb: float = _MIN_FREE_GB) -> Path | None:
    """Choose the FAT32 volume with the most free space (≥ ``min_free_gb``)."""
    best: Path | None = None
    best_free = -1
    min_free = int(min_free_gb * 1024**3)
    for vol in list_fat32_volumes():
        free = _free_bytes(vol)
        if free < min_free:
            continue
        if free > best_free:
            best = vol
            best_free = free
    return best


def resolve_cache_root(
    explicit: str | Path | None = None,
    *,
    env_var: str = "DEPTHCRAFTER_CACHE_DIR",
    min_free_gb: float = _MIN_FREE_GB,
    create: bool = True,
) -> Path:
    """Resolve runtime cache root.

    Order:
      1. ``explicit`` (CLI ``--cache-dir``)
      2. ``$DEPTHCRAFTER_CACHE_DIR``, then ``$DVD_CACHE_DIR`` (legacy alias)
      3. Auto-pick a mounted FAT32 volume → ``<vol>/dvd_cache``
    """
    if explicit is not None:
        root = Path(explicit).expanduser()
    else:
        env = (
            os.environ.get(env_var, "").strip()
            or os.environ.get("DVD_CACHE_DIR", "").strip()
            or os.environ.get("DEPTHCRAFTER_CACHE_DIR", "").strip()
        )
        if env:
            root = Path(env).expanduser()
        else:
            vol = pick_fat32_volume(min_free_gb=min_free_gb)
            if vol is None:
                vols = list_fat32_volumes()
                hint = (
                    ", ".join(str(v) for v in vols)
                    if vols
                    else "none mounted under /Volumes"
                )
                raise RuntimeError(
                    "No suitable external FAT32 volume found for runtime caches. "
                    f"Mounted FAT volumes: {hint}. "
                    f"Need ≥{min_free_gb:.0f} GiB free, or pass --cache-dir / "
                    f"set {env_var}."
                )
            root = vol / CACHE_DIRNAME

    root = root.resolve() if root.exists() else root
    if create:
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot create cache root at {root}: {exc}. "
                "Check that the external volume is mounted and writable."
            ) from exc
        # Probe writability early (FAT32 mounts can be read-only).
        probe = root / ".dvd_cache_write_probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"Cache root {root} is not writable: {exc}"
            ) from exc

    free_gb = _free_bytes(root) / (1024**3)
    fat = is_fat32(root)
    print(
        f"Runtime cache root: {root} "
        f"(free ≈ {free_gb:.1f} GiB"
        f"{', FAT32' if fat else ''})",
        flush=True,
    )
    return root


def dit_cache_dir(cache_root: Path, dtype_name: str) -> Path:
    # Avoid leading-dot names on FAT32 (some tools hide/skip them).
    return cache_root / f"dit_cache_{dtype_name}"


def resolve_dit_cache_dir(
    cache_root: Path,
    dtype_name: str,
    *,
    need_gb: float = 3.5,
    project_root: Path | None = None,
) -> Path:
    """Prefer a fast local (non-FAT32) dir for the DiT mmap blob.

    Order: existing local cache → project ``.cache/dit_cache_*`` with free space →
    ``$TMPDIR`` → ``cache_root`` (may be FAT32 USB). Upsample caches still use
    ``cache_root``.
    """
    name = f"dit_cache_{dtype_name}"
    need = int(need_gb * 1024**3)
    candidates: list[Path] = []
    root = project_root or Path(__file__).resolve().parent
    candidates.append(root / ".cache" / name)
    tmp = os.environ.get("TMPDIR") or os.environ.get("TMP") or "/tmp"
    candidates.append(Path(tmp) / "dvd" / name)
    candidates.append(cache_root / name)

    # Prefer an already-built mmap cache on a non-FAT volume.
    for c in candidates:
        idx = c / "index.json"
        blob = c / "weights.f16.bin"
        if idx.exists() and blob.exists() and not is_fat32(c):
            print(f"DiT cache (reuse local): {c}", flush=True)
            return c

    for c in candidates:
        parent = c.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if is_fat32(c) or is_fat32(parent):
            continue
        if _free_bytes(parent) < need:
            continue
        try:
            c.mkdir(parents=True, exist_ok=True)
            probe = c / ".dvd_dit_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
        except OSError:
            continue
        print(f"DiT cache (local APFS/SSD): {c}", flush=True)
        return c

    fallback = cache_root / name
    fallback.mkdir(parents=True, exist_ok=True)
    print(
        f"DiT cache (fallback on cache root): {fallback}"
        f"{' [FAT32 — slower reload]' if is_fat32(fallback) else ''}",
        flush=True,
    )
    return fallback


def up_cache_dir(cache_root: Path, pid: int | None = None) -> Path:
    return cache_root / f"up_cache_{pid if pid is not None else os.getpid()}"


def up_cache_dir_for_video(
    cache_root: Path,
    video_path: str | Path,
    *,
    total_frames: int,
    out_w: int,
    out_h: int,
) -> Path:
    """Stable upsample cache path for a video + output geometry (reusable across runs)."""
    import hashlib

    resolved = str(Path(video_path).resolve())
    stem = re.sub(r"[^\w.\-]+", "_", Path(video_path).stem).strip("_")[:40] or "video"
    digest = hashlib.sha1(
        f"{resolved}|{int(total_frames)}|{int(out_w)}x{int(out_h)}".encode()
    ).hexdigest()[:10]
    return cache_root / f"up_cache_v_{stem}_{digest}"
