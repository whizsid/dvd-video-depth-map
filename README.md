# DVD Video Depth (MPS + CUDA T4)

Generate **grayscale depth-map videos** with [EnVision-Research/DVD](https://github.com/EnVision-Research/DVD) on Apple Silicon (**MPS**) or **CUDA** (full-GPU, Colab T4).

## Setup

```bash
# 1. Create & activate the venv (already created if you followed the project bootstrap)
python3.10 -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch + project deps
pip install -r requirements.txt

# 3. Install the vendored DVD package (editable)
pip install -e vendor/DVD --no-deps

# 4. Download DVD weights (~GB-scale) — only when you are ready
HF_TOKEN=… python scripts/download_weights.py
```

Weights land in `ckpt/` (`dvd_1.1.safetensors`, `model_config.yaml`). On first inference, DVD also downloads Wan2.1 backbone files into `./models/`.

## Generate a grayscale depth video

```bash
source .venv/bin/activate
# Requires ckpt/dvd_1.1.safetensors already downloaded
HF_TOKEN=… python generate_depth.py \
  --input-video vendor/DVD/demo/robot_navi.mp4 \
  --device mps \
  --dtype float16
```

Output: `outputs/<name>_depth_gray.mp4`

### Useful flags

| Flag | Default | Notes |
|------|---------|--------|
| `--device` | `mps` | Also: `auto`, `cpu`, `cuda` |
| `--dtype` | `float16` | Prefer `float16` on MPS (DiT + VAE) |
| `--cpu` | off | Full CPU mode (most reliable on 8GB if MPS is still killed) |
| `--cache-dir` | auto | Runtime caches (upsample memmap; DiT mmap prefers local APFS). Default: `$DVD_CACHE_DIR`, else a mounted FAT32 volume under `/Volumes/…/dvd_cache`. HF models stay in the project |
| `--height` / `--width` | auto | On 8GB: `256×448` (fallback: `192×320`) |
| `--window-size` / `--overlap` | auto | On 8GB: `9` / `3` (fallback: `5` / `2`) |
| `--no-upsample` | off | Skip RGB-guided JBU (keep infer res) |
| `--upsample-workers` | `auto` | JBU threads (`auto` sizes from RAM/CPU) |
| `--keep-upsample-cache` | off | Keep/reuse per-video `up_cache_v_*` memmap (skips slow FAT32 re-alloc on re-runs) |
| `--no-pipeline-parallel` | off | Disable prep‖infer‖post overlap |

Depth is inferred at a small resolution, then restored to the original video size with **Lanczos + joint bilateral upsampling** guided by the source RGB (same approach as the depth-anything streaming pipeline).

A **prep | infer | post** assembly line overlaps decode, DVD/MPS inference, and CPU JBU. A resource governor watches RAM / CPU / MPS and **pressures toward ≥90%** usable utilization (grows upsample workers and prep queue depth; sheds near OOM).

On **8GB M1**, DiT stays **resident in RAM** across VAE encode/decode (VAE runs **CPU float16**). DiT blocks swap to MPS a few at a time. A float16 **mmap** blob is written once as an OOM fallback (not reloaded every window). Native-res JBU depth stays on a float16 memmap and is **stream-encoded** to MP4 (never a full float32 stack in RAM). If you still see `Killed: 9` during inference, retry with `--height 192 --width 320 --window-size 5 --overlap 2`, or `--cpu`.

**Runtime caches**: upsample memmaps prefer an external **FAT32** volume when mounted. With `--keep-upsample-cache`, the memmap is stored as a stable `up_cache_v_<video>_<hash>/` folder (with `meta.json`) and **reused** on later runs of the same video/geometry — avoiding multi-minute FAT32 re-allocation. Ephemeral `up_cache_<pid>/` dirs are still cleaned automatically. The DiT mmap blob prefers **local APFS** (project `.cache/` or `$TMPDIR`) for fast reload if park is needed. Hugging Face / model downloads stay in the project (`.cache/`, `ckpt/`, `models/`). Upsample caches larger than ~4 GiB are split automatically to respect the FAT32 file-size limit. Override with `--cache-dir` or `DVD_CACHE_DIR`. Output videos still land in `outputs/`.

## CUDA T4 (full GPU)

For NVIDIA GPUs (Colab **T4** ~15 GiB), use [`generate_depth_cuda.py`](generate_depth_cuda.py): **DiT + VAE stay on CUDA** (no RAM↔GPU block swap). Runtime caches use local `.cache/` (or `--cache-dir` / `$DVD_CACHE_DIR`) — **not** FAT32 external volumes.

```bash
python generate_depth_cuda.py \
  --input-video path/to/video.mp4 \
  --output-dir outputs \
  --cache-dir .cache
```

Defaults are **VRAM-scaled**. On Colab/T4 (short windows for host RAM), infer is typically **544×960** so the GPU fills out; high-RAM hosts use upstream **480×640**, window **81**, overlap **21**. Colab/≤14 GiB RAM keeps window **17** / overlap **5**.

### Colab

Open [`notebooks/dvd_colab_t4.ipynb`](notebooks/dvd_colab_t4.ipynb) on a **T4** runtime. It clones from GitHub, installs deps, mounts Drive for **input/output only**, downloads weights, runs a smoke test, then writes `<stem>_depth_gray.mp4` next to your Drive input.

## Layout

```
.
├── generate_depth.py          # CLI (grayscale + MPS block-swap)
├── generate_depth_cuda.py     # CLI (full CUDA GPU residency)
├── notebooks/dvd_colab_t4.ipynb
├── cache_root.py              # External FAT32 + local DiT cache resolution (MPS)
├── pipeline.py                # prep | infer | post assembly line
├── resources.py               # RAM/CPU/MPS concurrency governor
├── upsample.py                # RGB-guided JBU sharp upsample
├── requirements.txt           # Mac-friendly deps (no CUDA packages)
├── scripts/download_weights.py
├── ckpt/                      # DVD weights (downloaded)
├── models/                    # Wan2.1 backbone (auto-downloaded)
├── outputs/                   # Depth videos
└── vendor/DVD/                # EnVision-Research DVD source
```

Runtime caches (MPS): DiT mmap under local `.cache/dit_cache_*` when possible; upsample under `dvd_cache/up_cache_*` on the external volume. CUDA path: local `.cache/` only.

## Notes

- Official DVD scripts hard-code CUDA; this project also runs on **MPS** via `generate_depth.py`.
- CUDA-only extras (`cupy-cuda12x`, `deepspeed`, …) are intentionally omitted from `requirements.txt` (Colab provides CUDA torch).
- Model weights are **CC BY-NC 4.0** (non-commercial). Code is Apache 2.0.
