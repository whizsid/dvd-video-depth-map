# DepthCrafter Video Depth (CUDA T4) + DA3 Photos

Generate **grayscale depth-map videos** with [Tencent/DepthCrafter](https://github.com/Tencent/DepthCrafter) on **CUDA** (Colab **T4** oriented), and **still-photo depth PFMs** with Depth Anything 3 on MPS/CPU.

Video inference is **CUDA-only**. Photo depth stays on Apple Silicon / CPU via DA3.

## Setup

```bash
# 1. Create & activate the venv (Mac: photo/DA3 + shared deps)
python3.10 -m venv .venv
source .venv/bin/activate

# Or use the helper script:
# bash scripts/setup_venv.sh

# 2. Install project deps
pip install -r requirements.txt

# 3. Download DepthCrafter + SVD-XT weights (~GB-scale) — needs HF login for SVD
#    Accept the license at https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
HF_TOKEN=… python scripts/download_weights.py
```

Weights land in `ckpt/DepthCrafter/` and `ckpt/stable-video-diffusion-img2vid-xt/`. If those dirs are missing, `generate_depth.py` falls back to Hugging Face ids (`tencent/DepthCrafter`, `stabilityai/stable-video-diffusion-img2vid-xt`).

Vendored DepthCrafter source lives under `vendor/DepthCrafter/` (imported via `sys.path`; no editable install required — upstream pins Python ≥3.13).

### Photo depth (CR3 / stills, DA3)

For **still photos** (Canon CR3 and other RAW, plus JPEG/PNG/TIFF), install Depth Anything 3 and optional RAW support:

```bash
pip install rawpy
pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git
# If rawpy fails to build on macOS:
# brew install libraw && pip install rawpy
```

Generate six PFMs per image (depth, inverse, min/max and their complements):

```bash
python generate_depth_photo.py \
  --input-dir /path/to/photos \
  --device mps \
  --model da3-small
```

Output layout (child folder named after each image stem):

```
photos/IMG_1234.CR3
photos/IMG_1234/depth.pfm
photos/IMG_1234/inverse.pfm
photos/IMG_1234/min.pfm
photos/IMG_1234/min_inverse.pfm
photos/IMG_1234/max.pfm
photos/IMG_1234/max_inverse.pfm
```

Uses **DA3-SMALL** by default (fast on MPS). Depth is inferred at **`process_res=auto`**, then restored to native resolution with **Lanczos + RGB-guided JBU**, followed by per-frame **denoise** and **edge smoothing**.

| Flag | Default | Notes |
|------|---------|--------|
| `--model` | `da3-small` | Also: `da3-base`, `da3-large`, `da3metric-large`, … |
| `--process-res` | `auto` | Long-side infer target; auto scales to RAM/model (up to 1680) |
| `--no-upsample` | off | Keep infer resolution |
| `--no-denoise` | off | Skip bilateral noise removal |
| `--no-edge-smooth` | off | Skip RGB-guided edge polish |
| `--overwrite` | off | Re-process folders that already have all six PFMs |

## Generate a grayscale depth video (CUDA)

```bash
source .venv/bin/activate   # or Colab runtime with CUDA torch
# Prefer a T4 / ≥9 GiB GPU; defaults auto-tune for Colab T4
HF_TOKEN=… python generate_depth.py \
  --input-video path/to/video.mp4 \
  --output-dir outputs \
  --cache-dir .cache
```

Output: `outputs/<name>_depth_gray.mp4`

`generate_depth_cuda.py` is a thin alias of the same CLI (older docs / Colab cells keep working).

### Useful flags

| Flag | Default | Notes |
|------|---------|--------|
| `--max-res` | auto | Long-side infer res (aligned to 64). T4/Colab: **512** (~9 GiB); high quality: **1024** (~26 GiB) |
| `--window-size` / `--overlap` | auto | Temporal segment size / stitch overlap (e.g. 40–110 / ≤25) |
| `--cpu-offload` | `auto` | `none` \| `model` \| `sequential` (T4/Colab → sequential/model) |
| `--num-inference-steps` | `5` | Denoising steps |
| `--guidance-scale` | `1.0` | Classifier-free guidance |
| `--unet-path` / `--svd-path` | auto | Local `ckpt/…` dirs or HF ids |
| `--no-upsample` | off | Skip RGB-guided JBU (keep infer res) |
| `--upsample-device` | `cuda` | JBU on CUDA or CPU |
| `--keep-upsample-cache` | off | Reuse stable `up_cache_v_*` memmap |
| `--no-denoise` | off | Skip bilateral noise removal |
| `--no-stabilize` | off | Skip shot band-lock + temporal median |
| `--no-dark-enhance` | off | Skip CLAHE + Scharr/saturation pre-pass |
| `--dark-enhance-strength` | `1.0` | 0–2; adaptive contrast/edge amount |
| `--max-frames` | off | Smoke-test first N frames |
| `--cache-dir` | `.cache` | Upsample memmaps. Also `$DEPTHCRAFTER_CACHE_DIR` / `$DVD_CACHE_DIR` |

Depth is inferred at a reduced resolution, then restored to the original video size with **Lanczos + joint bilateral upsampling** guided by the source RGB.

Dark scenes get an adaptive **CLAHE + cheap Scharr/saturation** pre-pass **before DepthCrafter**, so silhouettes stay visible in crushed shadows. JBU and stabilize still use the original RGB.

On **Colab T4**, the model is freed before CUDA JBU so VRAM is available for upsample. Prefer local `.cache/` for memmaps — not Google Drive mounts.

### Colab

Open [`notebooks/dvd_colab_t4.ipynb`](notebooks/dvd_colab_t4.ipynb) on a **T4** runtime. It clones from GitHub, installs deps, mounts Drive for **input/output only**, downloads DepthCrafter + SVD-XT (HF token required for SVD), then writes `<stem>_depth_gray.mp4` next to your Drive input.

## Layout

```
.
├── generate_depth.py          # CLI (DepthCrafter CUDA + post)
├── generate_depth_cuda.py     # Alias → generate_depth.main
├── generate_depth_photo.py    # CLI (CR3/stills → DA3 depth PFMs)
├── dark_enhance.py            # Dark-scene CLAHE + Scharr pre-pass
├── edge_smooth.py             # RGB-guided depth edge polish (photos)
├── pfm.py                     # Middlebury PFM writer
├── notebooks/dvd_colab_t4.ipynb
├── cache_root.py              # Cache path helpers
├── pipeline.py                # DepthCrafter load / run / release
├── resources.py               # RAM/CPU concurrency helpers
├── upsample.py                # RGB-guided JBU sharp upsample
├── requirements.txt
├── scripts/download_weights.py
├── ckpt/                      # DepthCrafter + SVD-XT (downloaded)
├── outputs/                   # Depth videos
└── vendor/DepthCrafter/       # Tencent DepthCrafter source
```

## Notes

- DepthCrafter inference is **CUDA-only** in this project (official memory: ~9 GiB @ 512, ~26 GiB @ 1024×576).
- Photo depth (DA3) still runs on **MPS/CPU**.
- DepthCrafter is licensed for **academic / research / education** use (see `vendor/DepthCrafter/LICENSE`). SVD-XT is under Stability’s community license and usually needs a Hugging Face token + license accept.
- Optional `xformers` improves attention memory on CUDA when installed; otherwise attention slicing is used.
