#!/usr/bin/env python3
"""Download DepthCrafter + SVD-XT weights from Hugging Face into ./ckpt."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download DepthCrafter UNet and SVD-XT backbone weights"
    )
    parser.add_argument(
        "--unet-repo",
        default="tencent/DepthCrafter",
        help="Hugging Face repo id for the DepthCrafter UNet",
    )
    parser.add_argument(
        "--svd-repo",
        default="stabilityai/stable-video-diffusion-img2vid-xt",
        help="Hugging Face repo id for the SVD-XT backbone",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("ckpt"),
        help="Root directory for checkpoints",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Repo revision / branch",
    )
    args = parser.parse_args()

    root = args.local_dir
    root.mkdir(parents=True, exist_ok=True)
    unet_dir = root / "DepthCrafter"
    svd_dir = root / "stable-video-diffusion-img2vid-xt"

    print(f"Downloading {args.unet_repo}@{args.revision} -> {unet_dir.resolve()}")
    snapshot_download(
        repo_id=args.unet_repo,
        revision=args.revision,
        local_dir=str(unet_dir),
        local_dir_use_symlinks=False,
    )
    print(
        f"Downloading {args.svd_repo}@{args.revision} -> {svd_dir.resolve()}\n"
        "  (SVD-XT usually requires a Hugging Face token + license accept)"
    )
    snapshot_download(
        repo_id=args.svd_repo,
        revision=args.revision,
        local_dir=str(svd_dir),
        local_dir_use_symlinks=False,
        allow_patterns=[
            "feature_extractor/*",
            "image_encoder/*",
            "scheduler/*",
            "unet/*",
            "vae/*",
            "model_index.json",
            "*.json",
            "*fp16*",
            "*.safetensors",
        ],
    )
    print(
        "Done. Expected:\n"
        f"  {unet_dir}/ (config.json, diffusion_pytorch_model*.safetensors)\n"
        f"  {svd_dir}/ (SVD-XT fp16 components)"
    )


if __name__ == "__main__":
    main()
