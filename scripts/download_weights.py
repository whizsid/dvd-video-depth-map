#!/usr/bin/env python3
"""Download DVD checkpoints from Hugging Face into ./ckpt."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DVD model weights")
    parser.add_argument(
        "--repo",
        default="FayeHongfeiZhang/DVD",
        help="Hugging Face repo id",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("ckpt"),
        help="Where to store checkpoints",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Repo revision / branch",
    )
    args = parser.parse_args()

    args.local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo}@{args.revision} -> {args.local_dir.resolve()}")
    snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=str(args.local_dir),
        local_dir_use_symlinks=False,
    )
    print("Done. Expected files: ckpt/dvd_1.1.safetensors, ckpt/model_config.yaml")


if __name__ == "__main__":
    main()
