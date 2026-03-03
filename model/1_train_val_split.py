"""
Step 1: Split dataset into train and validation folders.

Usage:
    python 1_train_val_split.py --datapath /path/to/dataset
    python 1_train_val_split.py --datapath /path/to/dataset --train_pct 0.85
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path


def split_dataset(data_path: str, train_percent: float = 0.8) -> None:
    data_path = Path(data_path)
    if not data_path.is_dir():
        print(f"[ERROR] Directory not found: {data_path}")
        sys.exit(1)

    if not (0.01 <= train_percent <= 0.99):
        print("[ERROR] --train_pct must be between 0.01 and 0.99.")
        sys.exit(1)

    input_image_path = data_path / "images"
    input_label_path = data_path / "labels"

    if not input_image_path.exists():
        print(f"[ERROR] No 'images' folder found inside {data_path}")
        sys.exit(1)

    # Output directories (cross-platform)
    cwd = Path.cwd()
    dirs = {
        "train_img":  cwd / "data" / "train" / "images",
        "train_lbl":  cwd / "data" / "train" / "labels",
        "val_img":    cwd / "data" / "validation" / "images",
        "val_lbl":    cwd / "data" / "validation" / "labels",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Collect and shuffle images
    img_files = [
        p for p in input_image_path.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    if not img_files:
        print(f"[ERROR] No image files found in {input_image_path}")
        sys.exit(1)

    random.shuffle(img_files)
    split_idx = int(len(img_files) * train_percent)
    splits = {
        "train": img_files[:split_idx],
        "val":   img_files[split_idx:],
    }

    print(f"Total images : {len(img_files)}")
    print(f"  → Train    : {len(splits['train'])}")
    print(f"  → Val      : {len(splits['val'])}")

    for split_name, files in splits.items():
        img_dst = dirs[f"{split_name[:3]}_img"] if split_name == "train" else dirs["val_img"]
        lbl_dst = dirs[f"{split_name[:3]}_lbl"] if split_name == "train" else dirs["val_lbl"]

        for img_path in files:
            shutil.copy2(img_path, img_dst / img_path.name)

            lbl_path = input_label_path / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_dst / lbl_path.name)

    print("Dataset split complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val folders.")
    parser.add_argument("--datapath",  required=True, help="Path to folder containing images/ and labels/")
    parser.add_argument("--train_pct", type=float, default=0.8, help="Fraction for training (default: 0.8)")
    args = parser.parse_args()
    split_dataset(args.datapath, args.train_pct)
