#!/usr/bin/env python3
"""
flatten_pseudo.py — Flatten pseudo_labeled_walls/d{1,2} into raw_predictions/
with d1_/d2_ prefix for unique filenames.
"""
import shutil, sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

PSEUDO = Path.home() / "data" / "pseudo_labeled_walls"
DST = Path.home() / "data" / "raw_predictions"
SPLITS = ["train", "valid", "test"]


def main():
    total_img = 0
    total_lbl = 0

    for ds_dir in sorted(PSEUDO.iterdir()):
        if not ds_dir.is_dir():
            continue
        prefix = ds_dir.name  # d1 or d2
        print(f"\n=== {ds_dir.name} ===")

        for split in SPLITS:
            src_img = ds_dir / split / "images"
            src_lbl = ds_dir / split / "labels"
            if not src_img.is_dir():
                continue

            dst_img = DST / split / "images"
            dst_lbl = DST / split / "labels"
            dst_img.mkdir(parents=True, exist_ok=True)
            dst_lbl.mkdir(parents=True, exist_ok=True)

            n_img = 0
            for p in sorted(src_img.iterdir()):
                if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    continue
                fname = f"{prefix}_{p.name}"
                shutil.copy2(p, dst_img / fname)

                lbl_src = src_lbl / (p.stem + ".txt")
                if lbl_src.exists():
                    lbl_dst = dst_lbl / (p.stem + ".txt")
                    shutil.copy2(lbl_src, dst_lbl / f"{prefix}_{p.stem}.txt")
                    n_img += 1

            print(f"  {split}: {n_img} images")
            total_img += n_img

    print(f"\nTotal: {total_img} images → {DST}")


if __name__ == "__main__":
    main()
