#!/usr/bin/env python3
"""
merge_doors_windows.py — Scala d1+d2+d3 z doors_windows w jeden dataset z 2 klasami.
Mapowanie: 2door,door -> 0:door | baywindow,window,window1..6 -> 1:window
"""

import shutil
from pathlib import Path

SRC = Path("/mnt/d/rzuty/dane/yolo11datasets/doors_windows")
DST = Path.home() / "data" / "merged_doors_windows"
SPLITS = ["train", "valid", "test"]

CLASS_MAP = {
    "d1": {0: 0, 1: 0, 2: 1},       # 2door->door, door->door, window->window
    "d2": {0: 0, 1: 1, 2: 0,          # 2door->door, baywindow->window, door->door
           3: 1, 4: 1, 5: 1, 6: 1, 7: 1},  # window1..6 -> window
    "d3": {0: 0, 1: 1},               # door->door, window->window
}


def process_labels(src_labels, dst_labels, prefix, class_map):
    kept = {0: 0, 1: 0}
    for lbl_file in sorted(src_labels.glob("*.txt")):
        dst = dst_labels / f"{prefix}_{lbl_file.name}"
        with open(lbl_file) as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            old_cls = int(parts[0])
            if old_cls not in class_map:
                continue
            new_cls = class_map[old_cls]
            parts[0] = str(new_cls)
            new_lines.append(" ".join(parts))
            kept[new_cls] += 1
        dst.write_text("\n".join(new_lines) + "\n" if new_lines else "")
    return kept


def copy_images(src_images, dst_images, prefix):
    count = 0
    for img in sorted(src_images.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        shutil.copy2(img, dst_images / f"{prefix}_{img.name}")
        count += 1
    return count


def main():
    for split in SPLITS:
        (DST / split / "images").mkdir(parents=True, exist_ok=True)
        (DST / split / "labels").mkdir(parents=True, exist_ok=True)

    totals = {"images": 0, "door": 0, "window": 0}

    for ds_name in sorted(SRC.iterdir()):
        if not ds_name.is_dir() or ds_name.name.startswith("."):
            continue
        prefix = ds_name.name
        cmap = CLASS_MAP.get(ds_name.name, {})
        print(f"\n--- {ds_name.name} ---")

        for split in SPLITS:
            src_img = ds_name / split / "images"
            src_lbl = ds_name / split / "labels"
            if not src_img.is_dir():
                continue

            n_img = copy_images(src_img, DST / split / "images", prefix)
            kept = process_labels(src_lbl, DST / split / "labels", prefix, cmap)
            totals["images"] += n_img
            totals["door"] += kept[0]
            totals["window"] += kept[1]
            print(f"  {split}: {n_img} img, door={kept[0]}, window={kept[1]}")

    yaml = DST / "data.yaml"
    yaml.write_text(f"""train: train/images
val: valid/images
test: test/images

nc: 2
names:
  0: door
  1: window
""")

    print(f"\n{'='*50}")
    print(f"  PODSUMOWANIE")
    print(f"  images: {totals['images']}")
    print(f"  door:   {totals['door']}")
    print(f"  window: {totals['window']}")
    print(f"  data.yaml: {yaml}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
