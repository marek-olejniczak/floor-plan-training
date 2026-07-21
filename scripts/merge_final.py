#!/usr/bin/env python3
"""
merge_final.py — Scala wszystkie datasety w jeden kompletny 3-klasowy dataset.
Źródła:
1. pseudo_labeled_walls (już 0=wall, 1=door, 2=window)
2. walls_doors_windows d1+d2 (remap: door=0→1, wall=1→0, window=2→2)
"""
import shutil, sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# --- Sources ---
PSEUDO = Path.home() / "data" / "corrected_walls"         # flat (train/valid/test), filenames have d1_/d2_ prefix
WDW = Path("/mnt/d/rzuty/dane/yolo11datasets/walls_doors_windows")  # d1, d2 subdirs
DST = Path.home() / "data" / "combined_dataset"

SPLITS = ["train", "valid", "test"]

# Walls_doors_windows class remap: door=0→1, wall=1→0, window=2→2
WDW_CLASS_MAP = {0: 1, 1: 0, 2: 2}


def to_bbox_line(line, class_map=None):
    """Convert any label line to 5-field YOLO bbox format: cls cx cy w h."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls_id = int(parts[0])
    if class_map is not None:
        if cls_id not in class_map:
            return None
        cls_id = class_map[cls_id]
    coords = list(map(float, parts[1:]))

    if len(coords) == 4:
        cx, cy, w, h = coords
    elif len(coords) >= 10:
        xs = [coords[i] for i in range(0, 10, 2)]
        ys = [coords[i + 1] for i in range(0, 10, 2)]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin
    else:
        cx, cy, w, h = coords[:4]

    w = max(w, 1e-10)
    h = max(h, 1e-10)
    return f"{cls_id} {cx:.10f} {cy:.10f} {w:.10f} {h:.10f}"


def copy_images_labels(src_dir, dst_dir, prefix, class_map=None):
    """Copy images and labels, convert all labels to 5-field YOLO bbox."""
    img_dir = src_dir / "images"
    lbl_dir = src_dir / "labels"
    if not img_dir.is_dir():
        return 0, 0, 0

    n_img = 0
    class_counts = {}
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue

        dst_img = dst_dir / "images" / f"{prefix}_{img_path.name}"
        shutil.copy2(img_path, dst_img)

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        new_lines = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    bbox_line = to_bbox_line(line, class_map)
                    if bbox_line is not None:
                        new_lines.append(bbox_line)
                        cls_id = int(bbox_line.split()[0])
                        class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

        dst_lbl = dst_dir / "labels" / f"{prefix}_{img_path.stem}.txt"
        dst_lbl.write_text("\n".join(new_lines) + "\n" if new_lines else "")
        n_img += 1

    return n_img, class_counts


def main():
    print("=" * 60)
    print("  FINAL DATASET MERGE")
    print("=" * 60)

    for split in SPLITS:
        (DST / split / "images").mkdir(parents=True, exist_ok=True)
        (DST / split / "labels").mkdir(parents=True, exist_ok=True)

    totals = {"images": 0, 0: 0, 1: 0, 2: 0}

    # --- 1. Pseudo-labeled walls (corrected_walls, flat structure) ---
    print("\n--- Corrected walls (pseudo-labeled, flat) ---")
    for split in SPLITS:
        n, cc = copy_images_labels(PSEUDO / split, DST / split, "pseudo", class_map=None)
        totals["images"] += n
        for k, v in cc.items():
            totals[k] = totals.get(k, 0) + v
        if n:
            print(f"    {split}: {n} img, classes={dict(cc)}")

    # --- 2. Walls-doors-windows (remapped) ---
    print("\n--- Walls+Doors+Windows (ground truth, remapped) ---")
    for ds_dir in sorted(WDW.iterdir()):
        if not ds_dir.is_dir() or not ds_dir.name.startswith("d"):
            continue
        prefix = f"wdw_{ds_dir.name}"
        print(f"  {ds_dir.name}:")
        for split in SPLITS:
            n, cc = copy_images_labels(ds_dir / split, DST / split, prefix, class_map=WDW_CLASS_MAP)
            totals["images"] += n
            for k, v in cc.items():
                totals[k] = totals.get(k, 0) + v
            if n:
                print(f"    {split}: {n} img, classes={dict(cc)}")

    # --- 3. data.yaml ---
    yaml = DST / "data.yaml"
    yaml.write_text(f"""train: train/images
val: valid/images
test: test/images

nc: 3
names:
  0: wall
  1: door
  2: window
""")

    print(f"\n{'='*60}")
    print(f"  PODSUMOWANIE")
    print(f"  Total images: {totals['images']}")
    print(f"  Wall labels:   {totals.get(0, 0)}")
    print(f"  Door labels:   {totals.get(1, 0)}")
    print(f"  Window labels: {totals.get(2, 0)}")
    print(f"  data.yaml:     {yaml}")
    print(f"  Output:        {DST}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
