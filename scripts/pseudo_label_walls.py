#!/usr/bin/env python3
"""
pseudo_label_walls.py — Nanosi predykcje drzwi/okien na walls d1+d2.
Używa wytrenowanego modelu doors_windows_v1 do wykrycia door (1) i window (2),
a następnie łączy z istniejącymi adnotacjami wall (0).
"""
import os, shutil, sys
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from pathlib import Path
from ultralytics import YOLO

# --- Config ---
MODEL_PATH = Path.home() / "projects" / "trening" / "runs" / "doors_windows_v1" / "weights" / "best.pt"
WALLS_SRC = Path("/mnt/d/rzuty/dane/yolo11datasets/walls")
OUTPUT = Path.home() / "data" / "pseudo_labeled_walls"
CONF_THRESHOLD = 0.5
SPLITS = ["train", "valid", "test"]

# Class mapping for model output: model outputs 0=door, 1=window
# We want final labels: 0=wall, 1=door, 2=window
DOOR_CLASS = 1
WINDOW_CLASS = 2


def main():
    print(f"Loading model from {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    model.model.cuda()

    total_imgs = 0
    total_pseudo_door = 0
    total_pseudo_window = 0

    for ds_name in sorted(WALLS_SRC.iterdir()):
        if not ds_name.is_dir() or not ds_name.name.startswith("d"):
            continue
        print(f"\n=== {ds_name.name} ===")

        for split in SPLITS:
            src_img_dir = ds_name / split / "images"
            src_lbl_dir = ds_name / split / "labels"
            if not src_img_dir.is_dir():
                continue

            dst_img_dir = OUTPUT / ds_name.name / split / "images"
            dst_lbl_dir = OUTPUT / ds_name.name / split / "labels"
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_lbl_dir.mkdir(parents=True, exist_ok=True)

            img_files = sorted([
                p for p in src_img_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            ])
            print(f"  {split}: {len(img_files)} images")

            BATCH_SIZE = 32
            for i in range(0, len(img_files), BATCH_SIZE):
                batch = img_files[i : i + BATCH_SIZE]
                batch_paths = [str(p) for p in batch]

                # Read existing wall labels for all images
                batch_wall_lines = {}
                for img_path in batch:
                    lbl_path = src_lbl_dir / (img_path.stem + ".txt")
                    wall_lines = []
                    if lbl_path.exists():
                        with open(lbl_path) as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    parts = line.split()
                                    if len(parts) >= 5 and parts[0] == "0":
                                        wall_lines.append(line)
                    batch_wall_lines[img_path.name] = wall_lines

                # Batch inference
                results = model(batch_paths, conf=CONF_THRESHOLD, verbose=False)

                for img_path, result in zip(batch, results):
                    # Copy image
                    shutil.copy2(img_path, dst_img_dir / img_path.name)

                    pseudo_lines = list(batch_wall_lines[img_path.name])
                    if result.boxes is not None:
                        for box, cls_id in zip(result.boxes.xywhn, result.boxes.cls):
                            cls_id = int(cls_id)
                            if cls_id == 0:
                                target_class = DOOR_CLASS
                                total_pseudo_door += 1
                            elif cls_id == 1:
                                target_class = WINDOW_CLASS
                                total_pseudo_window += 1
                            else:
                                continue
                            cx, cy, w, h = box.tolist()
                            pseudo_lines.append(f"{target_class} {cx:.10f} {cy:.10f} {w:.10f} {h:.10f}")

                    # Write combined labels
                    dst_lbl = dst_lbl_dir / (img_path.stem + ".txt")
                    dst_lbl.write_text("\n".join(pseudo_lines) + "\n" if pseudo_lines else "")
                    total_imgs += 1

                if (i // BATCH_SIZE + 1) % 20 == 0:
                    print(f"    ... {i + len(batch)}/{len(img_files)}")

    print(f"\n{'='*50}")
    print(f"  PSEUDO-LABELING COMPLETE")
    print(f"  Total images: {total_imgs}")
    print(f"  + Door labels: {total_pseudo_door}")
    print(f"  + Window labels: {total_pseudo_window}")
    print(f"  Output: {OUTPUT}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
