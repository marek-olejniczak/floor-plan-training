#!/usr/bin/env python3
"""
prepare_walls_dataset.py — Przygotowuje dataset tylko-sciany (1 klasa, YOLO bbox).
Źródla:
  - ~/data/walls/d1/               (class 0, mieszany OBB+YOLO)
  - /mnt/d/rzuty/dane/yolo11datasets/walls/d2/  (class 0, mieszany OBB+YOLO)
  - ~/data/walls_doors_windows/d1/ (class 1, YOLO)
  - ~/data/walls_doors_windows/d2/ (class 1, OBB polygon)

Wyjscie: ~/data/walls_only/  (YOLO detect, 5 pol: class cx cy w h, class=0)
"""

import shutil
import sys
from pathlib import Path

import numpy as np

OUT = Path.home() / "data" / "walls_only"
SPLITS = ["train", "valid", "test"]

SOURCES = {
    "walls_d1": {
        "path": Path.home() / "data" / "walls" / "d1",
        "wall_class": 0,
    },
    "walls_d2": {
        "path": Path("/mnt/d/rzuty/dane/yolo11datasets/walls/d2"),
        "wall_class": 0,
    },
    "wdw_d1": {
        "path": Path.home() / "data" / "walls_doors_windows" / "d1",
        "wall_class": 1,
    },
    "wdw_d2": {
        "path": Path.home() / "data" / "walls_doors_windows" / "d2",
        "wall_class": 1,
    },
}


def obb_to_yolo_bbox(values):
    pairs = np.array(values, dtype=float).reshape(-1, 2)
    x_min = pairs[:, 0].min()
    x_max = pairs[:, 0].max()
    y_min = pairs[:, 1].min()
    y_max = pairs[:, 1].max()
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    if w <= 0 or h <= 0:
        return None
    return cx, cy, w, h


def convert_label_line(line, wall_class):
    parts = line.strip().split()
    if not parts:
        return None
    cls = int(parts[0])
    if cls != wall_class:
        return None

    coords = parts[1:]
    n = len(coords)

    if n == 4:
        # juz YOLO bbox: cx cy w h
        cx, cy, w, h = map(float, coords)
        if w <= 0 or h <= 0:
            return None
        return f"0 {cx:.10f} {cy:.10f} {w:.10f} {h:.10f}"

    if n >= 4 and n % 2 == 0:
        # OBB polygon lub inny wielokat
        result = obb_to_yolo_bbox(coords)
        if result is None:
            return None
        cx, cy, w, h = result
        return f"0 {cx:.10f} {cy:.10f} {w:.10f} {h:.10f}"

    return None


def process_source(name, cfg, split, seen_images, local_stats):
    src = cfg["path"] / split
    img_dir = src / "images"
    lbl_dir = src / "labels"
    if not img_dir.exists() or not lbl_dir.exists():
        local_stats[split]["skipped_no_dir"] = True
        return

    wall_class = cfg["wall_class"]
    out_dir = OUT / split
    out_img = out_dir / "images"
    out_lbl = out_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    lbl_files = sorted(lbl_dir.glob("*.txt"))
    for lbl_path in lbl_files:
        stem = lbl_path.stem
        if stem in seen_images:
            local_stats[split]["dedup_skipped"] += 1
            continue
        seen_images.add(stem)

        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            local_stats[split]["no_image"] += 1
            continue

        out_lines = []
        with open(lbl_path) as f:
            for line in f:
                converted = convert_label_line(line, wall_class)
                if converted is not None:
                    out_lines.append(converted)

        if not out_lines:
            local_stats[split]["empty_after_convert"] += 1
            continue

        shutil.copy2(img_path, out_img / img_path.name)
        (out_lbl / lbl_path.name).write_text("\n".join(out_lines) + "\n")
        local_stats[split]["copied"] += 1


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    stats = {}
    for name in SOURCES:
        stats[name] = {}
        for s in SPLITS:
            stats[name][s] = {
                "copied": 0,
                "dedup_skipped": 0,
                "no_image": 0,
                "empty_after_convert": 0,
                "skipped_no_dir": False,
            }

    seen_images = set()

    for name, cfg in SOURCES.items():
        print(f"\n[{name}]")
        for split in SPLITS:
            process_source(name, cfg, split, seen_images, stats[name])
            s = stats[name][split]
            if s["skipped_no_dir"]:
                print(f"  {split}: (brak katalogu)")
            else:
                print(f"  {split}: skopiowano {s['copied']}, dedup {s['dedup_skipped']}, brak_img {s['no_image']}, puste_po_konwersji {s['empty_after_convert']}")

    # Podsumowanie
    print("\n" + "=" * 50)
    print("PODSUMOWANIE")
    print("=" * 50)
    total = {"train": 0, "valid": 0, "test": 0}
    for name in SOURCES:
        for split in SPLITS:
            total[split] += stats[name][split]["copied"]
    for split in SPLITS:
        out_dir = OUT / split
        img_count = len(list((out_dir / "images").glob("*"))) if (out_dir / "images").exists() else 0
        print(f"  {split}: {img_count} obrazow")

    # data.yaml
    yaml_path = OUT / "data.yaml"
    yaml_path.write_text(
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n"
        f"\n"
        f"nc: 1\n"
        f"names: ['wall']\n"
    )
    print(f"\ndata.yaml -> {yaml_path}")


if __name__ == "__main__":
    main()
