#!/usr/bin/env python3
"""visualize_dw.py — 30 losowych przykładów z doors_windows d1/d2/d3."""

import sys, os, random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC = Path("/mnt/d/rzuty/dane/yolo11datasets/doors_windows")
OUT_DIR = Path("/mnt/d/rzuty/trening")
N = 30
COLS, ROWS = 6, 5
CELL_W, CELL_H = 512, 512

CLASS_COLORS = {0: (0, 100, 255), 1: (255, 100, 0)}
CLASS_LABELS = {0: "door", 1: "window"}

# Mapy klas dla każdego datasetu
CLASS_MAPS = {
    "d1": {0: "2door", 1: "door", 2: "window"},
    "d2": {0: "2door", 1: "baywindow", 2: "door", 3: "window1", 4: "window2", 5: "window3", 6: "window4", 7: "window5"},
    "d3": {0: "door", 1: "window"},
}

random.seed(42)

try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
    font_info = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    font_leg = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    font_s = ImageFont.truetype("DejaVuSans.ttf", 14)
except:
    font = font_info = font_leg = font_s = ImageFont.load_default()


def make_montage(ds_name):
    ds_dir = SRC / ds_name
    cl_map = CLASS_MAPS[ds_name]

    entries = []
    for split in ("train", "valid", "test"):
        img_dir = ds_dir / split / "images"
        lbl_dir = ds_dir / split / "labels"
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl = lbl_dir / (img_path.stem + ".txt")
            entries.append((img_path, lbl, split))

    print(f"{ds_name}: {len(entries)} obrazów")
    selected = random.sample(entries, min(N, len(entries)))

    montage = Image.new("RGB", (COLS * CELL_W, ROWS * CELL_H), (30, 30, 30))

    for idx, (img_path, lbl_path, split) in enumerate(selected):
        row, col = divmod(idx, COLS)
        cx, cy = col * CELL_W, row * CELL_H

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((CELL_W, CELL_H), Image.LANCZOS)
        draw = ImageDraw.Draw(img_pil)

        lines = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                lines = f.readlines()

        class_counts = {}
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cls_name = cl_map.get(cls, f"cls{cls}")
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            cx_n, cy_n, w_n, h_n = map(float, parts[1:5])
            x1 = int((cx_n - w_n / 2) * CELL_W)
            y1 = int((cy_n - h_n / 2) * CELL_H)
            x2 = int((cx_n + w_n / 2) * CELL_W)
            y2 = int((cy_n + h_n / 2) * CELL_H)

            # Kolor: door=nieb, window=poraż, reszta=szary
            if cls_name in ("window", "baywindow", "window1", "window2", "window3", "window4", "window5", "window6"):
                color = (255, 100, 0)
                label = "window"
            elif cls_name in ("door", "2door"):
                color = (0, 100, 255)
                label = "door"
            else:
                color = (200, 200, 200)
                label = cls_name

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.rectangle([x1, y1 - th - 2, x1 + tw + 4, y1], fill=color)
            draw.text((x1 + 2, y1 - th - 2), label, fill=(0, 0, 0), font=font)

        info = f"{split} | " + " ".join(f"{k}:{v}" for k, v in sorted(class_counts.items()))
        bbox = draw.textbbox((0, 0), info, font=font_info)
        iw, ih = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([2, 2, iw + 8, ih + 6], fill=(0, 0, 0, 180))
        draw.text((6, 4), info, fill=(255, 255, 255), font=font_info)

        montage.paste(img_pil, (cx, cy))

    # Legenda
    leg = Image.new("RGB", (COLS * CELL_W, 60), (30, 30, 30))
    draw = ImageDraw.Draw(leg)
    x = 20
    for cls_id, color in [(0, (0, 100, 255)), (1, (255, 100, 0))]:
        draw.rectangle([x, 10, x + 30, 40], fill=color)
        draw.text((x + 36, 12), f"cls{cls_id}={['door','window'][cls_id]}", fill=(255, 255, 255), font=font_leg)
        x += 200

    draw.text((x + 20, 14), f"Pokazano: {len(selected)}/{len(entries)}", fill=(200,) * 3, font=font_s)

    # Wypisz wszystkie mapowanie klas
    map_info = " | ".join(f"cls{k}={v}" for k, v in sorted(cl_map.items()))
    draw.text((20, 44), f"Klasy: {map_info}", fill=(180,) * 3, font=font_s)

    final = Image.new("RGB", (COLS * CELL_W, ROWS * CELL_H + 70), (30, 30, 30))
    final.paste(montage, (0, 0))
    final.paste(leg, (0, ROWS * CELL_H))

    out_path = OUT_DIR / f"dw_{ds_name}_review.png"
    final.save(str(out_path))
    print(f"  Zapisano: {out_path}")


if __name__ == "__main__":
    for ds in ("d1", "d2", "d3"):
        make_montage(ds)
