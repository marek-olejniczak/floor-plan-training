#!/usr/bin/env python3
"""visualize_d3.py — Wizualizacja 30 losowych przykładów z doors_windows/d3."""

import os, sys, random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC = Path("/mnt/d/rzuty/dane/yolo11datasets/doors_windows/d3")
OUTPUT = Path("/mnt/d/rzuty/trening/d3_review.png")
N = 30
COLS, ROWS = 6, 5
CELL_W, CELL_H = 512, 512

CLASS_COLORS = {0: (0, 100, 255), 1: (255, 100, 0)}
CLASS_LABELS = {0: "door", 1: "window"}

random.seed(42)

entries = []
for split in ("train", "valid", "test"):
    img_dir = SRC / split / "images"
    lbl_dir = SRC / split / "labels"
    if not img_dir.is_dir():
        continue
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl = lbl_dir / (img_path.stem + ".txt")
        entries.append((img_path, lbl, split))

print(f"Znaleziono {len(entries)} obrazów w d3")
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

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        font_info = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except:
        font = font_info = ImageFont.load_default()

    lines = []
    if lbl_path.exists():
        with open(lbl_path) as f:
            lines = f.readlines()

    class_counts = {0: 0, 1: 0}
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        if cls not in CLASS_COLORS:
            continue
        class_counts[cls] = class_counts.get(cls, 0) + 1

        cx_n, cy_n, w_n, h_n = map(float, parts[1:5])
        x1 = int((cx_n - w_n / 2) * CELL_W)
        y1 = int((cy_n - h_n / 2) * CELL_H)
        x2 = int((cx_n + w_n / 2) * CELL_W)
        y2 = int((cy_n + h_n / 2) * CELL_H)

        color = CLASS_COLORS[cls]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = CLASS_LABELS[cls]
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - th - 2, x1 + tw + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - th - 2), label, fill=(0, 0, 0), font=font)

    info = f"{split} | door:{class_counts.get(0,0)} win:{class_counts.get(1,0)}"
    bbox = draw.textbbox((0, 0), info, font=font_info)
    iw, ih = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([2, 2, iw + 8, ih + 6], fill=(0, 0, 0, 180))
    draw.text((6, 4), info, fill=(255, 255, 255), font=font_info)

    montage.paste(img_pil, (cx, cy))

# Legenda
leg = Image.new("RGB", (COLS * CELL_W, 60), (30, 30, 30))
draw = ImageDraw.Draw(leg)
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    font_s = ImageFont.truetype("DejaVuSans.ttf", 14)
except:
    font = font_s = ImageFont.load_default()

x = 20
for cls_id, color in CLASS_COLORS.items():
    draw.rectangle([x, 10, x + 30, 40], fill=color)
    draw.text((x + 36, 12), f"{cls_id}={CLASS_LABELS[cls_id]}", fill=(255, 255, 255), font=font)
    x += 180
draw.text((x + 20, 12), f"Total: {len(entries)} img | Pokazano: {len(selected)}", fill=(200,) * 3, font=font_s)

final = Image.new("RGB", (COLS * CELL_W, ROWS * CELL_H + 70), (30, 30, 30))
final.paste(montage, (0, 0))
final.paste(leg, (0, ROWS * CELL_H))
final.save(str(OUTPUT))
print(f"Zapisano: {OUTPUT}  ({final.size})")
