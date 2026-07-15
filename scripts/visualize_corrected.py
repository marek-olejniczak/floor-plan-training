#!/usr/bin/env python3
"""
visualize_corrected.py — Wizualizacja corrected_walls z info o rotacji.
"""

import os, sys, random, math, json
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

DATASET = Path.home() / "data" / "corrected_walls"
META = DATASET / "rotation_meta.json"
OUTPUT = Path("/mnt/d/rzuty/trening/corrected_review.png")
N_EXAMPLES = 30
GRID_COLS = 6
GRID_ROWS = 5
CELL_W, CELL_H = 512, 512

CLASS_COLORS = {
    0: (0, 200, 0),    # wall
    1: (255, 100, 0),   # door
    2: (0, 100, 255),   # window
}
CLASS_LABELS = {0: "wall", 1: "door", 2: "window"}


def parse_label_line(line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls_id = int(parts[0])
    coords = list(map(float, parts[1:]))
    if len(coords) >= 10:
        pts = [(coords[i], coords[i + 1]) for i in range(0, 10, 2)]
        return cls_id, pts
    cx, cy, w, h = coords[:4]
    pts = [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2),
           (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2),
           (cx - w / 2, cy - h / 2)]
    return cls_id, pts


def draw_annotations(img_pil, label_lines, img_w, img_h):
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("DejaVuSans-Bold.ttf", 11)
    except (OSError, AttributeError):
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
            font_small = ImageFont.truetype("DejaVuSans.ttf", 11)
        except (OSError, AttributeError):
            font = font_small = ImageFont.load_default()

    for line in label_lines:
        parsed = parse_label_line(line)
        if parsed is None:
            continue
        cls_id, pts_norm = parsed
        color = CLASS_COLORS.get(cls_id, (255, 255, 0))
        label = CLASS_LABELS.get(cls_id, str(cls_id))

        pts = [(int(x * img_w), int(y * img_h)) for x, y in pts_norm]
        thickness = 3
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=color, width=thickness)

        text = label
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx, ty = pts[0]
        draw.rectangle([tx, ty - th - 2, tx + tw + 4, ty], fill=color)
        draw.text((tx + 2, ty - th - 2), text, fill=(0, 0, 0), font=font)


def create_montage():
    # Wczytaj metadane rotacji
    rotated_names = set()
    if META.exists():
        with open(META) as f:
            meta = json.load(f)
        rotated_names = {e["name"] for e in meta.get("rotated_examples", [])}
        stats = meta.get("stats", {})
        print(f"  Z metadanych: rotated={stats.get('rotated', '?')}/{stats.get('total', '?')}")
    else:
        print("  [UWAGA] Brak rotation_meta.json")

    # Zbierz obrazki
    all_entries = []
    for split in ("train", "valid", "test"):
        img_dir = DATASET / split / "images"
        lbl_dir = DATASET / split / "labels"
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            src_name = img_path.stem[3:] + ".jpg"  # usuń prefiks "d1_"/"d2_"
            was_rotated = src_name in rotated_names
            all_entries.append((img_path, lbl_path, was_rotated, split))

    if not all_entries:
        print("[BLAD] Brak obrazów!")
        return

    random.seed(42)
    # Wybierz po 15 rotowanych i 15 nierotowanych
    rotated_entries = [e for e in all_entries if e[2]]
    non_rotated_entries = [e for e in all_entries if not e[2]]

    selected = []
    n_rot = min(N_EXAMPLES // 2, len(rotated_entries))
    n_non = N_EXAMPLES - n_rot
    selected += random.sample(rotated_entries, n_rot)
    selected += random.sample(non_rotated_entries, min(n_non, len(non_rotated_entries)))
    random.shuffle(selected)

    n_rot_selected = sum(1 for e in selected if e[2])
    print(f"Wybrano {len(selected)} przykładów ({n_rot_selected} rotowanych, {len(selected)-n_rot_selected} nierotowanych)")

    montage = Image.new("RGB", (GRID_COLS * CELL_W, GRID_ROWS * CELL_H), (30, 30, 30))

    for idx, (img_path, lbl_path, was_rotated, split) in enumerate(selected):
        row, col = divmod(idx, GRID_COLS)
        cx, cy = col * CELL_W, row * CELL_H

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((CELL_W, CELL_H), Image.LANCZOS)

        lines = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                lines = f.readlines()

        draw_annotations(img_pil, lines, CELL_W, CELL_H)

        draw = ImageDraw.Draw(img_pil)
        try:
            font_info = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        except (OSError, AttributeError):
            try:
                font_info = ImageFont.truetype("DejaVuSans.ttf", 16)
            except (OSError, AttributeError):
                font_info = ImageFont.load_default()

        # Etykieta rotacji
        total_objs = sum(1 for l in lines if l.strip())
        if was_rotated:
            # Znajdź źródłowy kąt z metadanych
            src_name = img_path.stem[3:] + ".jpg"
            rot_info = ""
            if META.exists():
                with open(META) as f:
                    meta = json.load(f)
                for ex in meta.get("rotated_examples", []):
                    if ex["name"] == src_name:
                        rot_info = f"  rot:{ex['angle']}°  dev:{ex['max_dev']}°"
                        break
            info = f"ROTATED{rot_info} | obj: {total_objs}"
            bg_color = (80, 20, 20)
        else:
            info = f"AXIS-ALIGNED | {split} | obj: {total_objs}"
            bg_color = (20, 40, 20)

        bbox = draw.textbbox((0, 0), info, font=font_info)
        iw, ih = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([2, 2, iw + 8, ih + 6], fill=bg_color)
        draw.text((6, 4), info, fill=(255, 255, 255), font=font_info)

        montage.paste(img_pil, (cx, cy))

    # Legenda
    leg_img = Image.new("RGB", (GRID_COLS * CELL_W, 80), (30, 30, 30))
    draw = ImageDraw.Draw(leg_img)
    try:
        font_leg = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except (OSError, AttributeError):
        font_leg = font_small = ImageFont.load_default()

    x = 20
    for cls_id, color in CLASS_COLORS.items():
        draw.rectangle([x, 10, x + 30, 40], fill=color)
        draw.text((x + 36, 12), f"{cls_id}={CLASS_LABELS[cls_id]}", fill=(255, 255, 255), font=font_leg)
        x += 180

    x += 20
    draw.rectangle([x, 10, x + 20, 40], fill=(80, 20, 20))
    draw.text((x + 26, 12), "= rotated", fill=(255, 255, 255), font=font_small)
    x += 120
    draw.rectangle([x, 10, x + 20, 40], fill=(20, 40, 20))
    draw.text((x + 26, 12), "= axis-aligned", fill=(255, 255, 255), font=font_small)

    draw.text((20, 52), f"Total: {len(all_entries)} img | Pokazano: {len(selected)} ({n_rot_selected} rotated)",
              fill=(200, 200, 200), font=font_small)

    final = Image.new("RGB", (GRID_COLS * CELL_W, GRID_ROWS * CELL_H + 90), (30, 30, 30))
    final.paste(montage, (0, 0))
    final.paste(leg_img, (0, GRID_ROWS * CELL_H))

    final.save(str(OUTPUT))
    print(f"Mozaika zapisana do {OUTPUT}")
    print(f"  Rozmiar: {final.size} px")


if __name__ == "__main__":
    create_montage()
