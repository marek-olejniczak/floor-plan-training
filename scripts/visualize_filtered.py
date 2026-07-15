#!/usr/bin/env python3
"""
visualize_filtered.py — 10 examples, all windows in one view.
Kept=blue outline+conf, removed=thick red/orange outline + fill + reason.
Walls/doors drawn thin gray for orientation context.
"""

import os, sys, random, json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

RAW = Path.home() / "data" / "raw_predictions"
FILTERED = Path.home() / "data" / "corrected_walls"
DECISIONS_PATH = FILTERED / "filter_decisions.json"
OUTPUT = Path("/mnt/d/rzuty/trening") / "filter_review.png"

N_EXAMPLES = 10
GRID_COLS = 2
GRID_ROWS = 5
CELL_W, CELL_H = 640, 640


def parse_label_line(line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls_id = int(parts[0])
    coords = list(map(float, parts[1:]))
    conf = float(coords[4]) if len(coords) >= 5 else 0.0
    fs = int(coords[5]) if len(coords) >= 6 else 0
    if len(coords) >= 10:
        pts = [(coords[i], coords[i + 1]) for i in range(0, 10, 2)]
        return cls_id, pts, conf, fs
    cx, cy, w, h = coords[:4]
    pts = [
        (cx - w / 2, cy - h / 2),
        (cx + w / 2, cy - h / 2),
        (cx + w / 2, cy + h / 2),
        (cx - w / 2, cy + h / 2),
        (cx - w / 2, cy - h / 2),
    ]
    return cls_id, pts, conf, fs


def create_montage():
    if not DECISIONS_PATH.exists():
        print(f"[BLAD] Brak {DECISIONS_PATH}")
        return
    with open(DECISIONS_PATH) as f:
        meta = json.load(f)
    filter_totals = meta.get("filter_totals", {})
    cfg = meta.get("config", {})

    candidates = []
    for split in ("train", "valid", "test"):
        img_dir = RAW / split / "images"
        lbl_dir = FILTERED / split / "labels"
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue
            n_removed = 0
            n_kept = 0
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 7 and parts[0] == "2":
                        fs = int(parts[6])
                        if fs != 0:
                            n_removed += 1
                        else:
                            n_kept += 1
            if n_removed > 0:
                candidates.append((img_path, lbl_path, n_kept, n_removed))

    if not candidates:
        print("[BLAD] Brak obrazow!")
        return

    random.shuffle(candidates)
    selected = candidates[:N_EXAMPLES]
    print(f"  Wybrano {len(selected)} przykladow")

    montage = Image.new("RGB", (GRID_COLS * CELL_W, GRID_ROWS * CELL_H), (30, 30, 30))

    for idx, (img_path, lbl_path, n_kept, n_removed) in enumerate(selected):
        row, col = divmod(idx, GRID_COLS)
        cx, cy = col * CELL_W, row * CELL_H

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((CELL_W, CELL_H), Image.LANCZOS)
        overlay = Image.new("RGBA", (CELL_W, CELL_H), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        draw = ImageDraw.Draw(img_pil)

        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
            font_info = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        except (OSError, AttributeError):
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 14)
                font_info = ImageFont.truetype("DejaVuSans.ttf", 16)
            except (OSError, AttributeError):
                font = font_info = ImageFont.load_default()

        lines = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                lines = f.readlines()

        for line in lines:
            parsed = parse_label_line(line)
            if parsed is None:
                continue
            cls_id, pts_norm, conf, fs = parsed
            if cls_id == 2 and conf < 0.1:
                continue

            pts = [(int(x * CELL_W), int(y * CELL_H)) for x, y in pts_norm]

            if cls_id == 0:
                draw.line(pts + [pts[0]], fill=(80, 80, 80), width=1)
                continue
            elif cls_id == 1:
                draw.line(pts + [pts[0]], fill=(100, 80, 40), width=1)
                continue
            elif cls_id == 2:
                if fs == 0:
                    color = (0, 140, 255)
                    thickness = 3
                    text = f"{conf:.2f}"
                    for i in range(len(pts) - 1):
                        draw.line([pts[i], pts[i + 1]], fill=color, width=thickness)
                else:
                    if fs == 1:
                        color = (220, 20, 20)
                        text = "area"
                        fill_rgba = (220, 20, 20, 50)
                    else:
                        color = (255, 120, 40)
                        text = "ov"
                        fill_rgba = (255, 120, 40, 50)
                    thickness = 6
                    overlay_draw.polygon(pts + [pts[0]], fill=fill_rgba)
                    for i in range(len(pts) - 1):
                        draw.line([pts[i], pts[i + 1]], fill=color, width=thickness)

                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                tx, ty = pts[0]
                draw.rectangle([tx, ty - th - 2, tx + tw + 4, ty], fill=color)
                draw.text((tx + 2, ty - th - 2), text, fill=(0, 0, 0), font=font)

        img_pil = Image.alpha_composite(img_pil.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(img_pil)

        info = f"kept:{n_kept}  removed:{n_removed}"
        bg = (80, 20, 20)
        bbox = draw.textbbox((0, 0), info, font=font_info)
        iw, ih = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([2, 2, iw + 8, ih + 6], fill=bg)
        draw.text((6, 4), info, fill=(255, 255, 255), font=font_info)

        montage.paste(img_pil, (cx, cy))

    leg_h = 110
    leg_img = Image.new("RGB", (GRID_COLS * CELL_W, leg_h), (30, 30, 30))
    draw = ImageDraw.Draw(leg_img)
    try:
        font_sm = ImageFont.truetype("DejaVuSans.ttf", 14)
    except (OSError, AttributeError):
        font_sm = ImageFont.load_default()

    legend = [
        ((0, 140, 255), "kept (blue outline, conf)"),
        ((220, 20, 20), "removed area (red fill + thick outline)"),
        ((255, 120, 40), "removed overlap (orange fill + thick outline)"),
        ((80, 80, 80), "walls/doors (gray, context)"),
    ]
    x = 20
    for color, label in legend:
        draw.rectangle([x, 8, x + 20, 28], fill=color)
        draw.text((x + 26, 9), label, fill=(255, 255, 255), font=font_sm)
        x += 220 if "walls" not in label else 180

    ft = filter_totals
    total = ft.get("area", 0) + ft.get("overlap", 0) + ft.get("kept", 0)
    pct = 100 * (ft.get("area", 0) + ft.get("overlap", 0)) / max(total, 1)
    draw.text((20, 42),
        f"area > {cfg.get('MAX_WINDOW_AREA', '?')}  |  overlap < {cfg.get('MIN_WALL_OVERLAP', '?')}",
        fill=(180, 180, 180), font=font_sm)
    draw.text((20, 64),
        f"total: {total}  kept: {ft.get('kept',0)}  area: {ft.get('area',0)}  overlap: {ft.get('overlap',0)}  ({pct:.1f}% removed)",
        fill=(200, 200, 200), font=font_sm)

    final = Image.new("RGB", (GRID_COLS * CELL_W, GRID_ROWS * CELL_H + leg_h), (30, 30, 30))
    final.paste(montage, (0, 0))
    final.paste(leg_img, (0, GRID_ROWS * CELL_H))
    final.save(str(OUTPUT))
    print(f"  Zapisano: {OUTPUT}  ({final.size})")


if __name__ == "__main__":
    create_montage()
