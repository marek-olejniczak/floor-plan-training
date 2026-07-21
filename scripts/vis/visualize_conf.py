#!/usr/bin/env python3
"""
visualize_conf.py — 10 images × 3 confidence thresholds (0.3 / 0.4 / 0.5).
Area=P90 fixed, overlap=0.3 fixed. Vary min confidence.
"""

import os, sys, random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

RAW = Path.home() / "data" / "raw_predictions"         # obrazy
FILTERED = Path.home() / "data" / "corrected_walls"    # labelki z conf
OUTPUT = Path("/mnt/d/rzuty/trening") / "conf_review.png"

N_IMAGES = 10
MAX_WINDOW_AREA = 0.086897
MIN_WALL_OVERLAP = 0.3
CONF_THRESHOLDS = [0.3, 0.4, 0.5]
COLS = len(CONF_THRESHOLDS)
ROWS = N_IMAGES
CELL_W, CELL_H = 640, 640

HEADER_H = 40
LEG_H = 130


def parse_label_line(line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls_id = int(parts[0])
    coords = list(map(float, parts[1:]))
    conf = 0.0
    if len(coords) >= 10:
        pts = [(coords[i], coords[i + 1]) for i in range(0, 10, 2)]
        return cls_id, pts, conf
    if len(coords) >= 5:
        conf = coords[4]
    cx, cy, w, h = coords[:4]
    pts = [
        (cx - w / 2, cy - h / 2),
        (cx + w / 2, cy - h / 2),
        (cx + w / 2, cy + h / 2),
        (cx - w / 2, cy + h / 2),
        (cx - w / 2, cy - h / 2),
    ]
    return cls_id, pts, conf


def bbox_overlap_ratio(win_bbox, wall_bboxes):
    wx1, wy1, wx2, wy2 = win_bbox
    win_area = max(0, (wx2 - wx1)) * max(0, (wy2 - wy1))
    if win_area <= 0:
        return 0.0
    inter = 0.0
    for wax1, way1, wax2, way2 in wall_bboxes:
        ix1 = max(wx1, wax1)
        iy1 = max(wy1, way1)
        ix2 = min(wx2, wax2)
        iy2 = min(wy2, way2)
        inter += max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / win_area


def filter_windows(lines, conf_thresh, wall_bboxes):
    results = []
    for line in lines:
        parsed = parse_label_line(line)
        if parsed is None:
            continue
        cls_id, pts, conf = parsed
        if cls_id == 2:
            if conf < conf_thresh:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            w_b = max(xs) - min(xs)
            h_b = max(ys) - min(ys)
            area = w_b * h_b
            new_fs = 0
            if area > MAX_WINDOW_AREA:
                new_fs = 1
            else:
                cx = (min(xs) + max(xs)) / 2
                cy = (min(ys) + max(ys)) / 2
                win_bbox = (cx - w_b/2, cy - h_b/2, cx + w_b/2, cy + h_b/2)
                overlap = bbox_overlap_ratio(win_bbox, wall_bboxes)
                if overlap < MIN_WALL_OVERLAP:
                    new_fs = 2
            results.append((cls_id, pts, conf, new_fs))
        else:
            results.append((cls_id, pts, conf, 0))
    return results


def draw_cell(img_pil, results, font, font_info):
    overlay = Image.new("RGBA", (CELL_W, CELL_H), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(img_pil)

    n_kept = n_area = n_overlap = 0
    for cls_id, pts_norm, conf, fs in results:
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
                n_kept += 1
                color = (0, 140, 255)
                text = f"{conf:.2f}"
                for i in range(len(pts) - 1):
                    draw.line([pts[i], pts[i + 1]], fill=color, width=3)
            else:
                if fs == 1:
                    n_area += 1
                    color = (255, 0, 0)
                    text = "AREA"
                    fill_rgba = (255, 0, 0, 80)
                else:
                    n_overlap += 1
                    color = (255, 120, 40)
                    text = "ov"
                    fill_rgba = (255, 120, 40, 80)
                overlay_draw.polygon(pts + [pts[0]], fill=fill_rgba)
                for i in range(len(pts) - 1):
                    draw.line([pts[i], pts[i + 1]], fill=color, width=6)

            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx, ty = pts[0]
            draw.rectangle([tx, ty - th - 2, tx + tw + 4, ty], fill=color)
            draw.text((tx + 2, ty - th - 2), text, fill=(0, 0, 0), font=font)

    img_pil = Image.alpha_composite(img_pil.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img_pil)

    info = f"kept:{n_kept}  area:{n_area}  ov:{n_overlap}"
    bbox = draw.textbbox((0, 0), info, font=font_info)
    iw, ih = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([2, 2, iw + 8, ih + 6], fill=(80, 20, 20) if n_area + n_overlap > 0 else (20, 40, 20))
    draw.text((6, 4), info, fill=(255, 255, 255), font=font_info)

    return img_pil


def create_montage():
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
            with open(lbl_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            wall_bboxes = []
            for line in lines:
                parsed = parse_label_line(line)
                if parsed and parsed[0] == 0:
                    xs = [p[0] for p in parsed[1]]
                    ys = [p[1] for p in parsed[1]]
                    wall_bboxes.append((min(xs), min(ys), max(xs), max(ys)))
            # Szukaj: box z conf w [0.3, 0.5) + filtered przy conf=0.3
            has_mid = False
            has_filtered_any = False
            for line in lines:
                parsed = parse_label_line(line)
                if parsed and parsed[0] == 2:
                    conf = parsed[2]
                    if conf < 0.3:
                        continue
                    if conf < 0.5:
                        has_mid = True
                    pts = parsed[1]
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    w_b = max(xs) - min(xs)
                    h_b = max(ys) - min(ys)
                    if w_b * h_b > MAX_WINDOW_AREA:
                        has_filtered_any = True
                    else:
                        cx = (min(xs) + max(xs)) / 2
                        cy = (min(ys) + max(ys)) / 2
                        win_bbox = (cx - w_b/2, cy - h_b/2, cx + w_b/2, cy + h_b/2)
                        if bbox_overlap_ratio(win_bbox, wall_bboxes) < MIN_WALL_OVERLAP:
                            has_filtered_any = True
            if has_mid and has_filtered_any:
                candidates.append((img_path, lbl_path, lines, wall_bboxes))

    if not candidates:
        print("[BLAD] Brak obrazow!")
        return

    random.shuffle(candidates)
    selected = candidates[:N_IMAGES]
    print(f"  Wybrano {len(selected)} obrazow (filtered>0 przy conf=0.3)")

    total_w = COLS * CELL_W
    total_h = HEADER_H + ROWS * CELL_H + LEG_H
    montage = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    draw = ImageDraw.Draw(montage)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        font_info = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        font_header = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        font_sm = ImageFont.truetype("DejaVuSans.ttf", 14)
    except (OSError, AttributeError):
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
            font_info = ImageFont.truetype("DejaVuSans.ttf", 16)
            font_header = ImageFont.truetype("DejaVuSans.ttf", 18)
            font_sm = font
        except:
            font = font_info = font_header = font_sm = ImageFont.load_default()

    for ci, thresh in enumerate(CONF_THRESHOLDS):
        hx = ci * CELL_W + CELL_W // 2
        text = f"conf >= {thresh}"
        bbox = draw.textbbox((0, 0), text, font=font_header)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((hx - tw // 2, 8), text, fill=(200, 200, 200), font=font_header)

    for ri, (img_path, lbl_path, lines, wall_bboxes) in enumerate(selected):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        for ci, thresh in enumerate(CONF_THRESHOLDS):
            cx = ci * CELL_W
            cy = HEADER_H + ri * CELL_H
            img_pil = Image.fromarray(img_rgb.copy()).resize((CELL_W, CELL_H), Image.LANCZOS)
            results = filter_windows(lines, thresh, wall_bboxes)
            img_pil = draw_cell(img_pil, results, font, font_info)
            montage.paste(img_pil, (cx, cy))

    ly = HEADER_H + ROWS * CELL_H
    leg = Image.new("RGB", (total_w, LEG_H), (30, 30, 30))
    draw = ImageDraw.Draw(leg)

    legend_items = [
        ((0, 140, 255), "kept (blue, conf)"),
        ((255, 0, 0), "removed AREA (red fill)"),
        ((255, 120, 40), "removed overlap (orange fill)"),
        ((80, 80, 80), "walls/doors (gray context)"),
    ]
    x = 20
    for color, label in legend_items:
        draw.rectangle([x, 8, x + 22, 28], fill=color)
        draw.text((x + 28, 9), label, fill=(255, 255, 255), font=font_sm)
        x += 200

    draw.text((20, 42),
        f"area: P90={MAX_WINDOW_AREA:.4f}  |  overlap: < {MIN_WALL_OVERLAP}  |  conf: 0.3 / 0.4 / 0.5",
        fill=(180, 180, 180), font=font_sm)
    draw.text((20, 64),
        "Each row = same image  |  Each column = different conf threshold",
        fill=(160, 160, 160), font=font_sm)

    montage.paste(leg, (0, ly))
    montage.save(str(OUTPUT))
    print(f"  Zapisano: {OUTPUT}  ({montage.size})")


if __name__ == "__main__":
    create_montage()
