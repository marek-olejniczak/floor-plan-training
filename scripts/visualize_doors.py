#!/usr/bin/env python3
"""
visualize_doors.py — 10 images × 3 conf thresholds (0.3 / 0.4 / 0.5).
In-memory full pipeline for doors: conf → area → overlap → door_win → door_dup.
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
OUTPUT = Path("/mnt/d/rzuty/trening") / "doors_review.png"

N_IMAGES = 10
CONF_THRESHOLDS = [0.3, 0.4, 0.5]
COLS = len(CONF_THRESHOLDS)
ROWS = N_IMAGES
CELL_W, CELL_H = 640, 640
HEADER_H = 40
LEG_H = 150

MAX_WINDOW_AREA = 0.087
MIN_WALL_OVERLAP = 0.3
DOOR_WIN_IOU = 0.2
DOOR_DUP_COVER = 0.9

CLR = {
    "wall": (80, 80, 80),
    "door_kept": (60, 180, 60),
    "window_kept": (0, 140, 255),
    "filtered_area": (255, 0, 0),
    "filtered_overlap": (255, 120, 40),
    "filtered_by_door": (180, 0, 180),
    "filtered_door_dup": (200, 200, 0),
}


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


def bbox_overlap_ratio(bbox, others):
    wx1, wy1, wx2, wy2 = bbox
    a = max(0, wx2 - wx1) * max(0, wy2 - wy1)
    if a <= 0:
        return 0.0
    inter = 0.0
    for ox1, oy1, ox2, oy2 in others:
        ix1 = max(wx1, ox1)
        iy1 = max(wy1, oy1)
        ix2 = min(wx2, ox2)
        iy2 = min(wy2, oy2)
        inter += max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / a


def bbox_iou(b1, b2):
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def filter_all(lines, conf_thresh, wall_bboxes):
    walls = []
    doors = []
    windows = []

    for line in lines:
        parsed = parse_label_line(line)
        if parsed is None:
            continue
        cls_id, pts, conf = parsed

        if cls_id == 0:
            walls.append((cls_id, pts, conf))
            continue
        if cls_id not in (1, 2):
            continue
        if conf < conf_thresh:
            continue

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        bbox = (cx - w/2, cy - h/2, cx + w/2, cy + h/2)
        rec = (cls_id, pts, conf, cx, cy, w, h, bbox, 0, 0.0)

        if cls_id == 1:
            doors.append(list(rec))
        else:
            windows.append(list(rec))

    # area + overlap dla windows
    for i, rec in enumerate(windows):
        area = rec[5] * rec[6]
        fs = 0
        extra = 0.0
        if area > MAX_WINDOW_AREA:
            fs = 1
        else:
            extra = bbox_overlap_ratio(rec[7], wall_bboxes)
            if extra < MIN_WALL_OVERLAP:
                fs = 2
        windows[i][8] = fs
        windows[i][9] = extra

    # area dla doors
    for i, rec in enumerate(doors):
        area = rec[5] * rec[6]
        if area > MAX_WINDOW_AREA:
            doors[i][8] = 1

    # door-win overlap
    door_bboxes_kept = [d[7] for d in doors if d[8] == 0]
    for i, rec in enumerate(windows):
        if rec[8] != 0:
            continue
        for db in door_bboxes_kept:
            iou = bbox_iou(rec[7], db)
            if iou > DOOR_WIN_IOU:
                windows[i][8] = 3
                windows[i][9] = iou
                break

    # door dedup
    for i in range(len(doors)):
        if doors[i][8] != 0:
            continue
        bi, ai = doors[i][7], doors[i][5] * doors[i][6]
        for j in range(i + 1, len(doors)):
            if doors[j][8] != 0:
                continue
            bj, aj = doors[j][7], doors[j][5] * doors[j][6]
            ix1 = max(bi[0], bj[0])
            iy1 = max(bi[1], bj[1])
            ix2 = min(bi[2], bj[2])
            iy2 = min(bi[3], bj[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            ci = inter / ai if ai > 0 else 0
            cj = inter / aj if aj > 0 else 0
            if ci >= DOOR_DUP_COVER or cj >= DOOR_DUP_COVER:
                larger = j if ai < aj else i
                doors[larger][8] = 4
                doors[larger][9] = min(ai, aj) / max(ai, aj)

    return walls, doors, windows


def draw_cell(img_pil, walls, doors, windows, font, font_info):
    overlay = Image.new("RGBA", (CELL_W, CELL_H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(img_pil)

    for cls_id, pts, conf in walls:
        pts_ = [(int(x * CELL_W), int(y * CELL_H)) for x, y in pts]
        draw.line(pts_ + [pts_[0]], fill=CLR["wall"], width=1)

    n_door_kept = n_door_dup = n_win_kept = n_area = n_ov = n_door_win = 0

    for rec_list, kept_label, color_key in [
        (doors, "door", "door_kept"),
        (windows, "window", "window_kept"),
    ]:
        for rec in rec_list:
            cls_id, pts, conf, _, _, _, _, _, fs, extra = rec
            if conf < 0.1:
                continue
            pts_ = [(int(x * CELL_W), int(y * CELL_H)) for x, y in pts]

            if fs == 0:
                if cls_id == 1:
                    n_door_kept += 1
                    color = CLR["door_kept"]
                    text = f"{conf:.2f}"
                else:
                    n_win_kept += 1
                    color = CLR["window_kept"]
                    text = f"{conf:.2f}"
                thickness = 3
                for i in range(len(pts_) - 1):
                    draw.line([pts_[i], pts_[i + 1]], fill=color, width=thickness)
            else:
                if fs == 1:
                    n_area += 1
                    color = CLR["filtered_area"]
                    text = "area"
                elif fs == 2:
                    n_ov += 1
                    color = CLR["filtered_overlap"]
                    text = "ov"
                elif fs == 3:
                    n_door_win += 1
                    color = CLR["filtered_by_door"]
                    text = "door"
                else:
                    n_door_dup += 1
                    color = CLR["filtered_door_dup"]
                    text = "dup"
                thickness = 5
                od.polygon(pts_ + [pts_[0]], fill=(color[0], color[1], color[2], 70))
                for i in range(len(pts_) - 1):
                    draw.line([pts_[i], pts_[i + 1]], fill=color, width=thickness)

            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx, ty = pts_[0]
            draw.rectangle([tx, ty - th - 2, tx + tw + 4, ty], fill=color)
            draw.text((tx + 2, ty - th - 2), text, fill=(0, 0, 0), font=font)

    img_pil = Image.alpha_composite(img_pil.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img_pil)

    info = f"d:{n_door_kept}/{n_door_dup} w:{n_win_kept} a:{n_area} o:{n_ov} dw:{n_door_win}"
    bbox = draw.textbbox((0, 0), info, font=font_info)
    iw, ih = bbox[2] - bbox[0], bbox[3] - bbox[1]
    any_f = n_area + n_ov + n_door_win + n_door_dup > 0
    draw.rectangle([2, 2, iw + 8, ih + 6], fill=(80, 20, 20) if any_f else (20, 40, 20))
    draw.text((6, 4), info, fill=(255, 255, 255), font=font_info)

    return img_pil


def create_montage():
    if not DECISIONS_PATH.exists():
        print(f"[BLAD] Brak {DECISIONS_PATH}")
        return
    with open(DECISIONS_PATH) as f:
        meta = json.load(f)

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
            has_mid = False
            for line in lines:
                parsed = parse_label_line(line)
                if parsed and parsed[0] == 1:
                    conf = parsed[2]
                    if 0.3 <= conf < 0.5:
                        has_mid = True
                        break
            if has_mid:
                candidates.append((img_path, lbl_path, lines, wall_bboxes))

    if not candidates:
        print("[BLAD] Brak obrazow!")
        return
    random.shuffle(candidates)
    selected = candidates[:N_IMAGES]
    print(f"  Wybrano {len(selected)} obrazow (drzwi z conf w [0.3, 0.5))")

    total_w = COLS * CELL_W
    total_h = HEADER_H + ROWS * CELL_H + LEG_H
    montage = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    draw = ImageDraw.Draw(montage)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
        font_info = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
        font_header = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        font_sm = ImageFont.truetype("DejaVuSans.ttf", 14)
    except:
        font = font_info = font_header = font_sm = ImageFont.load_default()

    for ci, th in enumerate(CONF_THRESHOLDS):
        hx = ci * CELL_W + CELL_W // 2
        text = f"conf >= {th}"
        bbox = draw.textbbox((0, 0), text, font=font_header)
        draw.text((hx - (bbox[2]-bbox[0])//2, 8), text, fill=(200, 200, 200), font=font_header)

    for ri, (img_path, lbl_path, lines, wall_bboxes) in enumerate(selected):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        for ci, th in enumerate(CONF_THRESHOLDS):
            cx = ci * CELL_W
            cy = HEADER_H + ri * CELL_H
            img_pil = Image.fromarray(img_rgb.copy()).resize((CELL_W, CELL_H), Image.LANCZOS)
            walls, doors, windows = filter_all(lines, th, wall_bboxes)
            img_pil = draw_cell(img_pil, walls, doors, windows, font, font_info)
            montage.paste(img_pil, (cx, cy))

    ly = HEADER_H + ROWS * CELL_H
    leg = Image.new("RGB", (total_w, LEG_H), (30, 30, 30))
    draw = ImageDraw.Draw(leg)

    legend = [
        (CLR["door_kept"], "door kept (green)"),
        (CLR["window_kept"], "window kept (blue)"),
        (CLR["filtered_area"], "filtered area"),
        (CLR["filtered_overlap"], "filtered overlap"),
        (CLR["filtered_by_door"], "filtered by door (win)"),
        (CLR["filtered_door_dup"], "filtered door dup"),
        (CLR["wall"], "wall (gray)"),
    ]
    x = 15
    for color, label in legend:
        draw.rectangle([x, 6, x + 20, 24], fill=color)
        draw.text((x + 25, 6), label, fill=(255, 255, 255), font=font_sm)
        x += 160

    cfg = meta.get("config", {})
    draw.text((15, 38),
        f"area:P90={MAX_WINDOW_AREA}  overlap:<{MIN_WALL_OVERLAP}  door_iou>{DOOR_WIN_IOU}  door_dup>{DOOR_DUP_COVER}",
        fill=(180, 180, 180), font=font_sm)
    draw.text((15, 58),
        "green = door kept  |  blue = window kept  |  dup = duplicate door removed",
        fill=(160, 160, 160), font=font_sm)
    draw.text((15, 78),
        "Each row = same image  |  Each column = different conf threshold  |  All filters applied in-memory",
        fill=(160, 160, 160), font=font_sm)
    draw.text((15, 98),
        "Info bar: d:kept/dup w:kept a:area o:overlap dw:by_door",
        fill=(160, 160, 160), font=font_sm)

    montage.paste(leg, (0, ly))
    montage.save(str(OUTPUT))
    print(f"  Zapisano: {OUTPUT} ({montage.size})")


if __name__ == "__main__":
    create_montage()
