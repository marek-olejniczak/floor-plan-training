#!/usr/bin/env python3
"""
filter_predictions.py — Filters raw predictions.
Stages: conf → area → wall_overlap → door_win_overlap → door_dedup.
"""

import os, sys, json
from pathlib import Path
from collections import Counter

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

SRC = Path.home() / "data" / "raw_predictions"
DST = Path.home() / "data" / "corrected_walls"
DECISIONS_PATH = DST / "filter_decisions.json"

MIN_CONF = 0.4
MAX_WINDOW_AREA = 0.087
MIN_WALL_OVERLAP = 0.3
DOOR_WIN_IOU = 0.2
DOOR_DUP_COVER = 0.9

SPLITS = ["train", "valid", "test"]

KEPT = 0
FILTERED_AREA = 1
FILTERED_OVERLAP = 2
FILTERED_BY_DOOR = 3
FILTERED_DOOR_DUP = 4
STATUS_LABELS = {
    0: "kept", 1: "filtered_area", 2: "filtered_overlap",
    3: "filtered_by_door", 4: "filtered_door_dup",
}


def parse_label_line(line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls_id = int(parts[0])
    coords = list(map(float, parts[1:]))
    if len(coords) >= 10:
        pts = [(coords[i], coords[i + 1]) for i in range(0, 10, 2)]
        return cls_id, pts, 0.0, 0
    conf = 0.0
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
    return cls_id, pts, conf, 0


def bbox_overlap_ratio(bbox, other_bboxes):
    wx1, wy1, wx2, wy2 = bbox
    win_area = max(0, wx2 - wx1) * max(0, wy2 - wy1)
    if win_area <= 0:
        return 0.0
    inter = 0.0
    for ox1, oy1, ox2, oy2 in other_bboxes:
        ix1 = max(wx1, ox1)
        iy1 = max(wy1, oy1)
        ix2 = min(wx2, ox2)
        iy2 = min(wy2, oy2)
        inter += max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / win_area


def bbox_iou(b1, b2):
    x1, y1, x2, y2 = b1
    u1, v1, u2, v2 = b2
    ix1 = max(x1, u1)
    iy1 = max(y1, v1)
    ix2 = min(x2, u2)
    iy2 = min(y2, v2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = max(0, x2 - x1) * max(0, y2 - y1)
    a2 = max(0, u2 - u1) * max(0, v2 - v1)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def format_label(cls_id, cx, cy, w, h, conf=0.0, filter_status=0, extra=0.0):
    return f"{cls_id} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f} {conf:.4f} {filter_status} {extra:.4f}"


def filter_dataset():
    totals = Counter()
    filter_totals = Counter()
    decisions = {}

    for split in SPLITS:
        img_dir = SRC / split / "images"
        lbl_dir = SRC / split / "labels"
        if not img_dir.is_dir():
            continue

        dst_img_dir = DST / split / "images"
        dst_lbl_dir = DST / split / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_paths = sorted(img_dir.iterdir())
        print(f"{split}: {len(img_paths)} obrazow")

        for img_path in img_paths:
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            lbl_path = lbl_dir / (img_path.stem + ".txt")
            dst_img = dst_img_dir / img_path.name
            dst_lbl = dst_lbl_dir / (img_path.stem + ".txt")

            if not dst_img.exists():
                os.link(str(img_path), str(dst_img))

            lines = []
            if lbl_path.exists():
                with open(lbl_path) as f:
                    lines = [l.strip() for l in f if l.strip()]

            # Zbierz sciany
            wall_bboxes = []
            wall_out_lines = []
            # Zbierz drzwi/okna (przetrwaja conf filter)
            doors = []   # (cls, pts, conf, cx, cy, w, h, bbox_norm)
            windows = []
            image_decisions = []

            for line in lines:
                parsed = parse_label_line(line)
                if parsed is None:
                    continue
                cls_id, pts, conf, _ = parsed

                if cls_id == 0:
                    wall_bboxes.append((
                        min(p[0] for p in pts),
                        min(p[1] for p in pts),
                        max(p[0] for p in pts),
                        max(p[1] for p in pts),
                    ))
                    wall_out_lines.append(line)
                    totals["walls"] += 1
                    continue

                if cls_id not in (1, 2):
                    continue

                if conf < MIN_CONF:
                    filter_totals["low_conf"] += 1
                    continue

                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                cx = (min(xs) + max(xs)) / 2
                cy = (min(ys) + max(ys)) / 2
                w_b = max(xs) - min(xs)
                h_b = max(ys) - min(ys)
                bbox = (cx - w_b/2, cy - h_b/2, cx + w_b/2, cy + h_b/2)
                rec = (cls_id, pts, conf, cx, cy, w_b, h_b, bbox)

                if cls_id == 1:
                    doors.append(rec)
                else:
                    windows.append(rec)

            # ---- Filtry ----
            # 1. window area
            for i, (cls_id, pts, conf, cx, cy, w_b, h_b, bbox) in enumerate(windows):
                area = w_b * h_b
                fs = KEPT
                extra = 0.0
                if area > MAX_WINDOW_AREA:
                    fs = FILTERED_AREA
                    filter_totals["area"] += 1
                else:
                    extra = bbox_overlap_ratio(bbox, wall_bboxes)
                    if extra < MIN_WALL_OVERLAP:
                        fs = FILTERED_OVERLAP
                        filter_totals["overlap"] += 1
                windows[i] = (cls_id, pts, conf, cx, cy, w_b, h_b, bbox, fs, extra)

            # 2. wall overlap dla doors — tylko kept/area
            for i, (cls_id, pts, conf, cx, cy, w_b, h_b, bbox) in enumerate(doors):
                fs = KEPT
                extra = 0.0
                area = w_b * h_b
                if area > MAX_WINDOW_AREA:
                    fs = FILTERED_AREA
                    filter_totals["area"] += 1
                doors[i] = (cls_id, pts, conf, cx, cy, w_b, h_b, bbox, fs, extra)

            # 3. door-window overlap: window filtered by door
            door_bboxes_kept = [d[7] for d in doors if d[8] == KEPT]
            for i, wrec in enumerate(windows):
                if wrec[8] != KEPT:
                    continue
                win_bbox = wrec[7]
                for db in door_bboxes_kept:
                    iou = bbox_iou(win_bbox, db)
                    if iou > DOOR_WIN_IOU:
                        windows[i] = (*wrec[:8], FILTERED_BY_DOOR, iou)
                        filter_totals["door_win"] += 1
                        break

            # 4. door dedup
            for i in range(len(doors)):
                if doors[i][8] != KEPT:
                    continue
                bi = doors[i][7]
                ai = doors[i][5] * doors[i][6]
                for j in range(i + 1, len(doors)):
                    if doors[j][8] != KEPT:
                        continue
                    bj = doors[j][7]
                    aj = doors[j][5] * doors[j][6]
                    ix1 = max(bi[0], bj[0])
                    iy1 = max(bi[1], bj[1])
                    ix2 = min(bi[2], bj[2])
                    iy2 = min(bi[3], bj[3])
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    cover_i = inter / ai if ai > 0 else 0
                    cover_j = inter / aj if aj > 0 else 0
                    if cover_i >= DOOR_DUP_COVER or cover_j >= DOOR_DUP_COVER:
                        smaller = i if ai < aj else j
                        larger = j if ai < aj else i
                        doors[larger] = (*doors[larger][:8], FILTERED_DOOR_DUP, min(ai, aj) / max(ai, aj))
                        filter_totals["door_dup"] += 1

            # ---- Zapis ----
            out_lines = list(wall_out_lines)
            for rec in doors:
                cls_id, pts, conf, cx, cy, w_b, h_b, bbox, fs, extra = rec
                out_lines.append(format_label(cls_id, cx, cy, w_b, h_b, conf, fs, extra))
                totals["doors"] += 1
                if fs == KEPT:
                    filter_totals["kept"] += 1
                image_decisions.append({
                    "cls": cls_id, "conf": conf,
                    "filter_status": fs, "extra": round(extra, 4),
                    "reason": STATUS_LABELS.get(fs, "unknown"),
                })
            for rec in windows:
                cls_id, pts, conf, cx, cy, w_b, h_b, bbox, fs, extra = rec
                out_lines.append(format_label(cls_id, cx, cy, w_b, h_b, conf, fs, extra))
                totals["windows"] += 1
                if fs == KEPT:
                    filter_totals["kept"] += 1
                image_decisions.append({
                    "cls": cls_id, "conf": conf,
                    "filter_status": fs, "extra": round(extra, 4),
                    "reason": STATUS_LABELS.get(fs, "unknown"),
                })

            dst_lbl.write_text("\n".join(out_lines) + "\n")
            decisions[img_path.stem] = image_decisions

    # ---- PODSUMOWANIE ----
    total_objects = totals.get("doors", 0) + totals.get("windows", 0)
    print(f"\n{'='*55}")
    print(f"  FILTRACJA ZAKONCZONA")
    print(f"{'='*55}")
    print(f"  MIN_CONF          = {MIN_CONF}")
    print(f"  MAX_WINDOW_AREA   = {MAX_WINDOW_AREA}")
    print(f"  MIN_WALL_OVERLAP  = {MIN_WALL_OVERLAP}")
    print(f"  DOOR_WIN_IOU      = {DOOR_WIN_IOU}")
    print(f"  DOOR_DUP_COVER    = {DOOR_DUP_COVER}")
    print(f"  Sciany:   {totals.get('walls', 0)}")
    print(f"  Drzwi:    {totals.get('doors', 0)}")
    print(f"  Okna:     {totals.get('windows', 0)}")
    print(f"  --- Filtry ---")
    print(f"  Low conf (<{MIN_CONF}): {filter_totals.get('low_conf', 0)}")
    print(f"  Area:     {filter_totals.get('area', 0)}")
    print(f"  Overlap:  {filter_totals.get('overlap', 0)}")
    print(f"  By door:  {filter_totals.get('door_win', 0)}")
    print(f"  Door dup: {filter_totals.get('door_dup', 0)}")
    print(f"  Kept:     {filter_totals.get('kept', 0)}")
    total_filtered = sum(filter_totals.get(k, 0) for k in ("area", "overlap", "door_win", "door_dup"))
    if total_objects > 0:
        print(f"  Razem odrzucone: {total_filtered} ({100*total_filtered/(total_filtered+filter_totals.get('kept',1)):.1f}%)")
    print(f"  Output:   {DST}")
    print(f"{'='*55}")

    with open(DECISIONS_PATH, "w") as f:
        json.dump({
            "config": {
                "MIN_CONF": MIN_CONF,
                "MAX_WINDOW_AREA": MAX_WINDOW_AREA,
                "MIN_WALL_OVERLAP": MIN_WALL_OVERLAP,
                "DOOR_WIN_IOU": DOOR_WIN_IOU,
                "DOOR_DUP_COVER": DOOR_DUP_COVER,
            },
            "totals": dict(totals),
            "filter_totals": dict(filter_totals),
            "decisions": decisions,
        }, f, indent=1)
    print(f"  Decyzje: {DECISIONS_PATH}")


if __name__ == "__main__":
    filter_dataset()
