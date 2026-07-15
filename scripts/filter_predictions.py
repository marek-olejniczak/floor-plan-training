#!/usr/bin/env python3
"""
filter_predictions.py — Separate filtering step on saved raw predictions.
Reads ~/data/raw_predictions/, applies area+overlap filters,
writes to ~/data/corrected_walls/ with updated filter_status (7th field).
Also writes filter_decisions.json for visualization.
"""

import os, sys, json
from pathlib import Path
from collections import Counter

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

SRC = Path.home() / "data" / "raw_predictions"
DST = Path.home() / "data" / "corrected_walls"
DECISIONS_PATH = DST / "filter_decisions.json"

MAX_WINDOW_AREA = 0.087   # P90 z analyze_window_area.py
MIN_WALL_OVERLAP = 0.3

SPLITS = ["train", "valid", "test"]

# filter_status values
KEPT = 0
FILTERED_AREA = 1
FILTERED_OVERLAP = 2
STATUS_LABELS = {0: "kept", 1: "filtered_area", 2: "filtered_overlap"}


def parse_label_line(line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None, None
    cls_id = int(parts[0])
    coords = list(map(float, parts[1:]))
    # Polygon (cls x1 y1 ... x5 y5) — 11 pol
    if len(coords) >= 10:
        pts = [(coords[i], coords[i + 1]) for i in range(0, 10, 2)]
        return cls_id, pts, 0.0, 0
    # YOLO z conf+fs (7 pol: cls cx cy w h conf fs) lub bez (5 pol)
    conf = 0.0
    filter_status = 0
    if len(coords) >= 5:
        conf = coords[4]
    if len(coords) >= 6:
        filter_status = int(round(coords[5]))
    cx, cy, w, h = coords[:4]
    pts = [
        (cx - w / 2, cy - h / 2),
        (cx + w / 2, cy - h / 2),
        (cx + w / 2, cy + h / 2),
        (cx - w / 2, cy + h / 2),
        (cx - w / 2, cy - h / 2),
    ]
    return cls_id, pts, conf, filter_status


def bbox_overlap_ratio(win_bbox, wall_bboxes):
    wx1, wy1, wx2, wy2 = win_bbox
    win_area = max(0, (wx2 - wx1)) * max(0, (wy2 - wy1))
    if win_area <= 0:
        return 0.0
    intersection_area = 0.0
    for wax1, way1, wax2, way2 in wall_bboxes:
        ix1 = max(wx1, wax1)
        iy1 = max(wy1, way1)
        ix2 = min(wx2, wax2)
        iy2 = min(wy2, way2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        intersection_area += inter
    return intersection_area / win_area


def format_label(cls_id, cx, cy, w, h, conf=0.0, filter_status=0, overlap=0.0):
    return f"{cls_id} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f} {conf:.4f} {filter_status} {overlap:.4f}"


def filter_dataset():
    totals = Counter()
    filter_totals = Counter()
    decisions = {}  # image_name -> list of {cls, conf, area, overlap, filter_status}

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

            # Kopiuj obraz (hardlink jesli mozliwe, symlink nie dziala z YOLO)
            if not dst_img.exists():
                os.link(str(img_path), str(dst_img))

            # Wczytaj labelki
            lines = []
            if lbl_path.exists():
                with open(lbl_path) as f:
                    lines = [l.strip() for l in f if l.strip()]

            # Zbierz sciany (w znormalizowanych koordynatach)
            wall_bboxes = []
            for line in lines:
                parsed = parse_label_line(line)
                if parsed is None:
                    continue
                cls_id, pts, conf, fs = parsed
                if cls_id == 0:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    wall_bboxes.append((min(xs), min(ys), max(xs), max(ys)))

            # Filtruj
            out_lines = []
            image_decisions = []

            for line in lines:
                parsed = parse_label_line(line)
                if parsed is None:
                    continue
                cls_id, pts, conf, fs = parsed

                if cls_id == 0:
                    out_lines.append(line)
                    totals["walls"] += 1
                    continue

                # Dla drzwi/okien: wyciągnij cx,cy,w,h z pts (4 narożników YOLO)
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                cx = (min(xs) + max(xs)) / 2
                cy = (min(ys) + max(ys)) / 2
                w_b = max(xs) - min(xs)
                h_b = max(ys) - min(ys)

                new_fs = fs  # zachowaj oryginalny status

                overlap_val = 0.0
                if cls_id == 2:  # window — filtruj
                    area = w_b * h_b
                    area_filtered = area > MAX_WINDOW_AREA

                    if area_filtered:
                        new_fs = FILTERED_AREA
                        filter_totals["area"] += 1
                    else:
                        win_bbox_norm = (cx - w_b / 2, cy - h_b / 2, cx + w_b / 2, cy + h_b / 2)
                        overlap_val = bbox_overlap_ratio(win_bbox_norm, wall_bboxes)
                        if overlap_val < MIN_WALL_OVERLAP:
                            new_fs = FILTERED_OVERLAP
                            filter_totals["overlap"] += 1
                        else:
                            filter_totals["kept"] += 1

                    image_decisions.append({
                        "cls": cls_id, "conf": conf, "area": round(area, 6),
                        "overlap": round(overlap_val, 4) if not area_filtered else None,
                        "filter_status": new_fs,
                        "reason": STATUS_LABELS.get(new_fs, "unknown"),
                    })
                else:
                    filter_totals["kept"] += 1

                out_lines.append(format_label(cls_id, cx, cy, w_b, h_b, conf, new_fs, overlap_val))
                totals["doors" if cls_id == 1 else "windows"] += 1

            dst_lbl.write_text("\n".join(out_lines) + "\n")
            decisions[img_path.stem] = image_decisions

    # ---- PODSUMOWANIE ----
    total_windows = totals.get("windows", 0)
    print(f"\n{'='*55}")
    print(f"  FILTRACJA ZAKONCZONA")
    print(f"{'='*55}")
    print(f"  MAX_WINDOW_AREA  = {MAX_WINDOW_AREA}")
    print(f"  MIN_WALL_OVERLAP = {MIN_WALL_OVERLAP}")
    print(f"  Sciany:   {totals.get('walls', 0)}")
    print(f"  Drzwi:    {totals.get('doors', 0)}")
    print(f"  Okna RAW: {total_windows}")
    print(f"  --- Filtry okien ---")
    print(f"  Odrzucone area:   {filter_totals.get('area', 0)}")
    print(f"  Odrzucone overlap:{filter_totals.get('overlap', 0)}")
    print(f"  Zachowane:        {filter_totals.get('kept', 0)}")
    total_filtered = filter_totals.get('area', 0) + filter_totals.get('overlap', 0)
    if total_windows > 0:
        print(f"  Odrzucone:        {total_filtered} ({100*total_filtered/total_windows:.1f}%)")
    print(f"  Output:           {DST}")
    print(f"{'='*55}")

    # Zapisz decyzje do JSON
    with open(DECISIONS_PATH, "w") as f:
        json.dump({
            "config": {
                "MAX_WINDOW_AREA": MAX_WINDOW_AREA,
                "MIN_WALL_OVERLAP": MIN_WALL_OVERLAP,
            },
            "totals": dict(totals),
            "filter_totals": dict(filter_totals),
            "decisions": decisions,
        }, f, indent=1)
    print(f"  Decyzje: {DECISIONS_PATH}")


if __name__ == "__main__":
    filter_dataset()
