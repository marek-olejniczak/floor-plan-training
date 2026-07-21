#!/usr/bin/env python3
"""filter_d1_windows.py — filter oversized windows in d1 by area percentile/threshold."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

D1 = Path("/mnt/d/rzuty/dane/yolo11datasets/walls_doors_windows/d1")
OUT = Path("/mnt/d/rzuty/dane/yolo11datasets/walls_doors_windows/d1_filtered")
OUT_PNG = Path("/mnt/d/rzuty/trening")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
CELL = 300  # px per image cell
COLORS = {0: (0, 100, 255), 1: (0, 200, 0), 2: (255, 150, 0)}
LABEL_NAMES = {0: "door", 1: "wall", 2: "window"}


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--percentile", type=float, default=None,
                   help="Percentile threshold (e.g. 97)")
    g.add_argument("--threshold", type=float, default=None,
                   help="Direct area threshold (w*h)")
    return p.parse_args()


def load_areas():
    areas = []
    for split in ("train", "valid", "test"):
        lp = D1 / split / "labels"
        for f in lp.glob("*.txt"):
            with open(f) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[0] == "2":
                        areas.append(float(parts[3]) * float(parts[4]))
    return np.array(areas)


def filter_labels(src_label, dst_label, threshold):
    removed = 0
    with open(src_label) as fin, open(dst_label, "w") as fout:
        for line in fin:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] == "2":
                a = float(parts[3]) * float(parts[4])
                if a > threshold:
                    removed += 1
                    continue
            fout.write(line)


def count_windows(path):
    n = 0
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] == "2":
                n += 1
    return n


def draw_bbox(dw, parts, ox, oy, cw, ch, highlight_removed=False):
    cid = int(parts[0])
    if cid > 2:
        return
    cx, cy, w, h = map(float, parts[1:5])
    x1 = int((cx - w / 2) * cw + ox)
    y1 = int((cy - h / 2) * ch + oy)
    x2 = int((cx + w / 2) * cw + ox)
    y2 = int((cy + h / 2) * ch + oy)

    if highlight_removed:
        color = (255, 0, 0)
        width = 3
    else:
        color = COLORS.get(cid, (200, 200, 200))
        width = 2
    dw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def make_visualization(examples, threshold, label):
    """examples: list of (stem, n_removed, max_area, areas, orig_lines, filt_lines) sorted by max_area ascending"""
    n = len(examples)
    rows = n
    img_w = CELL * 2 + 20  # before + after + gap
    img_h = rows * (CELL + 48) + 60  # +48 per row for text, +60 top header

    canvas = Image.new("RGB", (img_w, img_h), (30, 30, 30))
    dw = ImageDraw.Draw(canvas)

    fnt = ImageFont.truetype(FONT, 20) if Path(FONT).exists() else None
    sf = ImageFont.truetype(FONT, 13) if fnt else None
    tf = ImageFont.truetype(FONT, 12) if fnt else None

    dw.text((10, 8),
            f"d1 window filter — threshold={threshold:.5f} ({label}) | before (left) vs after (right) | sorted by max removed area ascending",
            font=fnt, fill=(200, 200, 200))

    for i, (stem, n_removed, max_area, areas, orig_lines, filt_lines) in enumerate(examples):
        y0 = i * (CELL + 48) + 60
        ip = D1 / "train" / "images" / f"{stem}.jpg"
        try:
            img = Image.open(ip).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
        except Exception:
            img = Image.new("RGB", (CELL, CELL), (40, 40, 40))

        # Before panel
        canvas.paste(img, (10, y0))
        dw2 = ImageDraw.Draw(canvas)

        # Identify removed window lines for highlighting
        removed_parts = []
        kept_parts = []
        for line in orig_lines:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] == "2":
                a = float(parts[3]) * float(parts[4])
                if a > threshold:
                    removed_parts.append((a, parts))
                    continue
            kept_parts.append(parts)

        # Draw kept in normal colors
        for parts in kept_parts:
            draw_bbox(dw2, parts, 10, y0, CELL, CELL)
        # Draw removed in red (thick) with area label
        for a, parts in removed_parts:
            draw_bbox(dw2, parts, 10, y0, CELL, CELL, highlight_removed=True)

        # After panel
        canvas.paste(img, (CELL + 20, y0))
        dw3 = ImageDraw.Draw(canvas)
        for line in filt_lines:
            parts = line.strip().split()
            if parts:
                draw_bbox(dw3, parts, CELL + 20, y0, CELL, CELL)

        # Label row
        area_str = ", ".join(f"{a:.5f}" for a in sorted(areas, reverse=True)[:3])
        if len(areas) > 3:
            area_str += f" ... +{len(areas)-3}"
        label_text = f"{stem[:45]}  |  removed={n_removed}  |  areas=[{area_str}]"
        bb = dw.textbbox((10, y0 + CELL + 2), label_text, font=sf)
        dw.rectangle(bb, fill=(0, 0, 0, 180))
        dw.text((10, y0 + CELL + 2), label_text, font=sf, fill=(200, 200, 200))

    return canvas


def main():
    args = parse_args()

    print("Scanning d1 window areas...")
    areas = load_areas()
    print(f"  Total window annotations: {len(areas)}")

    if args.threshold is not None:
        threshold = args.threshold
        label = f"threshold={threshold}"
    elif args.percentile is not None:
        p = args.percentile
        threshold = float(np.percentile(areas, p))
        label = f"P{p}"
    else:
        p = 97.0
        threshold = float(np.percentile(areas, p))
        label = f"P{p}"

    print(f"  Threshold: {threshold:.6f} ({label})")

    total_before = 0
    total_after = 0
    total_removed = 0
    examples = []

    for split in ("train", "valid", "test"):
        src = D1 / split / "labels"
        dst = OUT / split / "labels"
        dst.mkdir(parents=True, exist_ok=True)

        split_before = 0
        split_after = 0
        split_removed = 0

        for f in sorted(src.glob("*.txt")):
            stem = f.stem
            orig_lines = open(f).readlines()
            n_before = count_windows(f)

            # Collect areas of all windows that will be removed
            removed_areas = []
            for line in orig_lines:
                parts = line.strip().split()
                if len(parts) >= 5 and parts[0] == "2":
                    a = float(parts[3]) * float(parts[4])
                    if a > threshold:
                        removed_areas.append(a)

            filter_labels(f, dst / f.name, threshold)

            n_after = count_windows(dst / f.name)
            n_removed = n_before - n_after

            total_before += n_before
            total_after += n_after
            total_removed += n_removed
            split_before += n_before
            split_after += n_after
            split_removed += n_removed

            if n_removed > 0 and split == "train":
                filt_lines = open(dst / f.name).readlines()
                max_area = max(removed_areas)
                examples.append((stem, n_removed, max_area, removed_areas,
                                 orig_lines, filt_lines))

        print(f"  {split:>6s}: {split_before:>6d} → {split_after:>6d}  (removed {split_removed:>5d})")

    print(f"\n  TOTAL: {total_before} → {total_after}  (removed {total_removed})")

    # --- Visualization ---
    out_png = OUT_PNG / f"d1_window_filter_{label.replace('.', '_')}.png"
    if examples:
        examples.sort(key=lambda x: x[2])  # sort by max_area ascending
        cv = make_visualization(examples[:16], threshold, label)
        cv.save(out_png)
        print(f"  Saved: {out_png}  ({min(16, len(examples))} examples)")
    else:
        print("  No windows removed — no visualization generated.")


if __name__ == "__main__":
    main()
