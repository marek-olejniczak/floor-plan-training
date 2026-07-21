#!/usr/bin/env python3
"""filter_d1_windows.py — filter oversized windows in d1 by area (+ optional aspect ratio)."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

D1 = Path("/mnt/d/rzuty/dane/yolo11datasets/walls_doors_windows/d1")
OUT = Path("/mnt/d/rzuty/dane/yolo11datasets/walls_doors_windows/d1_filtered")
OUT_PNG = Path("/mnt/d/rzuty/trening")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
CELL = 300
COLORS = {0: (0, 100, 255), 1: (0, 200, 0), 2: (255, 150, 0)}


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--percentile", type=float, default=None,
                   help="Percentile for area threshold (e.g. 99.7)")
    g.add_argument("--threshold", type=float, default=None,
                   help="Direct area threshold (w*h)")
    p.add_argument("--max-aspect", type=float, default=None,
                   help="Only filter windows where aspect ratio <= this value (long/short)")
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


def should_remove(w, h, area, threshold, max_aspect):
    if area <= threshold:
        return False
    if max_aspect is not None:
        aspect = max(w, h) / min(w, h) if min(w, h) > 0 else float("inf")
        if aspect > max_aspect:
            return False
    return True


def filter_labels(src_label, dst_label, threshold, max_aspect):
    with open(src_label) as fin, open(dst_label, "w") as fout:
        for line in fin:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] == "2":
                w = float(parts[3])
                h = float(parts[4])
                a = w * h
                if should_remove(w, h, a, threshold, max_aspect):
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
    color = (255, 0, 0) if highlight_removed else COLORS.get(cid, (200, 200, 200))
    width = 3 if highlight_removed else 2
    dw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def make_visualization(examples, threshold, max_aspect, label):
    n = len(examples)
    rows = n
    img_w = CELL * 2 + 20
    img_h = rows * (CELL + 48) + 60

    canvas = Image.new("RGB", (img_w, img_h), (30, 30, 30))
    dw = ImageDraw.Draw(canvas)

    fnt = ImageFont.truetype(FONT, 20) if Path(FONT).exists() else None
    sf = ImageFont.truetype(FONT, 13) if fnt else None

    aspect_desc = f"max_aspect<={max_aspect}" if max_aspect is not None else "no_aspect_filter"
    dw.text((10, 8),
            f"d1 window filter — area>{threshold:.5f} ({label}), {aspect_desc} | before vs after | sorted by max removed area ascending",
            font=fnt, fill=(200, 200, 200))

    for i, (stem, n_removed, max_area, removed_infos, orig_lines, filt_lines) in enumerate(examples):
        y0 = i * (CELL + 48) + 60
        ip = D1 / "train" / "images" / f"{stem}.jpg"
        try:
            img = Image.open(ip).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
        except Exception:
            img = Image.new("RGB", (CELL, CELL), (40, 40, 40))

        canvas.paste(img, (10, y0))
        dw2 = ImageDraw.Draw(canvas)

        removed_parts = []
        kept_parts = []
        for line in orig_lines:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] == "2":
                w = float(parts[3])
                h = float(parts[4])
                a = w * h
                if should_remove(w, h, a, threshold, max_aspect):
                    removed_parts.append((a, parts))
                    continue
            kept_parts.append(parts)

        for parts in kept_parts:
            draw_bbox(dw2, parts, 10, y0, CELL, CELL)
        for a, parts in removed_parts:
            draw_bbox(dw2, parts, 10, y0, CELL, CELL, highlight_removed=True)

        canvas.paste(img, (CELL + 20, y0))
        dw3 = ImageDraw.Draw(canvas)
        for line in filt_lines:
            parts = line.strip().split()
            if parts:
                draw_bbox(dw3, parts, CELL + 20, y0, CELL, CELL)

        info_str = ", ".join(
            f"a={a:.5f}" if ar is None else f"a={a:.5f}/ar={ar:.1f}"
            for a, ar in removed_infos[:3]
        )
        if len(removed_infos) > 3:
            info_str += f" ... +{len(removed_infos)-3}"
        label_text = f"{stem[:40]}  |  removed={n_removed}  |  [{info_str}]"
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
        label = f"thr={threshold}"
    elif args.percentile is not None:
        p = args.percentile
        threshold = float(np.percentile(areas, p))
        label = f"P{p}"
    else:
        p = 97.0
        threshold = float(np.percentile(areas, p))
        label = f"P{p}"

    max_aspect = args.max_aspect

    print(f"  Area threshold: {threshold:.6f} ({label})")
    if max_aspect:
        print(f"  Max aspect ratio: {max_aspect}")
    else:
        print(f"  No aspect ratio filter")

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

            removed_infos = []
            for line in orig_lines:
                parts = line.strip().split()
                if len(parts) >= 5 and parts[0] == "2":
                    w = float(parts[3])
                    h = float(parts[4])
                    a = w * h
                    if should_remove(w, h, a, threshold, max_aspect):
                        ar = (max(w, h) / min(w, h)) if min(w, h) > 0 else float("inf")
                        removed_infos.append((a, ar))

            filter_labels(f, dst / f.name, threshold, max_aspect)

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
                max_area = max(a for a, _ in removed_infos)
                examples.append((stem, n_removed, max_area, removed_infos,
                                 orig_lines, filt_lines))

        print(f"  {split:>6s}: {split_before:>6d} → {split_after:>6d}  (removed {split_removed:>5d})")

    print(f"\n  TOTAL: {total_before} → {total_after}  (removed {total_removed})")

    # --- Visualization ---
    aspect_suffix = f"_ar{max_aspect}" if max_aspect is not None else ""
    out_png = OUT_PNG / f"d1_window_filter_{label.replace('.', '_')}{aspect_suffix}.png"
    if examples:
        examples.sort(key=lambda x: x[2])
        cv = make_visualization(examples[:16], threshold, max_aspect, label)
        cv.save(out_png)
        print(f"  Saved: {out_png}  ({min(16, len(examples))} examples)")
    else:
        print("  No windows removed — no visualization generated.")


if __name__ == "__main__":
    main()
