#!/usr/bin/env python3
"""
visualize_pseudo.py — Wizualizacja pseudo-labelowanych przykładów z detekcją kąta.
Obsługuje oba formaty: YOLO (cls cx cy w h) i wielokąt (cls x1 y1 ... x5 y5).
"""

import os, sys, random, math
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

DATASET = Path.home() / "data" / "pseudo_labeled_walls"
OUTPUT = Path("/mnt/d/rzuty/trening/pseudo_label_review.png")
N_EXAMPLES = 30
GRID_COLS = 6
GRID_ROWS = 5
CELL_W, CELL_H = 512, 512

CLASS_COLORS = {
    0: (0, 200, 0),    # wall = green
    1: (255, 100, 0),   # door = orange
    2: (0, 100, 255),   # window = blue
}
CLASS_LABELS = {0: "wall", 1: "door", 2: "window"}


def parse_label_line(line: str):
    """Zwraca (cls_id, [(x_norm, y_norm), ...])."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls_id = int(parts[0])
    coords = list(map(float, parts[1:]))
    # Wielokąt: 10 wartości (x1 y1 x2 y2 x3 y3 x4 y4 x5 y5)
    if len(coords) >= 10:
        pts = [(coords[i], coords[i + 1]) for i in range(0, 10, 2)]
        return cls_id, pts
    # YOLO: cx cy w h
    cx, cy, w, h = coords[:4]
    pts = [
        (cx - w / 2, cy - h / 2),
        (cx + w / 2, cy - h / 2),
        (cx + w / 2, cy + h / 2),
        (cx - w / 2, cy + h / 2),
        (cx - w / 2, cy - h / 2),
    ]
    return cls_id, pts


def detect_dominant_angle(img_bgr):
    """Wykrywa dominujący kąt linii w obrazie używając HoughLinesP."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                            threshold=80, minLineLength=40, maxLineGap=10)
    if lines is None:
        return 0.0

    angle_weights = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 30:
            continue
        # Kąt w stopniach, znormalizowany do 0..180
        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
        # Zaokrągl do 1°
        angle_key = round(angle)
        angle_weights.append((angle_key, length))

    if not angle_weights:
        return 0.0

    # Suma długości na kąt
    angle_sums = Counter()
    for ang, w in angle_weights:
        angle_sums[ang] += w

    dominant = angle_sums.most_common(1)[0][0]
    # Dla czytelności: jeśli ~90°, to 90°, jeśli ~0°, to 0°
    return float(dominant)


def draw_annotations(img_pil, label_lines, img_w, img_h):
    """Rysuje wszystkie anotacje na obrazie PIL."""
    draw = ImageDraw.Draw(img_pil)

    # Pogrubiona czcionka
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("DejaVuSans-Bold.ttf", 11)
    except (OSError, AttributeError):
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
            font_small = ImageFont.truetype("DejaVuSans.ttf", 11)
        except (OSError, AttributeError):
            font = ImageFont.load_default()
            font_small = font

    for line in label_lines:
        parsed = parse_label_line(line)
        if parsed is None:
            continue
        cls_id, pts_norm = parsed
        color = CLASS_COLORS.get(cls_id, (255, 255, 0))
        label = CLASS_LABELS.get(cls_id, str(cls_id))

        # Konwersja znormalizowanych współrzędnych na piksele
        pts = [(int(x * img_w), int(y * img_h)) for x, y in pts_norm]

        # Rysuj obrys
        thickness = 3
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=color, width=thickness)

        # Etykieta przy pierwszym punkcie
        text = f"{label}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx, ty = pts[0]
        draw.rectangle([tx, ty - th - 2, tx + tw + 4, ty], fill=color)
        draw.text((tx + 2, ty - th - 2), text, fill=(0, 0, 0), font=font)


def create_montage():
    """Zbiera losowe przykłady i tworzy mozaikę."""
    # Zbierz wszystkie obrazki
    all_entries = []
    for ds_dir in sorted(DATASET.iterdir()):
        if not ds_dir.is_dir():
            continue
        for split in ("train", "valid", "test"):
            img_dir = ds_dir / split / "images"
            lbl_dir = ds_dir / split / "labels"
            if not img_dir.is_dir():
                continue
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                all_entries.append((img_path, lbl_path, ds_dir.name, split))

    if not all_entries:
        print("[BLAD] Brak obrazów w datasiecie!")
        return

    random.seed(42)
    selected = random.sample(all_entries, min(N_EXAMPLES, len(all_entries)))
    print(f"Wybrano {len(selected)} losowych przykładów z {len(all_entries)} dostępnych")

    montage = Image.new("RGB", (GRID_COLS * CELL_W, GRID_ROWS * CELL_H), (30, 30, 30))

    for idx, (img_path, lbl_path, ds_name, split) in enumerate(selected):
        row, col = divmod(idx, GRID_COLS)
        cx, cy = col * CELL_W, row * CELL_H

        # Wczytaj obraz
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_rgb.shape[:2]
        img_pil = Image.fromarray(img_rgb).resize((CELL_W, CELL_H), Image.LANCZOS)

        # Wczytaj labele
        lines = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                lines = f.readlines()

        # Rysuj anotacje
        draw_annotations(img_pil, lines, CELL_W, CELL_H)

        # Detekcja kąta (na oryginalnym obrazie)
        angle = detect_dominant_angle(img_bgr)

        # Nadruk informacji
        draw = ImageDraw.Draw(img_pil)
        try:
            font_info = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        except (OSError, AttributeError):
            try:
                font_info = ImageFont.truetype("DejaVuSans.ttf", 16)
            except (OSError, AttributeError):
                font_info = ImageFont.load_default()

        total_objs = sum(1 for l in lines if l.strip())
        info = f"{ds_name}/{split} | kąt: {angle:.0f}° | obj: {total_objs}"
        bbox = draw.textbbox((0, 0), info, font=font_info)
        iw, ih = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([2, 2, iw + 8, ih + 6], fill=(0, 0, 0, 180))
        draw.text((6, 4), info, fill=(255, 255, 255), font=font_info)

        # Wklej do mozaiki
        montage.paste(img_pil, (cx, cy))

    # Legenda
    legend_y = GRID_ROWS * CELL_H + 10
    leg_img = Image.new("RGB", (GRID_COLS * CELL_W, 80), (30, 30, 30))
    draw = ImageDraw.Draw(leg_img)
    try:
        font_leg = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except (OSError, AttributeError):
        font_leg = ImageFont.load_default()

    x = 20
    for cls_id, color in CLASS_COLORS.items():
        draw.rectangle([x, 10, x + 30, 40], fill=color)
        draw.text((x + 36, 12), f"{cls_id}={CLASS_LABELS[cls_id]}", fill=(255, 255, 255), font=font_leg)
        x += 180

    draw.text((x + 20, 12), f"Total: {len(all_entries)} img | Pokazano: {len(selected)}", fill=(200, 200, 200), font=font_leg)

    # Sklej
    final = Image.new("RGB", (GRID_COLS * CELL_W, GRID_ROWS * CELL_H + 90), (30, 30, 30))
    final.paste(montage, (0, 0))
    final.paste(leg_img, (0, GRID_ROWS * CELL_H))

    final.save(str(OUTPUT))
    print(f"Mozaika zapisana do {OUTPUT}")
    print(f"  Rozmiar: {final.size} px")


if __name__ == "__main__":
    create_montage()
