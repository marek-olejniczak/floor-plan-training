#!/usr/bin/env python3
"""
pseudo_label_rotated.py — Angle-aware pseudo-labeling.
Dla każdego obrazu walls:
  1. Wykryj TOP2 kąty dominantne (HoughLinesP)
  2. Jeżeli odchylenie od osi > 2° → obróć obraz + anotacje ścian
  3. Puść predykcję modelu na wyprostowanym obrazie
  4. Zapisz (obraz + ściany + pseudo drzwi/okna) do corrected_walls/
"""

import os, sys, math, json
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"
sys.stdout.reconfigure(line_buffering=True)

# --- Konfiguracja ---
WALLS_SRC = Path.home() / "data" / "pseudo_labeled_walls"  # wejściowe walls d1/d2
DST = Path.home() / "data" / "corrected_walls"              # wyjście (skorygowane)
MODEL_PATH = Path.home() / "projects" / "trening" / "runs" / "doors_windows_v2" / "weights" / "best.pt"
# Jeśli model nie istnieje, użyjemy domyślnego yolo11s.pt
FALLBACK_MODEL = "yolo11s.pt"

CONF_THRESH = 0.5
IOU_THRESH = 0.45
MAX_DET = 500

ANGLE_TOLERANCE = 2  # stopnie — margines błędu
SPLITS = ["train", "valid", "test"]

# Do klasyfikacji kątów
AXES = [0.0, 90.0, 180.0]

CLASS_NAMES = {0: "wall", 1: "door", 2: "window"}

# Filtry fałszywych okien (Etap B)
MAX_WINDOW_AREA = 0.356  # P95 × 1.5 z analyze_window_area.py
MIN_WALL_OVERLAP = 0.7   # minimalny overlap pseudo-window ze ścianą


# ============================================================
# 1. DETEKCJA KĄTA DOMINUJĄCEGO
# ============================================================

def detect_top2_angles(img_bgr):
    """
    Zwraca listę (angle_deg, weight) dla TOP2 dominantnych kątów.
    angle_deg w zakresie [0, 180). Używa skalowania dla szybkości.
    """
    h, w = img_bgr.shape[:2]
    scale = 512.0 / max(h, w)
    if scale < 1.0:
        small = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = img_bgr

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 30, 100, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=50, minLineLength=20, maxLineGap=15
    )

    if lines is None or len(lines) < 5:
        return []

    angle_weights = Counter()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 30:
            continue

        raw_angle = math.degrees(math.atan2(dy, dx))
        angle = raw_angle % 180
        if angle < 0:
            angle += 180

        # Zaokrągl do 0.5 stopnia
        bin_angle = round(angle * 2) / 2
        angle_weights[bin_angle] += length

    if not angle_weights:
        return []

    # Wyciągnij TOP2
    sorted_angles = angle_weights.most_common(4)
    # Filtruj: drugi pik musi być oddalony o >= 30° od pierwszego
    a1, w1 = sorted_angles[0]
    top2 = [(a1, w1)]
    for a, w in sorted_angles[1:]:
        if abs(a - a1) >= 30 and abs(a - a1) <= 150:
            top2.append((a, w))
            break

    # Jeśli nie znaleziono drugiego piku oddalonego, użyj a1
    if len(top2) < 2:
        top2.append((a1, 0))

    return top2


def compute_rotation(top2_angles):
    """
    Oblicza kąt rotacji na podstawie TOP2 dominantnych kątów.
    Zwraca (rotation_deg, max_deviation, needs_rotation, info_dict).
    """
    if not top2_angles:
        return 0.0, 0.0, False, {"reason": "no_lines"}

    a1, w1 = top2_angles[0]
    a2, w2 = top2_angles[1]

    # Dla każdego kąta znajdź minimalne odchylenie od osi (0, 90, 180)
    def deviation_from_axes(angle):
        return min(abs(angle - axis) for axis in AXES)

    dev1 = deviation_from_axes(a1)
    dev2 = deviation_from_axes(a2) if w2 > 0 else dev1
    max_dev = max(dev1, dev2)

    if max_dev <= ANGLE_TOLERANCE:
        return 0.0, max_dev, False, {
            "reason": "within_tolerance",
            "a1": a1, "dev1": dev1, "a2": a2, "dev2": dev2, "max_dev": max_dev
        }

    # Oblicz rotację: sprowadź a1 do najbliższej osi
    nearest_axis = min(AXES, key=lambda ax: abs(a1 - ax))
    rotation = nearest_axis - a1

    # Jeśli w2 ma dużą wagę, możemy też sprawdzić czy to ma sens
    return rotation, max_dev, True, {
        "reason": "rotated",
        "a1": a1, "dev1": dev1, "a2": a2, "dev2": dev2,
        "max_dev": max_dev, "rotation": round(rotation, 1)
    }


# ============================================================
# 2. ROTACJA OBRAZU I ANOTACJI
# ============================================================

def rotate_image(img_bgr, angle_deg):
    """Rotuje obraz i zwraca (rotated_img, rotation_matrix, new_w, new_h)."""
    h, w = img_bgr.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(w * cos + h * sin)
    new_h = int(w * sin + h * cos)

    M[0, 2] += new_w / 2 - w / 2
    M[1, 2] += new_h / 2 - h / 2

    rotated = cv2.warpAffine(
        img_bgr, M, (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    return rotated, M, new_w, new_h


def rotate_polygon(poly_pts_norm, M, orig_w, orig_h, new_w, new_h):
    """Rotuje wielokąt (lista (nx,ny) w norm [0,1]) przez macierz M."""
    rotated = []
    for nx, ny in poly_pts_norm:
        px = nx * orig_w
        py = ny * orig_h
        # Transformacja
        rx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
        ry = M[1, 0] * px + M[1, 1] * py + M[1, 2]
        # Z powrotem do norm
        rotated.append((rx / new_w, ry / new_h))
    return rotated


def rotate_yolo_box(cx, cy, w, h, M, orig_w, orig_h, new_w, new_h):
    """
    Rotuje YOLO box (cx,cy,w,h) i zwraca axis-aligned box w nowych współrzędnych.
    """
    # Narożniki w pikselach
    corners_px = [
        ((cx - w / 2) * orig_w, (cy - h / 2) * orig_h),
        ((cx + w / 2) * orig_w, (cy - h / 2) * orig_h),
        ((cx + w / 2) * orig_w, (cy + h / 2) * orig_h),
        ((cx - w / 2) * orig_w, (cy + h / 2) * orig_h),
    ]

    # Transformacja
    new_corners = []
    for px, py in corners_px:
        rx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
        ry = M[1, 0] * px + M[1, 1] * py + M[1, 2]
        new_corners.append((rx, ry))

    xs = [p[0] for p in new_corners]
    ys = [p[1] for p in new_corners]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    # Z powrotem do norm
    new_cx = (xmin + xmax) / 2 / new_w
    new_cy = (ymin + ymax) / 2 / new_h
    new_w_n = (xmax - xmin) / new_w
    new_h_n = (ymax - ymin) / new_h

    return new_cx, new_cy, new_w_n, new_h_n


# ============================================================
# 3. PARSOWANIE LABELI
# ============================================================

def parse_label_line(line):
    """Zwraca (cls_id, [(x_norm, y_norm), ...]) lub None."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls_id = int(parts[0])
    coords = list(map(float, parts[1:]))

    # Wielokąt (11 pól: cls x1 y1 x2 y2 x3 y3 x4 y4 x5 y5)
    if len(coords) >= 10:
        pts = [(coords[i], coords[i + 1]) for i in range(0, 10, 2)]
        return cls_id, pts

    # YOLO (5 pól: cls cx cy w h)
    cx, cy, w, h = coords[:4]
    pts = [
        (cx - w / 2, cy - h / 2),
        (cx + w / 2, cy - h / 2),
        (cx + w / 2, cy + h / 2),
        (cx - w / 2, cy + h / 2),
        (cx - w / 2, cy - h / 2),
    ]
    return cls_id, pts


def format_polygon_line(cls_id, pts):
    """Zapisuje wielokąt do linii YOLO (11 pól: cls x1 y1 ... x5 y5)."""
    coords = []
    for x, y in pts[:5]:  # max 5 punktów
        coords.append(f"{x:.8f} {y:.8f}")
    return f"{cls_id} {' '.join(coords)}"


def format_yolo_line(cls_id, cx, cy, w, h):
    """Zapisuje YOLO box."""
    return f"{cls_id} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}"


# ============================================================
# 4. FUNKCJE POMOCNICZE (filtry)
# ============================================================

def parse_bbox_from_wall(line, img_w, img_h):
    """Zwraca (x1, y1, x2, y2) w pikselach dla ściany (polygon lub YOLO)."""
    parsed = parse_label_line(line)
    if parsed is None:
        return None
    cls_id, pts_norm = parsed
    if cls_id != 0:  # tylko sciany
        return None
    xs = [p[0] * img_w for p in pts_norm]
    ys = [p[1] * img_h for p in pts_norm]
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_overlap_ratio(win_bbox, wall_bboxes):
    """
    Oblicza stosunek powierzchni przecięcia window_bbox z sumą ścian
    do powierzchni window_bbox.
    win_bbox: (x1, y1, x2, y2)
    wall_bboxes: lista [(x1, y1, x2, y2), ...]
    """
    wx1, wy1, wx2, wy2 = win_bbox
    win_area = max(0, wx2 - wx1) * max(0, wy2 - wy1)
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


# ============================================================
# 5. GŁÓWNA PĘTLA
# ============================================================

def process_walls():
    # Załaduj model
    model_path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL
    print(f"[INFO] Ładuję model: {model_path}")
    model = YOLO(str(model_path))

    # Przygotuj katalogi wyjściowe
    for split in SPLITS:
        (DST / split / "images").mkdir(parents=True, exist_ok=True)
        (DST / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = {
        "total": 0, "rotated": 0, "skipped_bad_angle": 0,
        "doors": 0, "windows": 0, "walls": 0,
        "rotated_examples": [],  # lista do wizualizacji
    }

    # Iteracja po d1, d2
    for ds_dir in sorted(WALLS_SRC.iterdir()):
        if not ds_dir.is_dir():
            continue
        print(f"\n{'='*60}")
        print(f"  PRZETWARZANIE: {ds_dir.name}")
        print(f"{'='*60}")

        for split in SPLITS:
            img_dir = ds_dir / split / "images"
            lbl_dir = ds_dir / split / "labels"
            if not img_dir.is_dir():
                continue

            img_paths = sorted(img_dir.iterdir())
            print(f"  Split {split}: {len(img_paths)} obrazów")

            for img_path in tqdm(img_paths, desc=f"  {ds_dir.name}/{split}", unit="img"):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue

                stats["total"] += 1
                lbl_path = lbl_dir / (img_path.stem + ".txt")

                # Wczytaj obraz
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue
                orig_h, orig_w = img_bgr.shape[:2]

                # Wczytaj oryginalne labele ścian
                wall_lines = []
                if lbl_path.exists():
                    with open(lbl_path) as f:
                        wall_lines = [l.strip() for l in f if l.strip()]

                # ---- DETEKCJA KĄTA ----
                top2 = detect_top2_angles(img_bgr)
                rotation_deg, max_dev, needs_rotation, angle_info = compute_rotation(top2)

                # ---- APLIKUJ ROTACJĘ (JEŚLI POTRZEBNA) ----
                if needs_rotation:
                    # Rotacja obrazu
                    rotated_img, M, new_w, new_h = rotate_image(img_bgr, rotation_deg)
                    current_img = rotated_img

                    # Rotacja anotacji ścian
                    rotated_wall_lines = []
                    for line in wall_lines:
                        parsed = parse_label_line(line)
                        if parsed is None:
                            continue
                        cls_id, pts = parsed
                        # Rotuj punkty
                        new_pts = rotate_polygon(pts, M, orig_w, orig_h, new_w, new_h)
                        rotated_wall_lines.append(format_polygon_line(cls_id, new_pts))

                    final_wall_lines = rotated_wall_lines
                    stats["rotated"] += 1

                    if len(stats["rotated_examples"]) < 20:
                        stats["rotated_examples"].append({
                            "src": f"{ds_dir.name}/{split}",
                            "name": img_path.name,
                            "angle": round(rotation_deg, 1),
                            "max_dev": round(max_dev, 1),
                            "a1": angle_info.get("a1"),
                            "a2": angle_info.get("a2"),
                        })
                else:
                    current_img = img_bgr
                    new_w, new_h = orig_w, orig_h
                    final_wall_lines = wall_lines

                # ---- PREDYKCJA ----
                results = model.predict(
                    current_img,
                    imgsz=max(new_w, new_h),  # dopasuj do rozmiaru
                    conf=CONF_THRESH,
                    iou=IOU_THRESH,
                    max_det=MAX_DET,
                    verbose=False,
                )

                # ---- ZAPIS ----
                out_name = f"{ds_dir.name}_{img_path.stem}"
                out_img = DST / split / "images" / f"{out_name}.jpg"
                out_lbl = DST / split / "labels" / f"{out_name}.txt"

                # Zapisz obraz (zawsze jako jpg)
                cv2.imwrite(str(out_img), current_img)

                # ---- FILTRY: area i overlap ze ścianą ----
                wall_bboxes = []
                for line in final_wall_lines:
                    bbox = parse_bbox_from_wall(line, new_w, new_h)
                    if bbox is not None:
                        wall_bboxes.append(bbox)

                filtered = {"area": 0, "overlap": 0, "kept": 0}

                # Zapisz labele (ściany + pseudo z filtrami)
                lines_out = list(final_wall_lines)
                for result in results:
                    if result.boxes is None:
                        continue
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        # Mapowanie: model zwraca 0=door, 1=window
                        # W naszym wyjściu: 1=door, 2=window
                        mapped_cls = cls_id + 1
                        xyxyn = box.xyxyn[0]
                        x1, y1, x2, y2 = xyxyn.tolist()
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        w_b = x2 - x1
                        h_b = y2 - y1

                        # Filtruj tylko windows (cls_id=1 przed mapowaniem)
                        if cls_id == 1:
                            area = w_b * h_b
                            if area > MAX_WINDOW_AREA:
                                filtered["area"] += 1
                                continue
                            win_bbox = (cx - w_b/2, cy - h_b/2, cx + w_b/2, cy + h_b/2)
                            win_bbox_px = (
                                win_bbox[0] * new_w,
                                win_bbox[1] * new_h,
                                win_bbox[2] * new_w,
                                win_bbox[3] * new_h,
                            )
                            overlap = bbox_overlap_ratio(win_bbox_px, wall_bboxes)
                            if overlap < MIN_WALL_OVERLAP:
                                filtered["overlap"] += 1
                                continue

                        lines_out.append(format_yolo_line(mapped_cls, cx, cy, w_b, h_b) + f" {conf:.4f}")
                        filtered["kept"] += 1

                out_lbl.write_text("\n".join(lines_out) + "\n")

                # Statystyki
                for line in lines_out:
                    cls = int(line.split()[0])
                    if cls == 1:
                        stats["doors"] += 1
                    elif cls == 2:
                        stats["windows"] += 1
                    elif cls == 0:
                        stats["walls"] += 1

                stats.setdefault("filtered_area", 0)
                stats.setdefault("filtered_overlap", 0)
                stats.setdefault("filtered_kept", 0)
                stats["filtered_area"] += filtered["area"]
                stats["filtered_overlap"] += filtered["overlap"]
                stats["filtered_kept"] += filtered["kept"]

    # ---- PODSUMOWANIE ----
    total_windows_before = stats["windows"] + stats.get("filtered_area", 0) + stats.get("filtered_overlap", 0)
    print(f"\n{'='*60}")
    print(f"  PODSUMOWANIE")
    print(f"  Total images:     {stats['total']}")
    print(f"  Rotated:          {stats['rotated']} ({100*stats['rotated']/max(stats['total'],1):.1f}%)")
    print(f"  Wall labels:      {stats['walls']}")
    print(f"  Door labels:      {stats['doors']}")
    print(f"  Window labels:    {stats['windows']}")
    if stats.get("filtered_kept", 0) > 0:
        print(f"  --- Filtry okien ---")
        print(f"  Przed filtrem:    {total_windows_before}")
        print(f"  Odrzucone area:   {stats.get('filtered_area', 0)}")
        print(f"  Odrzucone overlap:{stats.get('filtered_overlap', 0)}")
        print(f"  Zachowane:        {stats.get('filtered_kept', 0)}")
    print(f"  Output:           {DST}")
    print(f"{'='*60}")

    # Zapisz metadane rotacji do JSON
    meta_path = DST / "rotation_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "rotated_examples": stats["rotated_examples"],
            "stats": {
                "total": stats["total"],
                "rotated": stats["rotated"],
                "walls": stats["walls"],
                "doors": stats["doors"],
                "windows": stats["windows"],
            },
            "filters": {
                "MAX_WINDOW_AREA": MAX_WINDOW_AREA,
                "MIN_WALL_OVERLAP": MIN_WALL_OVERLAP,
                "filtered_area": stats.get("filtered_area", 0),
                "filtered_overlap": stats.get("filtered_overlap", 0),
                "filtered_kept": stats.get("filtered_kept", 0),
                "windows_before_filter": total_windows_before,
            }
        }, f, indent=2)
    print(f"  Metadane: {meta_path}")


if __name__ == "__main__":
    process_walls()
