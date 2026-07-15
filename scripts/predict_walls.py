#!/usr/bin/env python3
"""
predict_walls.py — Pure inference on walls datasets.
Saves ALL predictions (no filtering) to ~/data/raw_predictions/.
Format: cls cx cy w h conf filter_status (7 fields, filter_status=0).
Rotation correction applied before inference.
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

WALLS_SRC = Path.home() / "data" / "pseudo_labeled_walls"
DST = Path.home() / "data" / "raw_predictions"
MODEL_PATH = Path.home() / "projects" / "trening" / "runs" / "doors_windows_v2" / "weights" / "best.pt"
FALLBACK_MODEL = "yolo11s.pt"

CONF_THRESH = 0.5
IOU_THRESH = 0.45
MAX_DET = 500
ANGLE_TOLERANCE = 2
SPLITS = ["train", "valid", "test"]
AXES = [0.0, 90.0, 180.0]


def detect_top2_angles(img_bgr):
    h, w = img_bgr.shape[:2]
    scale = 512.0 / max(h, w)
    if scale < 1.0:
        small = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = img_bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 30, 100, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=20, maxLineGap=15)
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
        bin_angle = round(angle * 2) / 2
        angle_weights[bin_angle] += length
    if not angle_weights:
        return []
    sorted_angles = angle_weights.most_common(4)
    a1, w1 = sorted_angles[0]
    top2 = [(a1, w1)]
    for a, w in sorted_angles[1:]:
        if abs(a - a1) >= 30 and abs(a - a1) <= 150:
            top2.append((a, w))
            break
    if len(top2) < 2:
        top2.append((a1, 0))
    return top2


def compute_rotation(top2_angles):
    if not top2_angles:
        return 0.0, 0.0, False, {"reason": "no_lines"}
    a1, w1 = top2_angles[0]
    a2, w2 = top2_angles[1]
    def deviation_from_axes(angle):
        return min(abs(angle - axis) for axis in AXES)
    dev1 = deviation_from_axes(a1)
    dev2 = deviation_from_axes(a2) if w2 > 0 else dev1
    max_dev = max(dev1, dev2)
    if max_dev <= ANGLE_TOLERANCE:
        return 0.0, max_dev, False, {"reason": "within_tolerance", "a1": a1, "dev1": dev1, "a2": a2, "dev2": dev2, "max_dev": max_dev}
    nearest_axis = min(AXES, key=lambda ax: abs(a1 - ax))
    rotation = nearest_axis - a1
    return rotation, max_dev, True, {"reason": "rotated", "a1": a1, "dev1": dev1, "a2": a2, "dev2": dev2, "max_dev": max_dev, "rotation": round(rotation, 1)}


def rotate_image(img_bgr, angle_deg):
    h, w = img_bgr.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(w * cos + h * sin)
    new_h = int(w * sin + h * cos)
    M[0, 2] += new_w / 2 - w / 2
    M[1, 2] += new_h / 2 - h / 2
    rotated = cv2.warpAffine(img_bgr, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return rotated, M, new_w, new_h


def rotate_polygon(poly_pts_norm, M, orig_w, orig_h, new_w, new_h):
    rotated = []
    for nx, ny in poly_pts_norm:
        px = nx * orig_w
        py = ny * orig_h
        rx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
        ry = M[1, 0] * px + M[1, 1] * py + M[1, 2]
        rotated.append((rx / new_w, ry / new_h))
    return rotated


def rotate_yolo_box(cx, cy, w, h, M, orig_w, orig_h, new_w, new_h):
    corners_px = [
        ((cx - w / 2) * orig_w, (cy - h / 2) * orig_h),
        ((cx + w / 2) * orig_w, (cy - h / 2) * orig_h),
        ((cx + w / 2) * orig_w, (cy + h / 2) * orig_h),
        ((cx - w / 2) * orig_w, (cy + h / 2) * orig_h),
    ]
    new_corners = []
    for px, py in corners_px:
        rx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
        ry = M[1, 0] * px + M[1, 1] * py + M[1, 2]
        new_corners.append((rx, ry))
    xs = [p[0] for p in new_corners]
    ys = [p[1] for p in new_corners]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    new_cx = (xmin + xmax) / 2 / new_w
    new_cy = (ymin + ymax) / 2 / new_h
    new_w_n = (xmax - xmin) / new_w
    new_h_n = (ymax - ymin) / new_h
    return new_cx, new_cy, new_w_n, new_h_n


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
    pts = [
        (cx - w / 2, cy - h / 2),
        (cx + w / 2, cy - h / 2),
        (cx + w / 2, cy + h / 2),
        (cx - w / 2, cy + h / 2),
        (cx - w / 2, cy - h / 2),
    ]
    return cls_id, pts


def format_polygon_line(cls_id, pts):
    coords = []
    for x, y in pts[:5]:
        coords.append(f"{x:.8f} {y:.8f}")
    return f"{cls_id} {' '.join(coords)}"


def format_yolo_line(cls_id, cx, cy, w, h, conf=0.0, filter_status=0):
    return f"{cls_id} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f} {conf:.4f} {filter_status}"


def process_walls():
    model_path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL
    print(f"[INFO] Laduje model: {model_path}")
    model = YOLO(str(model_path))

    for split in SPLITS:
        (DST / split / "images").mkdir(parents=True, exist_ok=True)
        (DST / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = {
        "total": 0, "rotated": 0,
        "walls": 0, "doors": 0, "windows": 0,
        "rotated_examples": [],
    }

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
            print(f"  Split {split}: {len(img_paths)} obrazow")

            for img_path in tqdm(img_paths, desc=f"  {ds_dir.name}/{split}", unit="img"):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                stats["total"] += 1
                lbl_path = lbl_dir / (img_path.stem + ".txt")

                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue
                orig_h, orig_w = img_bgr.shape[:2]

                wall_lines = []
                if lbl_path.exists():
                    with open(lbl_path) as f:
                        wall_lines = [l.strip() for l in f if l.strip()]

                # ---- DETEKCJA KATA ----
                top2 = detect_top2_angles(img_bgr)
                rotation_deg, max_dev, needs_rotation, angle_info = compute_rotation(top2)

                # ---- ROTACJA ----
                if needs_rotation:
                    rotated_img, M, new_w, new_h = rotate_image(img_bgr, rotation_deg)
                    current_img = rotated_img
                    rotated_wall_lines = []
                    for line in wall_lines:
                        parsed = parse_label_line(line)
                        if parsed is None:
                            continue
                        cls_id, pts = parsed
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
                    M = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

                # ---- PREDYKCJA ----
                results = model.predict(
                    current_img,
                    imgsz=max(new_w, new_h),
                    conf=CONF_THRESH,
                    iou=IOU_THRESH,
                    max_det=MAX_DET,
                    verbose=False,
                )

                # ---- ZAPIS RAW (wszystkie predykcje, filter_status=0) ----
                out_name = f"{ds_dir.name}_{img_path.stem}"
                out_img = DST / split / "images" / f"{out_name}.jpg"
                out_lbl = DST / split / "labels" / f"{out_name}.txt"

                cv2.imwrite(str(out_img), current_img)

                lines_out = list(final_wall_lines)
                for result in results:
                    if result.boxes is None:
                        continue
                    for box_tensor in result.boxes:
                        cls_id = int(box_tensor.cls[0])
                        conf = float(box_tensor.conf[0])
                        mapped_cls = cls_id + 1  # 0→1 (door), 1→2 (window)
                        xyxyn = box_tensor.xyxyn[0]
                        x1, y1, x2, y2 = xyxyn.tolist()
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        w_b = x2 - x1
                        h_b = y2 - y1
                        lines_out.append(format_yolo_line(mapped_cls, cx, cy, w_b, h_b, conf, 0))

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

    # ---- PODSUMOWANIE ----
    print(f"\n{'='*60}")
    print(f"  PODSUMOWANIE")
    print(f"  Total images:     {stats['total']}")
    print(f"  Rotated:          {stats['rotated']} ({100*stats['rotated']/max(stats['total'],1):.1f}%)")
    print(f"  Wall labels:      {stats['walls']}")
    print(f"  Door labels:      {stats['doors']}")
    print(f"  Window labels:    {stats['windows']}")
    print(f"  Output:           {DST}")
    print(f"{'='*60}")

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
            }
        }, f, indent=2)
    print(f"  Metadane: {meta_path}")


if __name__ == "__main__":
    process_walls()
