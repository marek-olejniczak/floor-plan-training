#!/usr/bin/env python3
"""
infer_two_model.py — Inferencja dwoma modelami:
  1. Walls model (1 klasa: wall)
  2. Doors+Windows model (2 klasy: door, window)
Wyniki sa laczone z remapem klas: wall=0, door=1, window=2.

Uzycie:
  uv run python experiments/two_model/infer_two_model.py --source obraz.jpg
  uv run python experiments/two_model/infer_two_model.py --source folder/ --save
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

from ultralytics import YOLO

CLASS_NAMES = {0: "wall", 1: "door", 2: "window"}
CLASS_COLORS = {0: "#2ecc71", 1: "#3498db", 2: "#f39c12"}

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "runs"

DEFAULT_WALLS_MODEL = RUNS_DIR / "walls_v1" / "weights" / "best.pt"
DEFAULT_DW_MODEL = Path.home() / "projects" / "trening" / "runs" / "doors_windows_v2" / "weights" / "best.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Inferencja dwumodelowa (walls + doors/windows)")
    parser.add_argument("--source", required=True, help="Sciezka do obrazu lub folderu")
    parser.add_argument("--walls-model", default=str(DEFAULT_WALLS_MODEL), help="Wagi modelu scian")
    parser.add_argument("--dw-model", default=str(DEFAULT_DW_MODEL), help="Wagi modelu drzwi+okien")
    parser.add_argument("--conf", type=float, default=0.25, help="Prog ufnosci")
    parser.add_argument("--iou", type=float, default=0.5, help="Prog IoU NMS")
    parser.add_argument("--imgsz", type=int, default=1024, help="Rozmiar wejscia")
    parser.add_argument("--save", action="store_true", help="Zapisac obrazy z boxami")
    parser.add_argument("--save-dir", default="runs/infer_two_model", help="Katalog wyjsciowy (z --save)")
    return parser.parse_args()


def load_models(walls_path, dw_path):
    print(f"[MODEL] Ladowanie scian: {walls_path}")
    walls_model = YOLO(str(walls_path))
    print(f"  task={walls_model.task}")

    print(f"[MODEL] Ladowanie drzwi+okien: {dw_path}")
    dw_model = YOLO(str(dw_path))
    print(f"  task={dw_model.task}")

    return walls_model, dw_model


def run_inference(model, image, conf, iou, imgsz):
    results = model.predict(
        source=np.array(image),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=0,
        verbose=False,
    )[0]
    return results


def extract_detections(results, class_offset=0, class_map=None):
    detections = []
    if getattr(results, "obb", None) is not None:
        boxes = results.obb.xyxyxyxy.cpu().numpy()
        confs = results.obb.conf.cpu().numpy()
        cls_ids = results.obb.cls.cpu().numpy()
        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            mapped = class_map[int(cls_id)] if class_map else int(cls_id) + class_offset
            detections.append({
                "xyxyxyxy": box,
                "confidence": float(conf),
                "class_id": mapped,
            })
    elif getattr(results, "boxes", None) is not None and results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy()
        for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confs, cls_ids):
            mapped = class_map[int(cls_id)] if class_map else int(cls_id) + class_offset
            detections.append({
                "xyxy": (float(x1), float(y1), float(x2), float(y2)),
                "confidence": float(conf),
                "class_id": mapped,
            })
    return detections


def draw_detections(image, detections, font=None):
    draw = ImageDraw.Draw(image, "RGBA")

    for det in detections:
        cls_id = det["class_id"]
        color = CLASS_COLORS.get(cls_id, "#e74c3c")
        rgb = tuple(int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
        fill = rgb + (64,)

        if "xyxyxyxy" in det:
            poly = [(float(x), float(y)) for x, y in det["xyxyxyxy"]]
            draw.polygon(poly, fill=fill, outline=rgb + (255,), width=3)
            cx = sum(p[0] for p in poly) / len(poly)
            cy = sum(p[1] for p in poly) / len(poly)
        else:
            x1, y1, x2, y2 = det["xyxy"]
            draw.rectangle([x1, y1, x2, y2], fill=fill, outline=rgb + (255,), width=3)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        label = f"{CLASS_NAMES[cls_id]} {det['confidence']:.0%}"
        if font:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            tw, th = len(label) * 8, 14
        lx, ly = cx - tw / 2 - 4, cy - th / 2
        draw.rectangle([lx, ly, lx + tw + 8, ly + th + 4], fill=(0, 0, 0, 180))
        draw.text((cx, cy + 1), label, fill=(255, 255, 255), font=font, anchor="mm")

    return image


def main():
    args = parse_args()
    source = Path(args.source)

    if not source.exists():
        print(f"[BLAD] Nie znaleziono: {source}")
        return

    walls_model, dw_model = load_models(args.walls_model, args.dw_model)

    # kolekcja obrazow
    if source.is_file():
        paths = [source]
    else:
        paths = sorted(
            p for p in source.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    if args.save:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for img_path in paths:
        print(f"\n[INFER] {img_path.name}")
        image = Image.open(img_path).convert("RGB")

        # Walls
        r_walls = run_inference(walls_model, image, args.conf, args.iou, args.imgsz)
        det_walls = extract_detections(r_walls, class_map={0: 0})
        print(f"  sciany: {len(det_walls)}")

        # Doors+Windows
        r_dw = run_inference(dw_model, image, args.conf, args.iou, args.imgsz)
        det_dw = extract_detections(r_dw, class_map={0: 1, 1: 2})
        print(f"  drzwi+okna: {len(det_dw)}")

        all_detections = det_walls + det_dw
        all_detections.sort(key=lambda d: d["confidence"], reverse=True)

        print(f"  lacznie: {len(all_detections)}")

        if args.save:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except Exception:
                font = None
            result_img = draw_detections(image.copy(), all_detections, font=font)
            out_path = save_dir / img_path.name
            result_img.save(out_path)
            print(f"  zapisano: {out_path}")

    print("\nGotowe.")


if __name__ == "__main__":
    main()
