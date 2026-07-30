#!/usr/bin/env python3
"""
visualize_comparison.py — Porownanie wizualne dwoch podejsc
na losowych obrazkach z czystego test setu Roboflow.
"""

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

from ultralytics import YOLO

HOME = Path.home()
SCRIPT_DIR = Path(__file__).resolve().parent

D1_TEST = HOME / "data" / "walls_doors_windows" / "d1" / "test"
D2_VALID = HOME / "data" / "walls_doors_windows" / "d2" / "valid"

FINAL_V1 = HOME / "projects" / "trening" / "runs" / "final_v1" / "weights" / "best.pt"
WALLS_V1 = SCRIPT_DIR / ".." / "two_model" / "runs" / "walls_v1" / "weights" / "best.pt"
DW_V2 = HOME / "projects" / "trening" / "runs" / "doors_windows_v2" / "weights" / "best.pt"

OUTPUT_DIR = SCRIPT_DIR / "runs" / "compare_vis"

CONF = 0.25
IOU = 0.5
IMSZ_LIST = [640, 1024]
DEVICE = 0
NUM_SAMPLES = 20

CLASS_COLORS = {
    0: (46, 204, 113),    # wall -> zielony
    1: (52, 152, 219),    # door -> niebieski
    2: (243, 156, 18),    # window -> pomaranczowy
}
CLASS_NAMES = {0: "wall", 1: "door", 2: "window"}
GT_REMAP = {0: 1, 1: 0, 2: 2}  # d1 GT -> common space


def load_test_samples():
    samples = []
    sources = [
        ("d1_test", D1_TEST / "images", D1_TEST / "labels"),
        ("d2_valid", D2_VALID / "images", D2_VALID / "labels"),
    ]
    for src_name, img_dir, lbl_dir in sources:
        if not img_dir.exists():
            continue
        for lbl_path in sorted(lbl_dir.glob("*.txt")):
            stem = lbl_path.stem
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                cand = img_dir / f"{stem}{ext}"
                if cand.exists():
                    samples.append(cand)
                    break
    return samples


def run_inference(model, img, conf=CONF, imgsz=1024):
    results = model.predict(
        source=np.array(img),
        conf=conf,
        iou=IOU,
        imgsz=imgsz,
        device=DEVICE,
        verbose=False,
    )[0]
    dets = []
    if getattr(results, "obb", None) is not None:
        for box, conf, cid in zip(results.obb.xyxyxyxy.cpu(), results.obb.conf.cpu(), results.obb.cls.cpu()):
            pts = box.view(-1, 2)
            x1, y1 = pts[:, 0].min().item(), pts[:, 1].min().item()
            x2, y2 = pts[:, 0].max().item(), pts[:, 1].max().item()
            dets.append({"bbox": (x1, y1, x2, y2), "conf": float(conf), "class": int(cid)})
    elif getattr(results, "boxes", None) is not None and results.boxes is not None:
        for box, conf, cid in zip(results.boxes.xyxy.cpu(), results.boxes.conf.cpu(), results.boxes.cls.cpu()):
            dets.append({"bbox": box.tolist(), "conf": float(conf), "class": int(cid)})
    return dets


def load_gt_lbl(img_path):
    label_dirs = [D1_TEST / "labels", D2_VALID / "labels"]
    stem = img_path.stem
    img_pil = Image.open(img_path)
    img_w, img_h = img_pil.size
    for ld in label_dirs:
        lp = ld / f"{stem}.txt"
        if lp.exists():
            boxes = []
            labels = []
            with open(lp) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    coords = list(map(float, parts[1:]))

                    if len(coords) == 4:
                        cx, cy, bw, bh = coords
                        x1 = (cx - bw / 2) * img_w
                        y1 = (cy - bh / 2) * img_h
                        x2 = (cx + bw / 2) * img_w
                        y2 = (cy + bh / 2) * img_h
                    else:
                        pts = np.array(coords, dtype=float).reshape(-1, 2)
                        pts[:, 0] *= img_w
                        pts[:, 1] *= img_h
                        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
                        x2, y2 = pts[:, 0].max(), pts[:, 1].max()

                    new_cls = GT_REMAP.get(cls, cls)
                    boxes.append((x1, y1, x2, y2))
                    labels.append(new_cls)
            return boxes, labels
    return [], []


def draw_detections(draw, detections, label_prefix=""):
    font = None
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        pass

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_id = det["class"]
        color = CLASS_COLORS.get(cls_id, (231, 76, 60))
        fill = color + (60,)

        # Rysuj jako ImageDraw nie wspiera alpha fill bez dodatkowych operacji
        # Uzywamy outline i transparent overlay
        for i in range(3):
            offset = i * 2
            draw.rectangle([x1 + offset, y1 + offset, x2 - offset, y2 - offset],
                          outline=color, width=3)

        label = f"{label_prefix}{CLASS_NAMES[cls_id]} {det['conf']:.0%}" if "conf" in det else label_prefix or CLASS_NAMES[cls_id]
        if font:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        else:
            tw, th = len(label) * 8, 14
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        lx, ly = cx - tw / 2 - 4, y1 - th - 6
        draw.rectangle([lx, ly, lx + tw + 8, ly + th + 4], fill=(0, 0, 0, 200))
        draw.text((cx, ly + th / 2), label, fill=(255, 255, 255), font=font, anchor="mm")


def make_comparison(img_path, gt_boxes, gt_labels,
                    det_final_640, det_walls_640, det_dw_640,
                    det_final_1024, det_walls_1024, det_dw_1024):
    MAX_W = 640
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    scale = MAX_W / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h))

    def scale_bboxes(dets):
        for d in dets:
            d["bbox"] = tuple(v * scale for v in d["bbox"])
        return dets

    gt_scaled = [(x1*scale, y1*scale, x2*scale, y2*scale) for (x1,y1,x2,y2) in gt_boxes]
    scale_bboxes(det_final_640)
    scale_bboxes(det_walls_640)
    scale_bboxes(det_dw_640)
    scale_bboxes(det_final_1024)
    scale_bboxes(det_walls_1024)
    scale_bboxes(det_dw_1024)

    panel_w = new_w + 40
    row_h = new_h + 50
    total_w = panel_w * 4
    total_h = row_h * 2 + 10
    canvas = Image.new("RGB", (total_w, total_h), (26, 26, 46))
    draw = ImageDraw.Draw(canvas)

    font16 = None
    font14 = None
    try:
        font16 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font14 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        pass

    def safe_rect(draw, x1, y1, x2, y2, color, width=3):
        rx1 = int(min(x1, x2 - 1))
        ry1 = int(min(y1, y2 - 1))
        rx2 = int(max(x1 + 1, x2))
        ry2 = int(max(y1 + 1, y2))
        w = min(width, rx2 - rx1)
        if rx2 > rx1 and ry2 > ry1 and w > 0:
            draw.rectangle([rx1, ry1, rx2, ry2], outline=color, width=int(w))

    def draw_outline(pdraw, x1, y1, x2, y2, color):
        for j in range(3):
            of = j * 2
            safe_rect(pdraw, x1 + of, y1 + of, x2 - of, y2 - of, color)

    rows = [
        ("640", det_final_640, det_walls_640, det_dw_640),
        ("1024", det_final_1024, det_walls_1024, det_dw_1024),
    ]

    for row_idx, (imgsz_label, det_f, det_w, det_d) in enumerate(rows):
        y_base = row_idx * row_h + 10

        if font16:
            draw.text((total_w // 2, y_base), f"imgsz={imgsz_label}", fill=(180, 180, 180), font=font16, anchor="mt")

        panels = [
            (gt_scaled, gt_labels, True),
            (det_f, None, False),
            (det_w, None, False),
            (det_d, None, False),
        ]

        for col_idx, (data, extra, is_gt) in enumerate(panels):
            x_off = col_idx * panel_w + 20
            y_off = y_base + 25

            if col_idx > 0 and font14:
                titles = ["", "final_v1", "walls_v1", "dw_v2"]
                draw.text((x_off + new_w // 2, y_base + 25), titles[col_idx], fill=(200, 200, 200), font=font14, anchor="mt")
                y_off += 5

            panel = img_resized.copy()
            pdraw = ImageDraw.Draw(panel, "RGBA")

            if is_gt:
                for (x1, y1, x2, y2), cls_id in zip(gt_scaled, gt_labels):
                    color = CLASS_COLORS.get(cls_id, (231, 76, 60))
                    draw_outline(pdraw, x1, y1, x2, y2, color + (255,))
                    cx = (x1 + x2) / 2
                    pdraw.text((cx, y1 - 4), CLASS_NAMES[cls_id], fill=(255, 255, 255), font=font14, anchor="mb", stroke_width=1, stroke_fill=(0, 0, 0))
            else:
                for d in data:
                    x1, y1, x2, y2 = d["bbox"]
                    cls_id = d["class"]
                    color = CLASS_COLORS.get(cls_id, (231, 76, 60))
                    draw_outline(pdraw, x1, y1, x2, y2, color + (255,))
                    label = f"{CLASS_NAMES[cls_id]} {d['conf']:.0%}"
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    pdraw.text((cx, cy), label, fill=(255, 255, 255), font=font14, anchor="mm", stroke_width=1, stroke_fill=(0, 0, 0))

            canvas.paste(panel, (x_off, y_off))

    return canvas


def main():
    print("=" * 60)
    print("  WIZUALNE POROWNANIE: final_v1 vs two-model")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Ladowanie obrazkow testowych...")
    all_samples = load_test_samples()
    print(f"  Dostepnych: {len(all_samples)}")

    selected = random.sample(all_samples, min(NUM_SAMPLES, len(all_samples)))
    print(f"  Wybrano: {len(selected)}")

    print("\n[2/5] Ladowanie final_v1...")
    model_final = YOLO(str(FINAL_V1))
    print(f"  task={model_final.task}")

    print("\n[3/5] Ladowanie walls_v1...")
    model_walls = YOLO(str(WALLS_V1))
    print(f"  task={model_walls.task}")

    print("\n[4/5] Ladowanie doors_windows_v2...")
    model_dw = YOLO(str(DW_V2))
    print(f"  task={model_dw.task}")

    print(f"\n[5/5] Generowanie {len(selected)} porownan (640 + 1024)...")
    for idx, img_path in enumerate(selected):
        print(f"  [{idx+1}/{len(selected)}] {img_path.name}", end="", flush=True)

        gt_boxes, gt_labels = load_gt_lbl(img_path)
        img = Image.open(img_path).convert("RGB")

        # Inferencja w 2 rozmiarach
        det_final_640 = run_inference(model_final, img, imgsz=640)
        det_walls_640 = run_inference(model_walls, img, conf=0.25, imgsz=640)
        det_dw_640 = run_inference(model_dw, img, conf=0.25, imgsz=640)

        det_final_1024 = run_inference(model_final, img, imgsz=1024)
        det_walls_1024 = run_inference(model_walls, img, conf=0.25, imgsz=1024)
        det_dw_1024 = run_inference(model_dw, img, conf=0.25, imgsz=1024)

        # remap
        for dets, remap in [(det_final_640, {0:0,1:1,2:2}), (det_final_1024, {0:0,1:1,2:2}),
                            (det_walls_640, {}), (det_walls_1024, {}),
                            (det_dw_640, {0:1,1:2}), (det_dw_1024, {0:1,1:2})]:
            for d in dets:
                if remap:
                    d["class"] = remap.get(d["class"], d["class"])
                else:
                    d["class"] = 0

        canvas = make_comparison(img_path, gt_boxes, gt_labels,
                                 det_final_640, det_walls_640, det_dw_640,
                                 det_final_1024, det_walls_1024, det_dw_1024)
        out_path = OUTPUT_DIR / f"compare_{idx:03d}_{img_path.stem}.jpg"
        canvas.save(out_path, quality=90)
        print(f" -> ok")

    # cleanup
    del model_final, model_walls, model_dw
    torch.cuda.empty_cache()

    print(f"\nWyniki: {OUTPUT_DIR}/")
    print("Gotowe.")


if __name__ == "__main__":
    import torch
    main()
