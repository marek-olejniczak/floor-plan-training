#!/usr/bin/env python3
"""
evaluate_models.py — Porownanie dwoch podejsc na czystym
tekscie Roboflow (walls_doors_windows/d1/test + d2/valid).

Porownanie per-klasa: Precision, Recall, mAP50, mAP50-95.
Metryki liczone przez torchmetrics.detection.MeanAveragePrecision.
"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

from ultralytics import YOLO

# --- sciezki ---
HOME = Path.home()
SCRIPT_DIR = Path(__file__).resolve().parent

D1_TEST = HOME / "data" / "walls_doors_windows" / "d1" / "test"
D2_VALID = HOME / "data" / "walls_doors_windows" / "d2" / "valid"

FINAL_V1 = HOME / "projects" / "trening" / "runs" / "final_v1" / "weights" / "best.pt"
WALLS_V1 = SCRIPT_DIR / ".." / "two_model" / "runs" / "walls_v1" / "weights" / "best.pt"
DW_V2 = HOME / "projects" / "trening" / "runs" / "doors_windows_v2" / "weights" / "best.pt"

OUTPUT_DIR = SCRIPT_DIR / "runs" / "eval_results"

CONF = 0.001  # very low to get all predictions, NMS handles duplicates
IOU = 0.5
IMSZ_LIST = [640, 1024]
DEVICE = 0

# Common class space: wall=0, door=1, window=2
# d1/d2 GT: door=0, wall=1, window=2
GT_REMAP = {0: 1, 1: 0, 2: 2}

# torchmetrics
from torchmetrics.detection import MeanAveragePrecision


def load_test_set():
    """Zwraca liste (sciezka_obrazu, tensor_boxes, tensor_labels) dla d1/test + d2/valid."""
    samples = []

    sources = [
        ("d1_test", D1_TEST / "images", D1_TEST / "labels", 5),
        ("d2_valid", D2_VALID / "images", D2_VALID / "labels", 11),
    ]

    for src_name, img_dir, lbl_dir, expected_nf in sources:
        if not img_dir.exists() or not lbl_dir.exists():
            print(f"  [POMINIETO] {src_name}: brak katalogu")
            continue
        lbl_files = sorted(lbl_dir.glob("*.txt"))
        for lbl_path in lbl_files:
            stem = lbl_path.stem
            img_path = None
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                cand = img_dir / f"{stem}{ext}"
                if cand.exists():
                    img_path = cand
                    break
            if img_path is None:
                continue

            # wczytaj labele + obraz zeby poznac wymiary
            img_pil = Image.open(img_path)
            img_w, img_h = img_pil.size

            boxes = []
            labels = []
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    coords = list(map(float, parts[1:]))

                    # YOLO format (cx,cy,w,h) znormalizowane -> absolutne
                    if len(coords) == 4:
                        cx, cy, bw, bh = coords
                        x1 = (cx - bw / 2) * img_w
                        y1 = (cy - bh / 2) * img_h
                        x2 = (cx + bw / 2) * img_w
                        y2 = (cy + bh / 2) * img_h
                    elif len(coords) >= 4 and len(coords) % 2 == 0:
                        # OBB polygon znormalizowany -> bbox absolutny
                        pts = np.array(coords, dtype=float).reshape(-1, 2)
                        pts[:, 0] *= img_w
                        pts[:, 1] *= img_h
                        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
                        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
                    else:
                        continue

                    new_cls = GT_REMAP.get(cls, cls)
                    boxes.append([x1, y1, x2, y2])
                    labels.append(new_cls)

            if not boxes:
                continue

            samples.append({
                "img_path": str(img_path),
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
            })

    return samples


def run_approach(model, samples, class_remap, class_filter=None, imgsz=1024):
    """Wykonuje inferencje na probkach i zwraca (pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)."""
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_gt_boxes = []
    all_gt_labels = []

    for s in samples:
        img = Image.open(s["img_path"]).convert("RGB")
        results = model.predict(
            source=np.array(img),
            conf=CONF,
            iou=IOU,
            imgsz=imgsz,
            device=DEVICE,
            verbose=False,
        )[0]

        # ground truth
        gt_boxes = s["boxes"].clone()
        gt_labels = s["labels"].clone()

        # predykcje
        pred_boxes_list = []
        pred_scores_list = []
        pred_labels_list = []

        if getattr(results, "obb", None) is not None:
            boxes = results.obb.xyxyxyxy.cpu()
            confs = results.obb.conf.cpu()
            cls_ids = results.obb.cls.cpu()
            for box, conf, cid in zip(boxes, confs, cls_ids):
                pts = box.view(-1, 2)
                x1, y1 = pts[:, 0].min().item(), pts[:, 1].min().item()
                x2, y2 = pts[:, 0].max().item(), pts[:, 1].max().item()
                mapped = class_remap.get(int(cid), int(cid))
                if class_filter is not None and mapped not in class_filter:
                    continue
                pred_boxes_list.append([x1, y1, x2, y2])
                pred_scores_list.append(float(conf))
                pred_labels_list.append(mapped)

        elif getattr(results, "boxes", None) is not None and results.boxes is not None:
            boxes = results.boxes.xyxy.cpu()
            confs = results.boxes.conf.cpu()
            cls_ids = results.boxes.cls.cpu()
            for box, conf, cid in zip(boxes, confs, cls_ids):
                x1, y1, x2, y2 = box.tolist()
                mapped = class_remap.get(int(cid), int(cid))
                if class_filter is not None and mapped not in class_filter:
                    continue
                pred_boxes_list.append([x1, y1, x2, y2])
                pred_scores_list.append(float(conf))
                pred_labels_list.append(mapped)

        if class_filter is not None:
            # filtruj GT
            mask = torch.isin(gt_labels, torch.tensor(list(class_filter)))
            gt_boxes = gt_boxes[mask]
            gt_labels = gt_labels[mask]

        all_pred_boxes.append(torch.tensor(pred_boxes_list, dtype=torch.float32) if pred_boxes_list else torch.zeros((0, 4), dtype=torch.float32))
        all_pred_scores.append(torch.tensor(pred_scores_list, dtype=torch.float32) if pred_scores_list else torch.zeros((0,), dtype=torch.float32))
        all_pred_labels.append(torch.tensor(pred_labels_list, dtype=torch.int64) if pred_labels_list else torch.zeros((0,), dtype=torch.int64))
        all_gt_boxes.append(gt_boxes)
        all_gt_labels.append(gt_labels)

    return all_pred_boxes, all_pred_scores, all_pred_labels, all_gt_boxes, all_gt_labels


def compute_map(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels):
    metric = MeanAveragePrecision(iou_type="bbox", max_detection_thresholds=[1, 10, 100])
    metric.update(
        [{"boxes": b, "scores": s, "labels": l} for b, s, l in zip(pred_boxes, pred_scores, pred_labels)],
        [{"boxes": b, "labels": l} for b, l in zip(gt_boxes, gt_labels)],
    )
    result = metric.compute()
    return {
        "map50": float(result["map_50"]),
        "map75": float(result["map_75"]),
        "map": float(result["map"]),
        "mar_1": float(result["mar_1"]),
        "mar_10": float(result["mar_10"]),
        "mar_100": float(result["mar_100"]),
    }


def compute_map_per_class(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, num_classes=3):
    results = {}
    for cls_id in range(num_classes):
        pb = [b[p == cls_id] for b, p in zip(pred_boxes, pred_labels)]
        ps = [s[p == cls_id] for s, p in zip(pred_scores, pred_labels)]
        pl = [torch.full((b.shape[0],), cls_id, dtype=torch.int64) if b.dim() > 0 and b.shape[0] > 0 else torch.zeros((0,), dtype=torch.int64) for b in pb]
        gb = [b[l == cls_id] for b, l in zip(gt_boxes, gt_labels)]
        gl = [l[l == cls_id] for l in gt_labels]

        if all(len(b) == 0 for b in gb):
            results[cls_id] = {"map50": None, "map": None}
            continue

        results[cls_id] = compute_map(pb, ps, pl, gb, gl)
    return results


def print_table(all_results):
    class_names = {0: "wall", 1: "door", 2: "window"}
    metrics = ["mAP50", "mAP"]

    print("\n" + "=" * 95)
    print(f"{'POROWNANIE METRYK':^95}")
    print("=" * 95)
    print(f"{'Klasa':<8} {'imgsz':<6} {'Metryka':<8} {'final_v1':<16} {'walls_v1':<16} {'dw_v2':<16}")
    print("-" * 95)

        def fmt(r, key):
            v = r.get(key) if isinstance(r, dict) else None
            if v is None:
                return f"{'---':>9}"
            return f"{v*100:>7.2f}% "

    for cls_id in range(3):
        cname = class_names[cls_id]
        first = True
        for imgsz in [640, 1024]:
            rf = all_results[imgsz]["final"].get(cls_id, {})
            rw = all_results[imgsz]["walls"].get(cls_id, {})
            rd = all_results[imgsz]["dw"].get(cls_id, {})

            for m in metrics:
                label = f"{'':8}" if not first else f"{cname:<8}"
                imsz_label = f"{imgsz:<6}" if first or m == metrics[0] else f"{'':6}"
                row = f"{label} {imsz_label} {m:<8} {fmt(rf,m)}"
                if cls_id == 0:
                    row += f" {fmt(rw,m)} {'N/A':>9}"
                else:
                    row += f" {'N/A':>9} {fmt(rd,m)}"
                print(row)
                first = False

    print("=" * 95)
    print("final_v1 = 1 model (3 klasy), walls_v1 = tylko sciany, dw_v2 = drzwi+okna")
    print("Test set: walls_doors_windows/d1/test + d2/valid (oryginalne Roboflow)")
    print("=" * 95)


def evaluate_at_imgsz(model_final, model_walls, model_dw, samples, imgsz):
    print(f"\n--- imgsz={imgsz} ---")

    print(f"  final_v1...")
    pb_f, ps_f, pl_f, gb_f, gl_f = run_approach(
        model_final, samples, class_remap={0: 0, 1: 1, 2: 2}, imgsz=imgsz
    )

    print(f"  walls_v1...")
    pb_w, ps_w, pl_w, gb_w, gl_w = run_approach(
        model_walls, samples, class_remap={0: 0}, class_filter={0}, imgsz=imgsz
    )

    print(f"  dw_v2...")
    pb_d, ps_d, pl_d, gb_d, gl_d = run_approach(
        model_dw, samples, class_remap={0: 1, 1: 2}, class_filter={1, 2}, imgsz=imgsz
    )

    return {
        "final": compute_map_per_class(pb_f, ps_f, pl_f, gb_f, gl_f),
        "walls": compute_map_per_class(pb_w, ps_w, pl_w, gb_w, gl_w),
        "dw": compute_map_per_class(pb_d, ps_d, pl_d, gb_d, gl_d),
    }


def main():
    print("=" * 60)
    print("  EWALUACJA: final_v1 vs two-model (walls_v1 + dw_v2)")
    print("=" * 60)

    print("\n[1/5] Ladowanie test setu (czyste dane Roboflow)...")
    samples = load_test_set()
    print(f"  Zaladowano {len(samples)} obrazkow testowych")

    print("\n[2/5] Ladowanie modeli...")
    model_final = YOLO(str(FINAL_V1))
    model_walls = YOLO(str(WALLS_V1))
    model_dw = YOLO(str(DW_V2))
    print(f"  final_v1 task={model_final.task}")
    print(f"  walls_v1 task={model_walls.task}")
    print(f"  dw_v2 task={model_dw.task}")

    all_results = {}
    for imgsz in IMSZ_LIST:
        print(f"\n[3/5] Inferencja imgsz={imgsz}...")
        all_results[imgsz] = evaluate_at_imgsz(model_final, model_walls, model_dw, samples, imgsz)

    del model_final, model_walls, model_dw
    torch.cuda.empty_cache()

    print("\n[4/5] Obliczanie metryk...")

    print_table(all_results)

    print("\n[5/5] Zapis...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"num_samples": len(samples), "imgsz": {}}
    for imgsz in IMSZ_LIST:
        report["imgsz"][imgsz] = {
            "final": {str(k): v for k, v in all_results[imgsz]["final"].items()},
            "walls": {str(k): v for k, v in all_results[imgsz]["walls"].items()},
            "dw": {str(k): v for k, v in all_results[imgsz]["dw"].items()},
        }
    report_path = OUTPUT_DIR / "metrics.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Raport: {report_path}")


if __name__ == "__main__":
    main()
