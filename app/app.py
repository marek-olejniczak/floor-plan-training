#!/usr/bin/env python3
"""
app.py — Flask backend do inferencji YOLO11s (detect/obb) na rzutach.
Uruchomienie: python app/app.py  →  http://localhost:5000
Model: YOLO_MODEL=<sciezka> env var lub domyslnie app/models/best.pt
"""

import io
import os
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image

app = Flask(__name__)

# --- .env z katalogu projektu ---
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# --- Wczytywanie modelu ---
_DEFAULT_MODEL = Path(__file__).parent / "models" / "best.pt"
MODEL_PATH = Path(os.environ.get("YOLO_MODEL", str(_DEFAULT_MODEL)))
model = None


def load_model():
    global model
    if not MODEL_PATH.exists():
        print(f"[BLAD] Model nie istnieje: {MODEL_PATH}")
        return

    try:
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PATH))
        print(f"[OK] Model zaladowany: {MODEL_PATH} (task={model.task})")
    except Exception as e:
        print(f"[BLAD] Nie mozna zaladowac modelu: {e}")


load_model()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": f"Brak modelu — {MODEL_PATH}"}), 400

    if "file" not in request.files:
        return jsonify({"error": "Brak pliku"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Pusty plik"}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Nieobslugiwany format obrazu"}), 400

    img_w, img_h = image.size

    results = model.predict(
        source=np.array(image),
        imgsz=640,
        conf=0.15,
        iou=0.5,
        device=0,
        verbose=False,
    )[0]

    detections = []

    # --- OBB (oriented bounding boxes) ---
    if getattr(results, "obb", None) is not None:
        boxes = results.obb.xyxyxyxy.cpu().numpy()
        confs = results.obb.conf.cpu().numpy()
        cls_ids = results.obb.cls.cpu().numpy()
        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            vertices = [[float(x), float(y)] for x, y in box]
            detections.append({
                "vertices": vertices,
                "confidence": round(float(conf), 4),
                "class": results.names[int(cls_id)],
            })

    # --- Detekcja standardowa (axis-aligned boxes) ---
    elif getattr(results, "boxes", None) is not None and results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy()
        for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confs, cls_ids):
            vertices = [[float(x1), float(y1)],
                        [float(x2), float(y1)],
                        [float(x2), float(y2)],
                        [float(x1), float(y2)]]
            detections.append({
                "vertices": vertices,
                "confidence": round(float(conf), 4),
                "class": results.names[int(cls_id)],
            })

    return jsonify({
        "image_w": img_w,
        "image_h": img_h,
        "detections": detections,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
