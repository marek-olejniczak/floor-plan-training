#!/usr/bin/env python3
"""
app.py — Flask backend do inferencji YOLO11s-OBB na rzutach architektonicznych.
Uruchomienie: python app/app.py  →  http://localhost:5000
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image

app = Flask(__name__)

# --- Wczytywanie modelu ---
MODEL_PATH = Path(__file__).parent / "models" / "best.pt"
model = None


def load_model():
    global model
    if not MODEL_PATH.exists():
        return

    try:
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PATH))
        print(f"[OK] Model zaladowany: {MODEL_PATH}")
    except Exception as e:
        print(f"[BLAD] Nie mozna zaladowac modelu: {e}")


load_model()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Brak modelu — skopiuj best.pt do app/models/"}), 400

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
        imgsz=640, # bylo 640
        conf=0.15,
        iou=0.5,
        device=0,
        verbose=False,
    )[0]

    detections = []
    if results.obb is not None:
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

    return jsonify({
        "image_w": img_w,
        "image_h": img_h,
        "detections": detections,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
