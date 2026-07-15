#!/usr/bin/env python3
"""Test one training step with ultralytics YOLO on torch 2.0.1."""
import os
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

from pathlib import Path
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
model.train(
    data=str(Path.home() / "data" / "merged_doors_windows" / "data.yaml"),
    epochs=1,
    imgsz=640,
    batch=4,
    device=0,
    workers=0,
    amp=True,
    exist_ok=True,
    name="test_torch201",
    project=str(Path.home() / "projects" / "trening" / "runs"),
    fraction=0.02,
    verbose=True,
)
print("TRAINING COMPLETED SUCCESSFULLY!")
