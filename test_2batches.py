#!/usr/bin/env python3
"""Test training for 2 batches only."""
from pathlib import Path
from ultralytics import YOLO
import os
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

model = YOLO("yolo11s.pt")
# Train just 2 batches by overriding epochs and fraction
model.train(
    data=str(Path.home() / "data" / "merged_doors_windows" / "data.yaml"),
    epochs=1,
    imgsz=640,
    batch=4,
    device=0,
    workers=0,
    amp=True,
    exist_ok=True,
    name="test_2batches",
    project=str(Path.home() / "projects" / "trening" / "runs"),
    fraction=0.01,  # only 1% of data = ~20 images = ~5 batches
    verbose=True,
)
print("TRAINING COMPLETED SUCCESSFULLY!")
