#!/usr/bin/env python3
"""Test one training step with ultralytics YOLO."""
from ultralytics import YOLO

model = YOLO("yolo11s.pt")
model.train(
    data=str(__import__("pathlib").Path.home() / "data" / "merged_doors_windows" / "data.yaml"),
    epochs=1,
    imgsz=640,
    batch=8,
    device=0,
    workers=0,
    amp=True,
    exist_ok=True,
    name="test_single_step",
    project=str(__import__("pathlib").Path.home() / "projects" / "trening" / "runs"),
)
print("Training completed successfully!")
