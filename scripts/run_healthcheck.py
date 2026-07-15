#!/usr/bin/env python3
import sys
from pathlib import Path
from ultralytics import YOLO

home = Path.home()
project_dir = home / "projects" / "trening"
data_yaml = project_dir / "runs" / "mini_dataset" / "data.yaml"

if not data_yaml.exists():
    print(f"ERROR: {data_yaml} not found", file=sys.stderr)
    sys.exit(1)

model = YOLO("yolo11s-obb.pt")
model.train(
    data=str(data_yaml),
    epochs=5,
    imgsz=320,
    batch=2,
    device=0,
    workers=0,
    project=str(project_dir / "runs"),
    name="hc_fast",
    exist_ok=True,
    verbose=True,
)
print("TRAINING COMPLETED SUCCESSFULLY")
