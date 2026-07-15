#!/usr/bin/env python3
import os, sys
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("step 1: importing...")
from pathlib import Path
from ultralytics import YOLO

print("step 2: loading model...")
model = YOLO(str(Path.home() / "projects" / "trening" / "runs" / "doors_windows_v1" / "weights" / "best.pt"))
print("step 3: model loaded")

print("step 4: testing inference...")
results = model("/mnt/d/rzuty/dane/yolo11datasets/walls/d1/train/images/00c2f93fea030183b53673748cbafc67-1-_jpeg_jpg.rf.526634ee4a594a6168e67155cfe5faaa.jpg", conf=0.5, verbose=False)
print(f"step 5: inference done, {len(results)} results")
if results[0].boxes is not None:
    print(f"  found {len(results[0].boxes)} boxes")
print("DONE")
