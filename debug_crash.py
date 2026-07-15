#!/usr/bin/env python3
"""Debug: find which operation causes SIGILL."""
import os, sys
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

from pathlib import Path
import torch

# Test 1: basic torch ops
print("Test 1: basic torch ops...")
x = torch.randn(1, 3, 640, 640).cuda()
print("  tensor created")
conv = torch.nn.Conv2d(3, 16, 3).cuda()
y = conv(x)
print("  conv2d OK")
y.sum().backward()
print("  backward OK")

# Test 2: load ultralytics model
print("Test 2: loading model...")
from ultralytics import YOLO
model = YOLO("yolo11s.pt")
model.model.cuda()
print("  model loaded")

# Test 3: simple inference
print("Test 3: inference...")
results = model(x, verbose=False)
print(f"  inference OK, {len(results)} results")

# Test 4: load a real image
print("Test 4: load real image...")
import cv2
img_paths = list((Path.home() / "data" / "merged_doors_windows" / "train" / "images").glob("*.jpg"))
if img_paths:
    img = cv2.imread(str(img_paths[0]))
    print(f"  image shape: {img.shape}")
    # inference on real image
    results = model(img, verbose=False)
    print(f"  inference on real image OK")

print("\nALL TESTS PASSED!")
