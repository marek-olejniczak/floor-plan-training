#!/usr/bin/env python3
"""Quick test: does torch 2.13.0 survive a full forward+backward pass?"""
from ultralytics import YOLO
import torch

print("Loading model...")
model = YOLO("yolo11s.pt")
model.model.train()
model.model.cuda()

x = torch.randn(2, 3, 256, 256).cuda()
print("Forward pass...")
out = model.model(x)
print("Forward OK, output shapes:", [o.shape for o in out])

loss = sum(o.sum() for o in out)
print("Backward pass...")
loss.backward()
print("Backward OK!")
print("All tests passed!")
