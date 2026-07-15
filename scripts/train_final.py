#!/usr/bin/env python3
"""
train_final.py — Finalny trening YOLO11s na 3 klasach (wall, door, window).
"""
import os
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

from pathlib import Path
from ultralytics import YOLO

DATA = Path.home() / "data" / "combined_dataset" / "data.yaml"
PROJECT = Path.home() / "projects" / "trening" / "runs"
RUN_NAME = "final_v1"

EPOCHS = 50
BATCH = 16
IMSZ = 1024
WORKERS = 0
DEVICE = 0
LR0 = 0.001667
COS_LR = True
OPTIMIZER = "AdamW"
WARMUP_EPOCHS = 3
PATIENCE = 25
MOSAIC = 1.0
CLOSE_MOSAIC = 15
FLIPLR = 0.5
FLIPUD = 0.0
AMP = True
FRACTION = 1.0


def main():
    print("=" * 60)
    print("  YOLO11s — FINALNY TRENING (3 klasy: wall, door, window)")
    print("=" * 60)

    if not DATA.exists():
        print(f"[BLAD] Brak data.yaml: {DATA}")
        return

    model = YOLO("yolo11s.pt")

    model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=IMSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        lr0=LR0,
        cos_lr=COS_LR,
        warmup_epochs=WARMUP_EPOCHS,
        patience=PATIENCE,
        optimizer=OPTIMIZER,
        fraction=FRACTION,
        mosaic=MOSAIC,
        close_mosaic=CLOSE_MOSAIC,
        fliplr=FLIPLR,
        flipud=FLIPUD,
        amp=AMP,
        project=str(PROJECT),
        name=RUN_NAME,
        exist_ok=True,
    )

    print(f"\nModel zapisany: {PROJECT / RUN_NAME / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
