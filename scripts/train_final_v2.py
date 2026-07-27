#!/usr/bin/env python3
"""
train_final_v2.py — Dotrenowanie final_v1 przez kolejne 50 epok.
Load:  final_v1/weights/best.pt
Dane:  combined_dataset/ (3 klasy: wall, door, window)
W&B:   floor-plan-detection / final_v2
"""

import os
import sys
from pathlib import Path

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from ultralytics import YOLO

DATA = Path.home() / "data" / "combined_dataset" / "data.yaml"
CHECKPOINT = Path.home() / "projects" / "trening" / "runs" / "final_v1" / "weights" / "best.pt"
PROJECT = Path.home() / "projects" / "trening" / "runs"
RUN_NAME = "final_v2"

EPOCHS = 50
BATCH = 16
IMSZ = 1024
WORKERS = 0
DEVICE = 0
LR0 = 0.0005
COS_LR = True
OPTIMIZER = "AdamW"
WARMUP_EPOCHS = 0
PATIENCE = 25
MOSAIC = 1.0
CLOSE_MOSAIC = 5
FLIPLR = 0.5
FLIPUD = 0.0
AMP = True
FRACTION = 1.0


def init_wandb():
    api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if not api_key:
        print("[W&B] BRAK WANDB_API_KEY — przerywam")
        sys.exit(1)

    import wandb
    wandb.init(
        project="floor-plan-detection",
        name=RUN_NAME,
        config={
            "approach": "fine-tune-final-v1",
            "base_model": "final_v1",
            "epochs": EPOCHS,
            "batch": BATCH,
            "imgsz": IMSZ,
            "lr0": LR0,
            "cos_lr": COS_LR,
            "optimizer": OPTIMIZER,
            "warmup_epochs": WARMUP_EPOCHS,
        },
    )

    if wandb.run is None:
        print("[W&B] wandb.init() zwrocil None — przerywam")
        sys.exit(1)

    print(f"[W&B] Dashboard: https://wandb.ai/marek-olejniczak-cad-projekt-cad-projekt-k-a/floor-plan-detection/runs/{wandb.run.id}")
    print(f"[W&B] Projekt: floor-plan-detection / Run: {RUN_NAME}")


def main():
    print("=" * 60)
    print("  YOLO11s — DOTRENOWANIE final_v1 (3 klasy: wall, door, window)")
    print(f"  Checkpoint: {CHECKPOINT}")
    print("=" * 60)

    if not DATA.exists():
        print(f"[BLAD] Brak data.yaml: {DATA}")
        sys.exit(1)

    if not CHECKPOINT.exists():
        print(f"[BLAD] Brak checkpointu: {CHECKPOINT}")
        sys.exit(1)

    init_wandb()

    model = YOLO(str(CHECKPOINT))

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

    # W&B finish
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass

    print(f"\nModel zapisany: {PROJECT / RUN_NAME / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
