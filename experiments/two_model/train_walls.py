#!/usr/bin/env python3
"""
train_walls.py — Trening YOLO11s na scianach (1 klasa, detect).
Wszystkie metryki treningowe i walidacyjne logowane do W&B.
"""

import os
from pathlib import Path

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"
os.environ["WANDB_SILENT"] = "false"

# --- laduj .env ---
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from ultralytics import YOLO

DATA = Path.home() / "data" / "walls_only" / "data.yaml"
PROJECT = Path(__file__).resolve().parent / "runs"
RUN_NAME = "walls_v1"

EPOCHS = 30
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
CLOSE_MOSAIC = 10
FLIPLR = 0.5
FLIPUD = 0.0
AMP = True
FRACTION = 1.0


def init_wandb():
    api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if not api_key:
        os.environ["WANDB_MODE"] = "disabled"
        print("[W&B] BRAK WANDB_API_KEY — logging wylaczony")
        return

    try:
        import wandb
        wandb.init(
            project="floor-plan-detection-walls",
            name=RUN_NAME,
            config={
                "approach": "two-model",
                "model": "walls_v1",
                "epochs": EPOCHS,
                "batch": BATCH,
                "imgsz": IMSZ,
                "optimizer": OPTIMIZER,
                "lr0": LR0,
                "cos_lr": COS_LR,
            },
        )
        if wandb.run is not None:
            print(f"[W&B] OK — projekt=floor-plan-detection-walls, run={RUN_NAME}")
        else:
            os.environ["WANDB_MODE"] = "disabled"
            print("[W&B] wandb.init() zwrocilo None — logging wylaczony")
    except Exception as e:
        os.environ["WANDB_MODE"] = "disabled"
        print(f"[W&B] BLAD inicjalizacji: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("=" * 60)
    print("  YOLO11s — SCIANY (1 klasa: wall)")
    print("=" * 60)

    if not DATA.exists():
        print(f"[BLAD] Brak data.yaml: {DATA}")
        sys.exit(1)

    init_wandb()

    model = YOLO("yolo11s.pt")

    results = model.train(
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

    # Weryfikacja metryk po treningu
    print("\n[WYNIKI TRENINGU]")
    for key in ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
        val = getattr(results, key.split("/")[-1].split("(")[0], None)
        if val is not None:
            print(f"  {key}: {val:.4f}")

    # Zakoncz run W&B
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass

    model_path = PROJECT / RUN_NAME / "weights" / "best.pt"
    print(f"\nModel zapisany: {model_path}")


if __name__ == "__main__":
    import sys
    main()
