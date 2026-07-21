#!/usr/bin/env python3
"""
train_full.py — Właściwy trening YOLO11m-OBB dla detekcji ścian z rzutów.

Uruchomienie (WSL):
    export POLARS_SKIP_CPU_CHECK=1
    source .venv/bin/activate
    python scripts/train_full.py

Monitorowanie (TensorBoard):
    tensorboard --logdir runs/train_full --port 6006
    → http://localhost:6006
"""

# ============================================================================
# KONFIGURACJA
# ============================================================================

# --- Podstawowe ścieżki ---
DATA_YAML = "merged_dataset/data.yaml"     # ścieżka względna do data.yaml
PROJECT_DIR = "runs"                        # katalog na wyniki (względna — resolvuje do ~/projects/trening)
RUN_NAME = "train_full"                     # nazwa tego uruchomienia

# --- Parametry treningu ---
EPOCHS = 50             # 50 epok × ~1.5h = ~3 dni (m-model 1024px, workers=0)
BATCH = 16              # yolo11s ma 3× mniej parametrow niz m, batch=24 zmiesci sie w 16 GB
IMSZ = 1024             # 1024px — w pelni wykorzystujemy rozdzielczosc datasetu
WORKERS = 1             # 0 = brak pin_memory thread (Xeon bez AVX2 crashuje na workers>0)
DEVICE = 0              # GPU ID (0=first GPU, 'cpu' = CPU)

# --- Learning rate i optymalizacja ---
LR0 = 0.005             # niższy LR przy wyższej rozdzielczości (więcej detali)
COS_LR = True           # cosine annealing schedule (łagodne zmniejszanie LR)
OPTIMIZER = "AdamW"     # explicite! auto wybiera MuSGD który daje gorsze wyniki
WARMUP_EPOCHS = 3       # epoki warmupu (LR rośnie od 0 do LR0)
PATIENCE = 30           # early stopping — brak poprawy przez ~30 epok (~2 dni)

# --- Augmentacje danych ---
MOSAIC = 1.0            # mosaic: 1.0 = zawsze, 0.0 = wyłączone
CLOSE_MOSAIC = 15       # wyłącz mosaic po N epoch (dłużej z mosaicą = lepsza generalizacja na 1024px)
MIXUP = 0.0             # mixup: 0.1-0.2 pomaga na generalizację
COPY_PASTE = 0.0         # copy-paste augmentacja obiektów
DEGREES = 45.0           # rotacja w stopniach (np. 10.0 = ±10°)
TRANSLATE = 0.1         # translacja (ułamek obrazu, np. 0.1 = ±10%)
SCALE = 0.5             # skala (0.5 = 50-150% oryginału)
SHEAR = 0.0             # ścinanie w stopniach
PERSPECTIVE = 0.0       # deformacja perspektywy
FLIPUD = 0.5            # flip pionowy (prawdopodobieństwo)
FLIPLR = 0.5            # flip poziomy (50% szans) — bezpieczne dla rzutów
HSV_H = 0.0             # augmentacja odcienia (H)
HSV_S = 0.0             # augmentacja nasycenia (S)
HSV_V = 0.4             # augmentacja jasności (V)

# --- Zapis i walidacja ---
SAVE = True             # zapisuje checkpointy
SAVE_PERIOD = 10        # zapis co N epok (dodatkowo do best/last)
VAL = True              # walidacja po każdej epoce
VERBOSE = True          # szczegółowe logi
AMP = True              # Automatic Mixed Precision (szybciej, mniej VRAM)
DETERMINISTIC = False   # deterministyczne wyniki (True = wolniej, powtarzalne)

# ---------------------------------------------------------------------------
# KONIEC KONFIGURACJI — poniżej kod wykonawczy
# ---------------------------------------------------------------------------

import signal
from pathlib import Path
import sys


graceful_exit = False


def _ctrl_c_handler(sig, frame):
    global graceful_exit
    if graceful_exit:
        sys.exit(1)
    graceful_exit = True
    print("\n[!] Ctrl+C — konczenie biezacej epoki i zapis checkpointu...")
    print("[!] Nacisnij Ctrl+C ponownie aby wymusic zamkniecie")
    raise KeyboardInterrupt()


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

def main():
    print("=" * 64)
    print("  YOLO11m-OBB — Trening detekcji ścian z rzutów")
    print("=" * 64)

    # --- Rozwiazywanie ścieżek (na wypadek gdyby settings.yaml nadpisywał) ---
    data_path = PROJECT_ROOT / DATA_YAML
    project_dir_abs = str(PROJECT_ROOT / PROJECT_DIR)

    if not data_path.exists():
        print(f"[BLAD] Nie znaleziono {DATA_YAML}")
        print(f"  Szukano w: {data_path}")
        print("  Uruchom najpierw: python scripts/merge_dataset.py")
        sys.exit(1)

    # --- Import ultralytics ---
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[BLAD] Brak ultralytics. Zainstaluj: uv pip install ultralytics")
        sys.exit(1)

    # --- Podsumowanie konfiguracji ---
    print(f"\n  Konfiguracja:")
    print(f"  ├─ Data:     {DATA_YAML}")
    print(f"  ├─ Epochs:   {EPOCHS}")
    print(f"  ├─ Batch:    {BATCH}")
    print(f"  ├─ Imgsz:    {IMSZ}")
    print(f"  ├─ Workers:  {WORKERS}")
    print(f"  ├─ Device:   {DEVICE}")
    print(f"  ├─ LR:       {LR0}")
    print(f"  ├─ Optim:    {OPTIMIZER}")
    print(f"  ├─ Patience: {PATIENCE}")
    print(f"  ├─ Mosaic:   {MOSAIC} (close at epoch {CLOSE_MOSAIC})")
    print(f"  └─ AMP:      {AMP}")

    # --- Wczytanie modelu ---
    print(f"\n  Ladowanie yolo11s-obb.pt (9.4M params, 21.5 GFLOPs)...")
    model = YOLO("yolo11s-obb.pt")

    # --- Rejestracja handlera Ctrl+C ---
    signal.signal(signal.SIGINT, _ctrl_c_handler)

    # --- Trening ---
    print(f"\n{'=' * 64}")
    print(f"  START TRENINGU")
    print(f"  Szacowany czas: {EPOCHS} epok x ~40min = ~{EPOCHS * 40 // 60}h @ 1024px (yolo11s, batch=24)")
    print(f"{'=' * 64}")

    model.train(
        # Podstawowe
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        # Optymalizacja
        lr0=LR0,
        cos_lr=COS_LR,
        warmup_epochs=WARMUP_EPOCHS,
        patience=PATIENCE,
        optimizer=OPTIMIZER,
        weight_decay=0.0005,
        momentum=0.937,
        # Augmentacje
        mosaic=MOSAIC,
        close_mosaic=CLOSE_MOSAIC,
        mixup=MIXUP,
        copy_paste=COPY_PASTE,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        shear=SHEAR,
        perspective=PERSPECTIVE,
        flipud=FLIPUD,
        fliplr=FLIPLR,
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        # Zapis i logowanie
        save=SAVE,
        save_period=SAVE_PERIOD,
        val=VAL,
        verbose=VERBOSE,
        amp=AMP,
        deterministic=DETERMINISTIC,
        project=project_dir_abs,
        name=RUN_NAME,
        exist_ok=True,
    )

    print(f"\n{'=' * 64}")
    print(f"  TRENING ZAKONCZONY")
    print(f"  Wyniki: {project_dir_abs}/{RUN_NAME}/")
    print(f"  Wagi:   {project_dir_abs}/{RUN_NAME}/weights/best.pt")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
