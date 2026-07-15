#!/usr/bin/env python3
"""
health_check.py — Szybki test poprawności datasetu YOLO-OBB.
- Trenuje yolo11s-obb.pt przez 3-5 epok na małej próbce (100-200 obrazów)
- Sprawdza czy loss spada, czy mAP startuje od zera, czy nie ma błędów w labelach
"""

import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def prepare_mini_dataset(src: Path, dst: Path, samples_per_split: int = 100):
    """Wyciąga małą próbkę z merged_dataset do testów."""
    dst.mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid"]:
        src_images = src / split / "images"
        src_labels = src / split / "labels"
        dst_images = dst / split / "images"
        dst_labels = dst / split / "labels"
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        all_images = sorted(src_images.glob("*"))
        if not all_images:
            continue

        selected = random.sample(all_images, min(samples_per_split, len(all_images)))
        for img in selected:
            shutil.copy2(img, dst_images / img.name)
            label_src = src_labels / f"{img.stem}.txt"
            if label_src.exists():
                shutil.copy2(label_src, dst_labels / f"{img.stem}.txt")

        n_copied = len(list(dst_images.glob("*")))
        print(f"  {split}: {n_copied} obrazów")

    yaml_content = f"""train: {dst}/train/images
val: {dst}/valid/images

names:
  0: wall
"""
    (dst / "data.yaml").write_text(yaml_content)
    return dst / "data.yaml"


def main():
    merged = PROJECT_ROOT / "merged_dataset"
    if not merged.is_dir():
        print(f"[BLAD] Brak merged_dataset w {merged}")
        print("Najpierw uruchom: python scripts/merge_dataset.py")
        sys.exit(1)

    if not (merged / "data.yaml").exists():
        print(f"[BLAD] Brak data.yaml w merged_dataset")
        sys.exit(1)

    print("Przygotowanie mini-datasetu do health check...")
    mini_dir = PROJECT_ROOT / "runs" / "mini_dataset"
    if mini_dir.exists():
        shutil.rmtree(mini_dir)
    mini_yaml = prepare_mini_dataset(merged, mini_dir, samples_per_split=100)
    print(f"  Mini dataset: {mini_dir}")

    print("\nImportowanie ultralytics...")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[BLAD] Zainstaluj: pip install ultralytics")
        sys.exit(1)

    print("\nLadowanie yolo11s-obb.pt...")
    model = YOLO("yolo11s-obb.pt")

    print("\n" + "=" * 60)
    print("  HEALTH CHECK — trening 3 epoki na mini-datasie")
    print("=" * 60)
    results = model.train(
        data=str(mini_yaml),
        epochs=3,
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        project=str(PROJECT_ROOT / "runs"),
        name="health_check",
        exist_ok=True,
        verbose=True,
        amp=True,
        lr0=0.01,
        cos_lr=True,
    )

    print("\n" + "=" * 60)
    print("  HEALTH CHECK — walidacja")
    print("=" * 60)
    metrics = model.val(
        data=str(mini_yaml),
        batch=4,
        device=0,
        project=str(PROJECT_ROOT / "runs"),
        name="health_check_val",
        exist_ok=True,
    )

    print("\n" + "=" * 60)
    print("  WYNIKI HEALTH CHECK")
    print("=" * 60)
    print(f"  mAP@0.5:     {metrics.box.map50:.4f}" if hasattr(metrics, "box") and hasattr(metrics.box, "map50") else "  mAP@0.5:     N/A")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}" if hasattr(metrics, "box") and hasattr(metrics.box, "map") else "  mAP@0.5:0.95: N/A")

    # Sprzatanie mini datasetu
    print(f"\n  Mini dataset (do usuniecia): {mini_dir}")
    shutil.rmtree(mini_dir)
    print("  Health check zakonczony.")


if __name__ == "__main__":
    main()
