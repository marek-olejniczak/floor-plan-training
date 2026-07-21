#!/usr/bin/env python3
"""
merge_dataset.py — Scala 3 datasety YOLOv8-OBB w jeden zbiorczy dataset.
- Zachowuje oryginalne splity (train/valid/test)
- Wyciąga tylko klasę 'wall' (z cubicasa5k2 remap class 1→0)
- Dodaje prefiksy do nazw plików (zapobiega kolizjom)
"""

import shutil
from pathlib import Path

# --- Konfiguracja ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_SOURCES = {
    "walldetection": {
        "path": PROJECT_ROOT / "datasetYOLOv8obb" / "walldetection",
        "class_map": {0: 0},
    },
    "archvision": {
        "path": PROJECT_ROOT / "datasetYOLOv8obb" / "archvision",
        "class_map": {0: 0},
    },
    "cubicasa5k2": {
        "path": PROJECT_ROOT / "datasetYOLOv8obb" / "cubicasa5k2",
        "class_map": {1: 0},
    },
    "cubicasa_clean": {
        "path": Path("/mnt/d/rzuty/cubicasa/cubicasa5k_clean_yolo"),
        "class_map": {0: 0},
        "split_name_map": {"train": "train", "valid": "val", "test": "test"},
        "inverted_structure": True,
    },
}

TARGET = PROJECT_ROOT / "merged_dataset"
SPLITS = ["train", "valid", "test"]


def process_labels(src_labels_dir: Path, dst_labels_dir: Path, prefix: str, class_map: dict):
    """Kopiuje labele z remappingiem klas i filtrowaniem niepożądanych."""
    kept = 0
    skipped = 0
    for label_file in src_labels_dir.glob("*.txt"):
        dst_name = f"{prefix}_{label_file.name}"
        dst_path = dst_labels_dir / dst_name

        with open(label_file) as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            old_class = int(parts[0])
            if old_class in class_map:
                new_class = class_map[old_class]
                parts[0] = str(new_class)
                new_lines.append(" ".join(parts))
                kept += 1
            else:
                skipped += 1

        if new_lines:
            dst_path.write_text("\n".join(new_lines) + "\n")
        else:
            dst_path.write_text("")

    return kept, skipped


def copy_images(src_images_dir: Path, dst_images_dir: Path, prefix: str):
    """Kopiuje obrazy z dodaniem prefiksu."""
    count = 0
    for img_file in src_images_dir.glob("*"):
        if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        dst_name = f"{prefix}_{img_file.name}"
        shutil.copy2(img_file, dst_images_dir / dst_name)
        count += 1
    return count


def validate_pairing(dataset_name: str, images_dir: Path, labels_dir: Path, split: str):
    """Sprawdza czy każdy obraz ma label i vice versa."""
    img_stems = {p.stem for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}}
    label_stems = {p.stem for p in labels_dir.glob("*.txt")}
    missing_labels = img_stems - label_stems
    missing_images = label_stems - img_stems
    if missing_labels:
        print(f"  [UWAGA] {dataset_name}/{split}: {len(missing_labels)} obrazów bez labela")
    if missing_images:
        print(f"  [UWAGA] {dataset_name}/{split}: {len(missing_images)} labeli bez obrazu")
    return len(missing_labels), len(missing_images)


def main():
    total_images = 0
    total_labels = 0
    total_skipped = 0

    for dataset_name, cfg in DATA_SOURCES.items():
        print(f"\n{'='*60}")
        print(f"  Przetwarzanie: {dataset_name}")
        print(f"{'='*60}")

        src_root = cfg["path"]
        class_map = cfg["class_map"]

        split_map = cfg.get("split_name_map", {})
        inverted = cfg.get("inverted_structure", False)

        for split in SPLITS:
            actual_split = split_map.get(split, split)

            if inverted:
                src_images = src_root / "images" / actual_split
                src_labels = src_root / "labels" / actual_split
            else:
                src_images = src_root / actual_split / "images"
                src_labels = src_root / actual_split / "labels"

            dst_images = TARGET / split / "images"
            dst_labels = TARGET / split / "labels"

            if not src_images.is_dir():
                print(f"  → {split}: brak folderu, pomijam")
                continue

            prefix = dataset_name
            n_img = copy_images(src_images, dst_images, prefix)
            n_kept, n_skip = process_labels(src_labels, dst_labels, prefix, class_map)
            n_lab_miss, n_img_miss = validate_pairing(dataset_name, dst_images, dst_labels, split)

            total_images += n_img
            total_labels += n_kept
            total_skipped += n_skip

            print(f"  → {split}: {n_img} obrazów, {n_kept} adnotacji (odrzucono {n_skip})")

    # --- Generowanie data.yaml ---
    yaml_content = f"""train: train/images
val: valid/images
test: test/images

names:
  0: wall
"""
    yaml_path = TARGET / "data.yaml"
    yaml_path.write_text(yaml_content)

    # --- Podsumowanie ---
    print(f"\n{'='*60}")
    print(f"  PODSUMOWANIE")
    print(f"{'='*60}")
    print(f"  W sumie skopiowano: {total_images} obrazów")
    print(f"  W sumie adnotacji:  {total_labels}")
    print(f"  Odrzuconych (nie-wall): {total_skipped}")
    print(f"  DataYAML:             {yaml_path}")
    print(f"  Dataset gotowy pod:   {TARGET}")


if __name__ == "__main__":
    main()
