# Floor Plan YOLO Training

Automatyczna detekcja elementów rzutów architektonicznych: **ściany (wall), drzwi (door), okna (window)**.

Model: **YOLO11s** (detect) — 3 klasy, trenowany na zbiorze `combined_dataset` (~13.5k obrazów, ~517k instancji).

## Wymagania

- Linux (WSL2) z NVIDIA GPU + CUDA 12.4
- Python 3.11
- uv (menadżer pakietów)
- ~5 GB wolnego miejsca na torch + dane

## Szybki start

```bash
# 1. Instalacja zależności
cd ~/projects/trening
uv sync

# 2. Kopiuj dane (combined_dataset.zip) do ~/data/ i rozpakuj
unzip combined_dataset.zip -d ~/data/

# 3. Uruchom trening
uv run python scripts/train_final.py
```

## Struktura repozytorium

```
├── scripts/                    # Pipeline treningowy
│   ├── train_final.py          # Trening główny (3 klasy)
│   ├── train_final_v2.py       # Fine-tune final_v1 (+50 epoch)
│   ├── train_doors_windows.py  # Model pośredni (drzwi+okna)
│   ├── pseudo_label_walls.py   # Pseudo-labeling
│   ├── flatten_pseudo.py       # Spłaszczenie struktury
│   ├── filter_predictions.py   # Filtracja pseudo-etykiet
│   └── merge_final.py          # Merge → combined_dataset
├── experiments/
│   ├── two_model/              # Podejście dwumodelowe
│   └── compare/                # Ewaluacja + wizualizacja porównawcza
├── weights/                    # Wytrenowane wagi (na GitHub)
├── runs/                       # Logi treningów (gitignored)
├── .env.example                # Konfiguracja W&B
├── pyproject.toml              # Zależności
└── uv.lock
```

## Modele

Wszystkie wagi dostępne w `weights/` na GitHub.

| Model | Opis | mAP50 (wall) | mAP50 (door) | mAP50 (window) |
|-------|------|:---:|:---:|:---:|
| **final_v1** | 3 klasy, 50 epoch, imgsz=1024 | **77.1%** | **80.0%** | **88.1%** |
| final_v2 | Fine-tune final_v1 +50 epoch | 78.2% | 79.0% | 88.3% |
| raw_3class | 3 klasy, surowe dane (ablation) | 75.1% | 84.8% | 89.3% |
| doors_windows_v2 | 2 klasy: door, window (pośredni) | — | — | — |
| walls_v1 | 1 klasa: wall (dwumodelowy) | 76.6% | — | — |

final_v1 rekomendowany jako domyślny.