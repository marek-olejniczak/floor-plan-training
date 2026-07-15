# Action Plan — pozostałe etapy

## Obecny stan: ŚRODEK Etapu B

### Etap A — ZAKOŃCZONY
- [x] Poprawiony merge d1+d2+d3 (merged_doors_windows, 2678 img)
- [x] Trening YOLO11s 30 epok (doors_windows_v2, mAP50=0.927)
- [x] Angle-aware pseudo-labeling (corrected_walls, 12687 img, 26% rotated)

### Etap B — W TRAKCIE (naprawa fałszywych okien)

#### Krok 1: Analiza rozkładu powierzchni okien ✅
- [x] Przeczytać walls_doors_windows (d1,d2) — tylko klasa window
- [x] Obliczyć area = w*h dla każdego okna
- [x] Statystyki: percentyle 90, 95, 99
- [x] Ustalić threshold: P95 × 1.5 = 0.356
- [x] Skrypt: scripts/analyze_window_area.py

#### Krok 2: Filtr overlapu ze ścianą (70%) ✅
- [x] Dla każdego pseudo-window: obliczyć overlap z klasą 0 (wall)
- [x] Jeśli overlap < 0.7 → odrzucić detekcję
- [x] Dla ścian w formacie wielokąta: użyć bounding boxa

#### Krok 3: Confidence w wizualizacji ✅
- [x] Przy każdej detekcji pokazać conf score
- [x] Pozwoli dobrać próg conf dla window

#### Krok 4: Implementacja w pseudo_label_rotated.py ✅
- [x] Dodać MAX_WINDOW_AREA = 0.356 (z analizy)
- [x] Dodać MIN_WALL_OVERLAP = 0.7
- [x] Post-processing po predykcji, przed zapisem
- [x] Zachować conf w labelach (6. pole) + metadata JSON

#### Krok 5: Re-run pseudo-labelingu
- Skasować stare corrected_walls
- Puścić pseudo_label_rotated.py z filtrami

#### Krok 6: Wizualizacja z conf scores
- 30 przykładów (15 rotated + 15 nie)
- Conf score na każdej detekcji
- Oznaczenie które by odpadły/przeszły filtry

### Etap C — Finalny dataset + trening (po akceptacji B)

1. Merge: corrected_walls (po filtrach) + walls_doors_windows → combined_dataset
2. Ew. dodać merged_doors_windows jako dodatkowe door+window
3. Generacja data.yaml (3 klasy)
4. Finalny trening YOLO11s (50 epoch, 1024px, batch 16)

## Komendy do uruchomienia w WSL
```bash
# Wejście do WSL
wsl

# Trening
cd ~/projects/trening
source .venv/bin/activate  # lub .venv/bin/python scripts/foo.py

# Analiza okien
.venv/bin/python scripts/analyze_window_area.py

# Pseudo-labeling z filtrami
rm -rf ~/data/corrected_walls
.venv/bin/python scripts/pseudo_label_rotated.py

# Wizualizacja
.venv/bin/python scripts/visualize_corrected.py