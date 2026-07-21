# Checkpoint — trening YOLO floor plan detection
> 21.07.2026 — Po oczyszczeniu d1 (CubiCasa)

## Ustawione progi (finalne)
| Filtr | Threshold | Opis |
|---|---|---|
| Confidence | ≥ 0.4 | Dla drzwi i okien |
| Window area | P90 = 0.0869 | Okno >8.7% obrazu = room |
| Wall overlap | ≥ 0.3 | Okno bez sciany w tle = falszywe |
| Door-win IoU | > 0.2 | Okno nakladajace sie na drzwi → usun |
| Door dedup | cover ≥ 0.9 | Z dwoch identycznych drzwi → mniejsze wygrywa |

## Pipeline
```
predict_walls.py   → raw_predictions/     (12 687 img, 94 956 okien, 119 393 drzwi)
filter_predictions.py → corrected_walls/  (80 314 drzwi, 46 668 okien po conf=0.4)
```

## Oczyszczenie d1 (CubiCasa) — okna
Niektóre balkony/tarasy w d1 były oznaczone w całości jako okno (class 2). Filtracja po area + aspect ratio:
| Parametr | Wartość |
|---|---|
| Min area | > 0.0201 (P99.74) |
| Max aspect ratio | ≤ 4.5 (długie/smukłe okna pomijane) |
| Usunięte | 91 okien (0.23%) z 39 696 |
| Zachowane pomimo dużej powierzchni | okna o AR > 4.5 (wąskie/ribbon) |

Skrypt: `filter_d1_windows.py --threshold 0.0201 --max-aspect 4.5`

## Wyniki filtracji (ostatni run)
- Low conf (<0.4): 87 367
- Area: 0
- Overlap: 2 037
- By door: 147
- Door dup: 454
- Kept: 124 360

## Struktura projektu (scripts/)
| Skrypt | Rola |
|---|---|
| `predict_walls.py` | Inferencja YOLO + rotacja → raw predictions |
| `filter_predictions.py` | Filtrowanie (conf/area/overlap/door/dedup) → cleaned dataset |
| `train_doors_windows.py` | Trening modelu YOLO |
| `merge_doors_windows.py` | Scalanie d1+d2+d3 |
| `analyze_d1_windows.py` | Analiza rozkładu powierzchni okien w d1 |
| `filter_d1_windows.py` | Filtracja okien d1 po area + aspect ratio |
| `visualize_dataset.py` | Siatka 16 obrazów z adnotacjami (sciany jako poligony/bbox) |
| `visualize_dataset_aabb.py` | Siatka 16 obrazów, sciany jako AABB |

## Wizualizacje (do przegladu)
- `threshold_review.png` — area P85/P90/P95
- `overlap_review.png` — overlap 0.2/0.3/0.35
- `conf_review.png` — conf 0.3/0.4/0.5 dla okien
- `doors_review.png` — conf 0.3/0.4/0.5 dla drzwi + door/dup
- `dataset_review.png` — przegląd corrected_walls z polygonami scian
- `dataset_review_aabb.png` — jw. ale sciany jako AABB
- `d1_window_filter_thr=0_0201_ar4.5.png` — efekt filtracji okien d1

## Etap C — do zrobienia
1. Merge: corrected_walls + walls_doors_windows → combined_dataset
2. Merge: merged_doors_windows jako dodatkowe door+window
3. Generacja data.yaml (3 klasy: 0=wall, 1=door, 2=window)
4. Finalny trening YOLO11s (50 epoch, 1024px, batch 16)
