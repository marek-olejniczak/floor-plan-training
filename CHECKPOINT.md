# Checkpoint — trening YOLO floor plan detection
> 15.07.2026 — Po Etapie B (filtry zatwierdzone)

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

## Wizualizacje (do przegladu)
- `threshold_review.png` — area P85/P90/P95
- `overlap_review.png` — overlap 0.2/0.3/0.35
- `conf_review.png` — conf 0.3/0.4/0.5 dla okien
- `doors_review.png` — conf 0.3/0.4/0.5 dla drzwi + door/dup

## Etap C — do zrobienia
1. Merge: corrected_walls + walls_doors_windows → combined_dataset
2. Merge: merged_doors_windows jako dodatkowe door+window
3. Generacja data.yaml (3 klasy: 0=wall, 1=door, 2=window)
4. Finalny trening YOLO11s (50 epoch, 1024px, batch 16)
