# Checkpoint — trening YOLO floor plan detection
> 21.07.2026 — Full pipeline odświeżony, gotowy do treningu

## Ustawione progi (finalne)
| Filtr | Threshold | Opis |
|---|---|---|
| Confidence | ≥ 0.4 | Dla drzwi i okien |
| Window area | P90 = 0.0869 | Okno >8.7% obrazu = room |
| Wall overlap | ≥ 0.3 | Okno bez sciany w tle = falszywe |
| Door-win IoU | > 0.2 | Okno nakladajace sie na drzwi → usun |
| Door dedup | cover ≥ 0.9 | Z dwoch identycznych drzwi → mniejsze wygrywa |

## Pipeline (obecny — bez rotacji)
```
pseudo_label_walls.py          → pseudo_labeled_walls/  (12 687 img, walls + pseudo DW)
flatten_pseudo.py               → raw_predictions/       (flat structure, d1_/d2_ prefix)
filter_predictions.py           → corrected_walls/       (105 946 kept obiektów)
merge_final.py                  → combined_dataset/      (18 662 img, 489k scian, 133k drzwi, 78k okien)
train_final.py                  → runs/final_v1/         (YOLO11s, 50 epoch, 1024px, W&B optional)
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

## Wyniki pseudo-labelingu + filtracji (ostatni run)
```
pseudo_label_walls.py:
  Obrazy:        12 687 (d1: 6041, d2: 6646)
  Drzwi:         76 272
  Okna:          31 545

filter_predictions.py (conf=0.4, area=P90, overlap=0.3, door_iou=0.2, dedup=0.9):
  Sciany:        320 148 (bez filtracji)
  Drzwi+okna:    107 817 → 105 946 kept (1 928 usunięte)
  Low conf:      0 (wszystkie >= 0.4 z modelu)
  Area:          0
  Overlap:       924
  By door:       22
  Door dup:      982
```

## merged dataset
| Split | Images | Wall | Door | Window |
|-------|-------:|-----:|-----:|-------:|
| train | 13 963 | 357 762 | 98 702 | 57 560 |
| valid | 3 255 | 79 189 | 20 471 | 11 427 |
| test | 1 444 | 48 380 | 12 505 | 9 155 |
| **Total** | **18 662** | **489 063** | **133 678** | **78 142** |

## Struktura projektu (scripts/)
| Skrypt | Rola |
|---|---|
| `pseudo_label_walls.py` | Pseudo-labeling drzwi/okien na walls d1+d2 (z conf) |
| `flatten_pseudo.py` | Spłaszczenie d1/d2 w jeden katalog raw_predictions/ |
| `filter_predictions.py` | Filtracja (conf/area/overlap/door/dedup) |
| `merge_final.py` | Merge corrected_walls + walls_doors_windows → combined |
| `train_final.py` | Trening YOLO11s (3 klasy) z opcjonalnym W&B |
| `train_doors_windows.py` | Trening modelu DW (do pseudo-labelingu) |
| `filter_d1_windows.py` | Filtracja okien d1 po area + aspect ratio |
| `analyze_d1_windows.py` | Analiza rozkładu powierzchni okien w d1 |
| `merge_doors_windows.py` | Scalanie d1+d2+d3 dla DW |

## Etap C — do zrobienia
1. Odpalić `train_final.py` (50 epoch, 1024px, batch 16)
2. Ewaluacja modelu na zbiorze testowym
3. (opcjonalnie) Trening porównawczy na samym walls_doors_windows
