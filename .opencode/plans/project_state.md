# Project State — trening (YOLO floor plan detection)
> Ostatnia aktualizacja: 14.07.2026

## Środowisko
- OS: Windows 11 + WSL (Ubuntu)
- CPU: Intel(R) CPU @ 2.40GHz — brak AVX2/FMA/MOVBE
- Python: 3.11.15 (venv: ~/projects/trening/.venv/)
- torch: 2.6.0+cu124, ultralytics: 8.4.95
- polars zastąpiony pandas shim
- PIP: `python -m ensurepip; python -m pip install`

## System plików
Windows D:\rzuty\trening\ = WSL /mnt/d/rzuty/trening/ (projekt)
Windows D:\rzuty\dane\    = WSL /mnt/d/rzuty/dane/    (surowe dane)
WSL ~/data/               = przetworzone datasety
WSL ~/projects/trening/   = /mnt/d/rzuty/trening/

## Datasety źródłowe (/mnt/d/rzuty/dane/yolo11datasets/)
doors_windows/d1  — 642 img, klasy: 2door,door,window — czysty
doors_windows/d2  — 1677 img, klasy +baywindow,window1..5 — czysty
doors_windows/d3  — 359 img, klasy: door,window — poprawiony (był room↔window)
walls/d1          — ~6041 img, tylko wall — do pseudo-labelowania
walls/d2          — ~6646 img, tylko wall — do pseudo-labelowania
walls_doors_windows/d1 — ~4578 img, 3 klasy GT
walls_doors_windows/d2 — ~997 img, 3 klasy GT

## Datasety przetworzone (~/data/)
merged_doors_windows/    — 2678 img, 12706 door, 7218 window
pseudo_labeled_walls/    — 12687 img, walls + pseudo door/window
corrected_walls/         — 12687 img, 3304 (26%) rotowanych korektą kąta
combined_dataset/        — (jeszcze nie zrobiony)

## Wytrenowane modele
runs/doors_windows_v2/weights/best.pt — mAP50=0.927 (door:0.985, window:0.869)
Używany do pseudo-labelingu.

## Mapowanie klas (wewnętrznie: 0=wall, 1=door, 2=window)
doors_windows d1: {0→0, 1→0, 2→1}
doors_windows d2: {0→0, 1→1, 2→0, 3..7→1}
doors_windows d3: {0→0, 1→1}
walls_doors_windows: {0→1, 1→0, 2→2}
Model output: 0=door, 1=window → zapis jako cls_id+1

## Format labeli
YOLO (5 pól): cls cx cy w h
Wielokąt (11 pól): cls x1 y1 ... x5 y5
Ściany w obu formatach; drzwi/okna tylko YOLO.

## Znane problemy
1. POLARS_SKIP_CPU_CHECK=1 zawsze wymagany
2. workers=0 (CPU nie obsługuje wielowątkowości OpenCV)
3. Wykresy ultralytics padają na polars shim — nieszkodliwe
4. Pseudo-labeled okna detektują pokoje — w trakcie naprawy (filtr area+overlap)
5. d3 był zepsuty — poprawiony 14.07.2026; v1 trenowany na złych danych