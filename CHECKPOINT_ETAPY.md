# Checkpoint Etap 0-5 (14.07.2026)

## Status
- [x] Etap 0: Środowisko WSL (Python 3.11, torch 2.6.0+cu124, ultralytics 8.4.95)
- [x] Etap 1: Unifikacja doors_windows → 2 klasy (2721 img, 12422 door, 8282 window)
- [x] Etap 2: Trening YOLO11s doors+windows (30e, mAP50=0.87 na 23e)
- [x] Etap 3: Pseudo-labelowanie walls (12687 img, 33181 door, 35147 window)
- [x] Etap 4: Finalny merge wszystkich danych
- [x] Etap 5: Finalny trening (w trakcie - tmux sesja train_final)

## Finalny dataset
- Lokalizacja: `~/data/combined_dataset/`
- Train: 13,563 img / Valid: 3,225 / Test: 1,874
- Klasy: 0=wall (489k), 1=door (90k), 2=window (82k)

## Komendy
```bash
# Monitorowanie treningu
wsl tmux attach -t train_final
# Odłączenie: Ctrl+B, D

# Po treningu - ewaluacja
wsl bash -c 'cd ~/projects/trening && .venv/bin/python -c "
from ultralytics import YOLO
model = YOLO(\"runs/final_v1/weights/best.pt\")
results = model.val(data=\"$HOME/data/combined_dataset/data.yaml\")
print(results)
"'

# Eksport do użycia
wsl bash -c 'cd ~/projects/trening && .venv/bin/python -c "
from ultralytics import YOLO
model = YOLO(\"runs/final_v1/weights/best.pt\")
model.export(format=\"torchscript\")
model.export(format=\"onnx\")
"'

# Kopiowanie modelu do apki
wsl cp ~/projects/trening/runs/final_v1/weights/best.pt ~/projects/trening/app/models/
```
