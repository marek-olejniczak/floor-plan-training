from ultralytics import YOLO
from pathlib import Path

mini_yaml = Path("runs/mini_dataset/data.yaml")
if not mini_yaml.exists():
    print("No mini dataset found. Generate with health_check.py first.")
    exit(1)

model = YOLO("yolo11s-obb.pt")
results = model.train(
    data=str(mini_yaml),
    epochs=3,
    imgsz=640,
    batch=4,
    device=0,
    workers=2,
    project="runs",
    name="health_check",
    exist_ok=True,
    verbose=True,
    amp=True,
)
print("Training finished.")
