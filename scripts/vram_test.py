import torch
from ultralytics import YOLO

m = YOLO("yolo11m-obb.pt")
for batch in [12, 16]:
    torch.cuda.reset_peak_memory_stats()
    results = m.train(
        data="runs/mini_dataset/data.yaml",
        epochs=1,
        imgsz=1024,
        batch=batch,
        device=0,
        workers=0,
        project="runs", name=f"vram_test_b{batch}",
        exist_ok=True, verbose=False, plots=False,
    )
    mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"batch={batch}: VRAM peak = {mem:.2f} GB")
