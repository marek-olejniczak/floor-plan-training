from pathlib import Path
import random, shutil

src = Path("merged_dataset")
dst = Path("runs/mini_dataset")
if dst.exists():
    shutil.rmtree(dst)

for split in ["train", "valid"]:
    (dst / split / "images").mkdir(parents=True)
    (dst / split / "labels").mkdir(parents=True)
    imgs = sorted((src / split / "images").glob("*"))
    random.seed(0)
    for img in random.sample(imgs, min(100, len(imgs))):
        shutil.copy2(img, dst / split / "images" / img.name)
        lbl = src / split / "labels" / (img.stem + ".txt")
        if lbl.exists():
            shutil.copy2(lbl, dst / split / "labels" / (img.stem + ".txt"))

(dst / "data.yaml").write_text(
    "train: " + str(dst / "train" / "images") + "\n" +
    "val: " + str(dst / "valid" / "images") + "\n\n" +
    "names:\n  0: wall\n"
)
print("Mini dataset ready")
