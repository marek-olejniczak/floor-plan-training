#!/usr/bin/env python3
"""visualize_dataset.py — grid of 16 random images with annotations."""

import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

IMAGES = Path.home() / "data" / "raw_predictions" / "train" / "images"
LABELS = Path.home() / "data" / "corrected_walls" / "train" / "labels"
OUTPUT = Path("/mnt/d/rzuty/trening") / "dataset_review.png"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
N, C, R, CW, CH, HH = 16, 4, 4, 640, 640, 60

all_l = sorted(LABELS.glob("*.txt"))
sel = random.sample(all_l, min(N, len(all_l)))
fnt = ImageFont.truetype(FONT, 18) if Path(FONT).exists() else None
sf = ImageFont.truetype(FONT, 14) if fnt else None

cv = Image.new("RGB", (C * CW, R * CH + HH), (30, 30, 30))
d0 = ImageDraw.Draw(cv)
d0.text((10, 8), "Dataset review — green=wall poly, blue=door, orange=window", font=fnt, fill=(200, 200, 200))

for idx, lp in enumerate(sel):
    col, row = idx % C, idx // C
    ox, oy = col * CW, row * CH + HH
    stem = lp.stem

    ip = IMAGES / f"{stem}.jpg"
    if ip.exists():
        try:
            cv.paste(Image.open(ip).convert("RGB").resize((CW, CH), Image.LANCZOS), (ox, oy))
        except Exception:
            d0.rectangle([ox, oy, ox + CW, oy + CH], fill=(40, 40, 40))
    else:
        d0.rectangle([ox, oy, ox + CW, oy + CH], fill=(40, 40, 40))

    dw = ImageDraw.Draw(cv)
    ct = [0, 0, 0]
    extras = []
    with open(lp) as f:
        for line in f:
            p = line.strip().split()
            if not p:
                continue
            cid = int(p[0])
            if cid > 2:
                continue
            ct[cid] += 1

            if cid == 0 and len(p) >= 11:
                cs = list(map(float, p[1:11]))
                pts = [(int(cs[i] * CW + ox), int(cs[i + 1] * CH + oy)) for i in range(0, 10, 2)]
                dw.polygon(pts, outline=(0, 200, 0), width=2)

            elif cid in (1, 2) and len(p) >= 7:
                _, cx, cy, w, h, conf = p[:6]
                cx, cy, w, h, conf = map(float, (cx, cy, w, h, conf))
                extra = float(p[7]) if len(p) >= 8 else 0.0
                x1 = int((cx - w / 2) * CW + ox)
                y1 = int((cy - h / 2) * CH + oy)
                x2 = int((cx + w / 2) * CW + ox)
                y2 = int((cy + h / 2) * CH + oy)
                color = (0, 100, 255) if cid == 1 else (255, 150, 0)
                dw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                if cid == 2:
                    extras.append(extra)

    avg_extra = sum(extras) / len(extras) if extras else 0.0
    info = f"w:{ct[0]} d:{ct[1]} o:{ct[2]}"
    if extras:
        info += f" ov:{avg_extra:.2f}"
    bb = dw.textbbox((ox + 5, oy + CH - 22), info, font=sf)
    dw.rectangle(bb, fill=(0, 0, 0, 180))
    dw.text((ox + 5, oy + CH - 22), info, font=sf, fill=(200, 200, 200))

    bb2 = dw.textbbox((ox + 5, oy + 5), stem[:40], font=sf)
    dw.rectangle(bb2, fill=(0, 0, 0, 180))
    dw.text((ox + 5, oy + 5), stem[:40], font=sf, fill=(200, 200, 200))

cv.save(OUTPUT)
print(f"Saved: {OUTPUT}")
