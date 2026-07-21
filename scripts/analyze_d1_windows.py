#!/usr/bin/env python3
"""analyze_d1_windows.py — compute window area distribution for d1 (CubiCasa)."""

from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D1 = Path("/mnt/d/rzuty/dane/yolo11datasets/walls_doors_windows/d1")
OUT_PNG = Path("/mnt/d/rzuty/trening") / "d1_window_area_distribution.png"

areas = []
per_split = defaultdict(list)

for split in ("train", "valid", "test"):
    lp = D1 / split / "labels"
    for f in sorted(lp.glob("*.txt")):
        with open(f) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) >= 5 and parts[0] == "2":
                    _, cx, cy, w, h = parts[:5]
                    a = float(w) * float(h)
                    areas.append(a)
                    per_split[split].append(a)

areas = np.array(areas)
percentiles = [50, 75, 90, 95, 97, 99, 99.5, 99.9]
pvals = {p: np.percentile(areas, p) for p in percentiles}

print(f"{'Stat':>20s}  {'Value':>12s}")
print("-" * 34)
print(f"{'Total windows':>20s}  {len(areas):>12d}")
print(f"{'Mean area':>20s}  {areas.mean():>12.6f}")
print(f"{'Std area':>20s}  {areas.std():>12.6f}")
print(f"{'Min area':>20s}  {areas.min():>12.6f}")
print(f"{'Max area':>20s}  {areas.max():>12.6f}")
print()
print(f"{'Percentile':>20s}  {'Value':>12s}")
print("-" * 34)
for p in percentiles:
    print(f"{p:>19.1f}%  {pvals[p]:>12.6f}")

print()
for s in ("train", "valid", "test"):
    a = np.array(per_split[s])
    print(f"{s:>20s}: {len(a):>6d} windows, mean={a.mean():.6f}, max={a.max():.6f}")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

ax1.hist(areas, bins=200, color="steelblue", edgecolor="none")
ax1.set_xlabel("Normalized area (w * h)")
ax1.set_ylabel("Count")
ax1.set_title("Window area distribution (linear)")
ax1.set_yscale("log")
cols = ["red", "orange", "green", "cyan", "blue", "purple", "brown", "magenta"]
for i, p in enumerate(percentiles):
    ax1.axvline(pvals[p], color=cols[i % len(cols)], ls="--", lw=1,
                label=f"P{p}={pvals[p]:.5f}")
ax1.legend(fontsize=7)

ax2.hist(areas, bins=200, color="steelblue", edgecolor="none")
ax2.set_xlabel("Normalized area (w * h)")
ax2.set_ylabel("Count")
ax2.set_title("Window area distribution (linear scale, linear y)")
for i, p in enumerate(percentiles):
    ax2.axvline(pvals[p], color=cols[i % len(cols)], ls="--", lw=1,
                label=f"P{p}={pvals[p]:.5f}")
ax2.legend(fontsize=7)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"\nSaved: {OUT_PNG}")
