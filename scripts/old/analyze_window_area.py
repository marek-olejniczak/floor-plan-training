#!/usr/bin/env python3
"""
analyze_window_area.py — Analiza rozkładu powierzchni okien (klasa 2)
w walls_doors_windows d1/d2. Ustala threshold do odrzucania fałszywych okien.
"""

import os, sys
from pathlib import Path
import numpy as np

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

SRC = Path.home() / "data" / "walls_doors_windows"
CLASSES = {0: "wall", 1: "door", 2: "window"}
TARGET_CLASS = 2  # window

areas = []
counts = {0: 0, 1: 0, 2: 0}

for ds in sorted(SRC.iterdir()):
    if not ds.is_dir():
        continue
    for split in ("train", "valid", "test"):
        lbl_dir = ds / split / "labels"
        if not lbl_dir.is_dir():
            continue
        for lbl_file in sorted(lbl_dir.glob("*.txt")):
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    if cls not in counts:
                        continue
                    counts[cls] += 1
                    if cls != TARGET_CLASS:
                        continue
                    _, cx, cy, w, h = map(float, parts[:5])
                    area = w * h
                    areas.append(area)

areas = np.array(areas)

print(f"{'='*55}")
print(f"  ANALIZA POWIERZCHNI OKIEN (klasa 2)")
print(f"{'='*55}")
print(f"  Dane: {SRC}")
print(f"  Total okien: {len(areas)}")
print(f"  Total ścian: {counts[0]}")
print(f"  Total drzwi: {counts[1]}")
print(f"  Total okien: {counts[2]}")
print(f"{'='*55}")

print(f"\n  Statystyki (area = w_norm * h_norm):")
print(f"  min:        {areas.min():.8f}")
print(f"  max:        {areas.max():.8f}")
print(f"  mean:       {areas.mean():.8f}")
print(f"  median:     {np.median(areas):.8f}")
print(f"  std:        {areas.std():.8f}")

print(f"\n  Percentyle:")
for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
    print(f"  P{p}:       {np.percentile(areas, p):.8f}")

# Sugerowany threshold
p95 = np.percentile(areas, 95)
p99 = np.percentile(areas, 99)
threshold_95_1_5 = p95 * 1.5
threshold_99_1_5 = p99 * 1.5

print(f"\n  Sugerowane thresholde:")
print(f"  P95 × 1.5 = {threshold_95_1_5:.8f}")
print(f"  P99 × 1.5 = {threshold_99_1_5:.8f}")

# Ile by odpadło przy każdym progu
for label, thresh in [
    ("P95 × 1.5", threshold_95_1_5),
    ("P99 × 1.5", threshold_99_1_5),
]:
    n_removed = (areas > thresh).sum()
    print(f"  {label}: {n_removed}/{len(areas)} odrzuconych ({100*n_removed/len(areas):.2f}%)")

print(f"\n  Wizualna inspekcja outlierów (>P99):")
above_p99 = areas > p99
outliers = areas[above_p99]
print(f"  {len(outliers)} okien > P99 = {p99:.8f}")
if len(outliers) > 0:
    print(f"  max outlier: {outliers.max():.8f}")
    print(f"  min outlier: {outliers.min():.8f}")
    factor = outliers / p99
    print(f"  krotność P99: min={factor.min():.2f}x, max={factor.max():.2f}x")

print(f"\n  Pierwsze 20 outlierów (area, sqrt(area) approx side):")
for a in outliers[:20]:
    approx_side = np.sqrt(a) * 100
    print(f"    area={a:.6f}  (~{approx_side:.1f}% img side)")
