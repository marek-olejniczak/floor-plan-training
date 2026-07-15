#!/usr/bin/env python3
"""Extract polars usage context from ultralytics source."""
from pathlib import Path

site_pkg = Path.home() / "projects" / "trening" / ".venv" / "lib" / "python3.11" / "site-packages"

# Read key files
trainer_py = (site_pkg / "ultralytics/engine/trainer.py").read_text()
init_py = (site_pkg / "ultralytics/utils/__init__.py").read_text()

# Extract read_results_csv
in_func = False
for i, line in enumerate(trainer_py.split("\n"), 1):
    if "def read_results_csv" in line:
        in_func = True
    if in_func:
        print(f"trainer.py L{i}: {line}")
        if in_func and line.strip() and "return" in line and "def " not in line[1:]:
            if i > 10:
                break

print("\n--- utils/__init__.py L215-240 ---")
for i, line in enumerate(init_py.split("\n"), 1):
    if 215 <= i <= 240:
        print(f"L{i}: {line}")
