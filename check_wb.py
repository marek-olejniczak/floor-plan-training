#!/usr/bin/env python3
"""Check polars.selectors usage in wb.py"""
from pathlib import Path

code = (Path.home() / "projects" / "trening" / ".venv" / "lib" / "python3.11" / "site-packages" / "ultralytics" / "utils" / "callbacks" / "wb.py").read_text()

for i, line in enumerate(code.split("\n"), 1):
    if "polars" in line.lower() or "selectors" in line.lower() or "cs." in line:
        print(f"L{i}: {line}")
