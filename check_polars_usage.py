#!/usr/bin/env python3
"""Check how ultralytics uses polars"""
import ast, sys
from pathlib import Path

site_pkg = Path.home() / "projects" / "trening" / ".venv" / "lib" / "python3.11" / "site-packages"

files = [
    site_pkg / "ultralytics/utils/__init__.py",
    site_pkg / "ultralytics/engine/trainer.py",
    site_pkg / "ultralytics/utils/benchmarks.py",
    site_pkg / "ultralytics/utils/callbacks/wb.py",
    site_pkg / "ultralytics/utils/plotting.py",
]

for f in files:
    if not f.exists():
        continue
    print(f"=== {f.name} ===")
    code = f.read_text()
    for i, line in enumerate(code.split("\n"), 1):
        if "polars" in line.lower():
            print(f"  L{i}: {line.strip()}")
