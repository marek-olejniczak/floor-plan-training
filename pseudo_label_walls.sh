#!/usr/bin/env bash
export POLARS_SKIP_CPU_CHECK=1
cd /home/marek_olejniczak/projects/trening
source .venv/bin/activate
python scripts/pseudo_label_walls.py
