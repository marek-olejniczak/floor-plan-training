#!/usr/bin/env bash
export POLARS_SKIP_CPU_CHECK=1
export PATH="/home/marek_olejniczak/.local/bin:$PATH"
cd /home/marek_olejniczak/projects/trening
source .venv/bin/activate
python scripts/train_final.py
