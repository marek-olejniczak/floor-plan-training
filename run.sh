#!/bin/bash
source ~/.local/bin/env
export POLARS_SKIP_CPU_CHECK=1
cd ~/projects/trening
source .venv/bin/activate
python scripts/run_healthcheck.py
