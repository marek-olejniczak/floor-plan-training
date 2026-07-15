#!/bin/bash
source ~/.local/bin/env
export POLARS_SKIP_CPU_CHECK=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
cd ~/projects/trening
source .venv/bin/activate
echo 'Wszystkie wyniki trafia do: ~/projects/trening/runs/'
exec python scripts/train_full.py
