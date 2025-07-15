#!/bin/sh

export OPENBLAS_NUM_THREADS=1
export OPENMP_NUM_THREADS=1

uv run mpirun -n 10 python ./experiments.py small
uv run mpirun -n 25 python ./experiments.py large

uv run python ./compute_solutions.py
uv run python ./compute_err_vs_time.py
uv run python ./compute_time_table.py
uv run python ./compute_error_bounds.py
