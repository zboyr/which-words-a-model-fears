#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH -p cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o <workdir>/x106/logs/%x_%j.out
# x109 word-representation probe follow-up: table neighbours vs embedding neighbours (CPU only,
# reads wordrep/features.npz). Submit with an explicit --time.
set -euo pipefail
B=<workdir>/x106
PY=<workdir>/x100/venv/bin/python
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "$B/code"
exec "$PY" x109_wordrep_neighbors.py
