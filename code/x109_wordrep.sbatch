#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -o <workdir>/x106/logs/%x_%j.out
# x109 word-representation probe on Qwen3.5-4B: extract (GPU) then analyze (GPU ridge).
# Submit with an explicit --time and partition.
set -euo pipefail
B=<workdir>/x106
PY=<workdir>/x100/venv/bin/python
export HF_HOME=<workdir>/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error
cd "$B/code"
if [ ! -f "$B/data/runs/x109/wordrep/features.npz" ]; then "$PY" x109_wordrep_extract.py; fi
exec "$PY" x109_wordrep_analyze.py
