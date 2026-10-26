#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -o <workdir>/x106/logs/%x_%j.out
# Vocabulary-scan benchmark on Qwen3.5-4B (counts, cached-prefix validity, throughput).
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
exec "$PY" x109_scan_bench_gpu.py ${BENCH_ARGS:-}
