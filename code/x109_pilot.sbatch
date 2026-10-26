#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -o <workdir>/x106/logs/%x_%j.out
# Benign insertion pilot GPU stages on Qwen3.5-4B. env STAGE=
#   project   the 20 x (O + appended words) projection pass
#   gen       one generation shard: env NSHARDS (array index = shard)
# Submit with an explicit --time.
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
case "$STAGE" in
  project) exec "$PY" x109_pilot_project.py ;;
  gen) exec "$PY" x109_pilot_generate.py --shard "${SLURM_ARRAY_TASK_ID:-0}" --n-shards "${NSHARDS:-1}" ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
