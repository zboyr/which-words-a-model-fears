#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -o <workdir>/x101/logs/%x_%j.out
# x101 GPU stages, one (tag, stage) per job. env TAG=<x88 tag>,
# STAGE=preflight|detect|gen_O|gen_S|gen_M|gen_E. Submit with -p <gpu-partition> and an
# explicit --time (the devel partitions default to a 15-minute limit); raise --mem
# at submit for qwen27b. Generation resumes by prompt id from its output path.
set -euo pipefail
B=<workdir>/x101
# The x88 venv was swept from scratch with the model cache; the x100 venv is the
# surviving transformers-5.x environment (torch 2.11, transformers 5.15, accelerate 1.15).
PY=<workdir>/x100/venv/bin/python
export HF_HOME=<workdir>/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd "$B/code"
case "$STAGE" in
  preflight) exec "$PY" x101_preflight.py --tag "$TAG" ;;
  detect) exec "$PY" x101_detect.py --tag "$TAG" ;;
  gen_O|gen_S|gen_M|gen_E) exec "$PY" x101_generate.py --tag "$TAG" --condition "${STAGE#gen_}" ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
