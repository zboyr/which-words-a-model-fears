#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o <workdir>/e2_temp/logs/%x_%j.out
# x107: QK-vs-OV decomposition of the appended word's read on Qwen3.5-4B (x85 grid).
# Submit with -p and an explicit --time. env LIMIT=<n pairs> for a debug run, OUTNAME for the output dir.
set -euo pipefail
B=<workdir>/e2_temp
PY=<workdir>/x100/venv/bin/python
export HF_HOME=<workdir>/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd "$B/code"
exec "$PY" x107_value_sep_extract.py --out "$B/out/${OUTNAME:-vsep}" --limit "${LIMIT:-0}"
