#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -o <workdir>/x104/logs/%x_%j.out
# x104: GuardReasoner-8B judges one shard of the x104 rows on one GPU.
# env SHARD=<i> NSHARDS=<n> (required); BATCH=<n> sets the batch size (default 48);
# LIMIT=<n> and OUT=<path> route a smoke test to a throwaway output.
# Submit with an explicit --time and partition. The judge resumes by row key from its
# shard file, so a shard larger than one allocation is carried by a dependency chain of jobs.
set -euo pipefail
B=<workdir>/x104
PY=<workdir>/x100/venv/bin/python
export HF_HOME=<workdir>/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd "$B/code"
EXTRA=()
if [ -n "${BATCH:-}" ]; then EXTRA+=(--batch-size "$BATCH"); fi
if [ -n "${LIMIT:-}" ]; then EXTRA+=(--limit "$LIMIT"); fi
if [ -n "${OUT:-}" ]; then EXTRA+=(--out "$OUT"); fi
exec "$PY" x104_judge.py --shard "$SHARD" --n-shards "$NSHARDS" "${EXTRA[@]}"
