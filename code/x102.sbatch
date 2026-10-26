#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -o <workdir>/x102/logs/%x_%j.out
# x102 GPU judge stages, one judge per job. env STAGE=incumbent|wildguard|qwen3guard|guardreasoner.
# Submit with an explicit --time and partition. Open-weight arms judge the sample, then
# XSTest, in the same job; every stage resumes by row key from its output path, so a long
# arm can be carried by a dependency chain of jobs. Optional env BATCH=<n> sets the batch
# size; LIMIT=<n> and OUT=<path> route a smoke test to a throwaway output.
set -euo pipefail
B=<workdir>/x102
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
case "$STAGE" in
  incumbent) exec "$PY" x102_judge_incumbent_xstest.py "${EXTRA[@]}" ;;
  wildguard|qwen3guard|guardreasoner)
    "$PY" x102_judge_open.py --arm "$STAGE" "${EXTRA[@]}"
    if [ -n "${OUT:-}" ]; then OUT_X="${OUT%.jsonl}_xstest.jsonl"; EXTRA=(${BATCH:+--batch-size "$BATCH"} ${LIMIT:+--limit "$LIMIT"} --out "$OUT_X"); fi
    exec "$PY" x102_judge_open.py --arm "$STAGE" --xstest "${EXTRA[@]}" ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
