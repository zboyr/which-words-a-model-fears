#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -o <workdir>/x108/logs/%x_%j.out
# x108 GPU stages, one (tag, stage) per job. env TAG=<tag> (default qwen) and STAGE=
#   project                      Stage 0a: boundary states and projections of the 755 prompts
#   nominate                     Stage 0b shard: env TEXT (default O), NSHARDS (array index = shard),
#                                COHORT=1 restricts to the cohort, BATCH sets the batch size,
#                                PROJONLY=1 skips the deletion variants (text R)
#   gen_<arm>                    Stage 2: one arm (O_base, S_base, R_base), resumable by prompt id
#   gr                           Stage 3b: GuardReasoner-8B on one arm's responses: env ARM, NSHARDS
# Submit with an explicit --time (devel partitions default to 15 min, batch partitions to 1 h);
# raise --mem at submit for qwen27b (140G).
set -euo pipefail
TAG="${TAG:-qwen}"
B=<workdir>/x108
PY=<workdir>/x100/venv/bin/python
export HF_HOME=<workdir>/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error
cd "$B/code"
case "$STAGE" in
  project) exec "$PY" x108_stage0.py --tag "$TAG" --project ${LIMIT:+--limit $LIMIT} ;;
  nominate)
    EXTRA=()
    if [ "${COHORT:-0}" = "1" ]; then EXTRA+=(--cohort); fi
    if [ -n "${BATCH:-}" ]; then EXTRA+=(--batch-size "$BATCH"); fi
    if [ "${PROJONLY:-0}" = "1" ]; then EXTRA+=(--projection-only); fi
    exec "$PY" x108_stage0.py --tag "$TAG" --text "${TEXT:-O}" --shard "${SLURM_ARRAY_TASK_ID:-0}" --n-shards "${NSHARDS:-1}" ${LIMIT:+--limit $LIMIT} "${EXTRA[@]}" ;;
  gen_*) exec "$PY" x108_generate.py --tag "$TAG" --arm "${STAGE#gen_}" ;;
  gr) exec "$PY" x108_judge_guardreasoner.py --tag "$TAG" --arm "$ARM" --shard "${SLURM_ARRAY_TASK_ID:-0}" --n-shards "${NSHARDS:-1}" ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
