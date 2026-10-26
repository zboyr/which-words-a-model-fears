#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -o <workdir>/x106/logs/%x_%j.out
# x106 GPU stages, one (tag, stage) per job. env TAG=<tag> and STAGE=
#   project                      Stage 0a: boundary states and projections of all prompts
#   nominate                     Stage 0b shard: env TEXT (default O), NSHARDS (array index = shard),
#                                COHORT=1 restricts to the cohort, BATCH sets the batch size,
#                                PROJONLY=1 skips the deletion variants (texts C and R)
#   gen_<arm>                    Stage 2: one arm (x106_common.arms_for), resumable by prompt id
#   gr                           Stage 3b: GuardReasoner-8B on one arm's responses: env ARM, NSHARDS
# Submit with an explicit --time (devel partitions default to 15 min, batch partitions to 1 h);
# raise --mem at submit for qwen27b (140G).
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
  project) exec "$PY" x106_stage0_project.py --tag "$TAG" ${LIMIT:+--limit $LIMIT} ;;
  nominate)
    EXTRA=()
    if [ "${COHORT:-0}" = "1" ]; then EXTRA+=(--cohort); fi
    if [ -n "${BATCH:-}" ]; then EXTRA+=(--batch-size "$BATCH"); fi
    if [ "${PROJONLY:-0}" = "1" ]; then EXTRA+=(--projection-only); fi
    exec "$PY" x106_stage0_nominate.py --tag "$TAG" --text "${TEXT:-O}" --shard "${SLURM_ARRAY_TASK_ID:-0}" --n-shards "${NSHARDS:-1}" ${LIMIT:+--limit $LIMIT} "${EXTRA[@]}" ;;
  gen_*) exec "$PY" x106_generate.py --tag "$TAG" --arm "${STAGE#gen_}" ;;
  gr) exec "$PY" x106_judge_guardreasoner.py --tag "$TAG" --arm "$ARM" --shard "${SLURM_ARRAY_TASK_ID:-0}" --n-shards "${NSHARDS:-1}" ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
