#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -o <workdir>/x100/logs/%x_%A_%a.out
# x100 GPU stages. env STAGE=extract|generate|pilot. extract and generate map the
# array index SLURM_ARRAY_TASK_ID to the shard / chunk (24 shards, 16 chunks); both
# resume by cell id, so an array can be resubmitted. Pass --time at submit: devel
# partitions default to a 15-minute limit.
set -euo pipefail
B=<workdir>/x100
PY=$B/venv/bin/python
export HF_HOME=<workdir>/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd "$B/code"
case "$STAGE" in
  extract)
    exec "$PY" x100_extract_hidden.py --shard "${SHARD:-${SLURM_ARRAY_TASK_ID:?}}" --num-shards "${NSHARDS:-24}"
    ;;
  generate)
    exec "$PY" x100_generate.py --chunk "${CHUNK:-${SLURM_ARRAY_TASK_ID:?}}" --num-chunks "${NCHUNKS:-16}"
    ;;
  pilot)
    exec "$PY" x100_generate.py --pilot
    ;;
  *)
    echo "unknown STAGE=$STAGE" >&2
    exit 2
    ;;
esac
