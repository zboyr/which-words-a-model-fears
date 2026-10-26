#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -o <workdir>/x113/logs/%x_%A_%a.out
# x113 GPU stages, one model per job: env TAG=<tag> and STAGE=
#   project   Stage 5: projections of O / S_T / C_T on the model's direction
#   gen       Stage 3: one generation shard: env NSHARDS (array index = shard)
#   genloop   a work-stealing worker: claims unfinished generation shards of TAG one at a time
#             (mkdir lock under data/runs/x113/<tag>/genlocks, released on exit/TERM, reclaimed
#             when the owning job is gone) and runs each; shards are resumable per cell
# Submit with an explicit --time and partition; raise --mem at submit for qwen27b (140G).
set -euo pipefail
B=<workdir>/x113
PY=<workdir>/x100/venv/bin/python
export HF_HOME=<workdir>/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error
cd "$B/code"
case "$STAGE" in
  project) exec "$PY" x113_project.py --tag "$TAG" ${LIMIT:+--limit $LIMIT} ;;
  gen) exec "$PY" x113_generate.py --tag "$TAG" --shard "${SLURM_ARRAY_TASK_ID:-0}" --n-shards "${NSHARDS:-1}" ;;
  genloop)
    L="$B/data/runs/x113/$TAG/genlocks"; mkdir -p "$L"
    me="$SLURM_JOB_ID"; held=""; child=""
    release() { if [ -n "$held" ] && [ "$(cat "$L/$held/owner" 2>/dev/null)" = "$me" ]; then rm -rf "$L/$held"; fi; held=""; }
    on_term() { echo "worker $me: TERM at $(date)"; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null; [ -n "$child" ] && wait "$child" || true; release; exit 143; }
    trap on_term TERM
    trap release EXIT
    for k in $(seq 0 $((NSHARDS-1))); do
      [ -e "$L/done_$k" ] && continue
      if ! mkdir "$L/$k" 2>/dev/null; then
        owner=$(cat "$L/$k/owner" 2>/dev/null || true)
        [ -z "$owner" ] && continue
        state=$(scontrol show job "$owner" 2>&1 || true)
        if echo "$state" | grep -q -E "JobState=(RUNNING|PENDING|CONFIGURING|COMPLETING|SUSPENDED|REQUEUED)|timed out|Unable to contact"; then continue; fi
        echo "worker $me: reclaiming shard $k from job $owner"
        rm -rf "$L/$k"; mkdir "$L/$k" 2>/dev/null || continue
      fi
      echo "$me" > "$L/$k/owner"; held=$k
      echo "worker $me on $(hostname) takes $TAG shard $k at $(date)"
      "$PY" x113_generate.py --tag "$TAG" --shard "$k" --n-shards "$NSHARDS" & child=$!
      rc=0; wait "$child" || rc=$?; child=""
      if [ "$rc" -ne 0 ]; then echo "worker $me: shard $k exited $rc"; release; exit "$rc"; fi
      touch "$L/done_$k"; release
    done
    echo "worker $me: no $TAG shards left at $(date)"
    ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
