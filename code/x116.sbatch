#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -o <workdir>/x113/logs/x116_%x_%A_%a.out
# x116 GPU stages in the x113 the cluster tree, one model per job: env TAG=<tag> and STAGE=
#   genloop   work-stealing worker over NSHARDS resumable generation shards of TAG (x113's lock
#             scheme under data/runs/x116/<tag>/genlocks), then the projection if PROJECT=1 and
#             it is not done yet
#   project   Stage 3 projections of O / R_rand
# Submit with an explicit --time and partition; --mem=140G for qwen27b.
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
  project) exec "$PY" x116_project.py --tag "$TAG" ;;
  genloop)
    L="$B/data/runs/x116/$TAG/genlocks"; mkdir -p "$L"
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
      "$PY" x116_generate.py --tag "$TAG" --shard "$k" --n-shards "$NSHARDS" & child=$!
      rc=0; wait "$child" || rc=$?; child=""
      if [ "$rc" -ne 0 ]; then echo "worker $me: shard $k exited $rc"; release; exit "$rc"; fi
      touch "$L/done_$k"; release
    done
    echo "worker $me: no $TAG shards left at $(date)"
    if [ "${PROJECT:-0}" = "1" ] && mkdir "$L/project" 2>/dev/null; then
      echo "$me" > "$L/project/owner"
      "$PY" x116_project.py --tag "$TAG" || { rm -rf "$L/project"; exit 1; }
    fi
    ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
