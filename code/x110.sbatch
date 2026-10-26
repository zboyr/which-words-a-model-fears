#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -o <workdir>/x110/logs/%x_%A_%a.out
# x110 GPU stages on Qwen3.5-4B. env STAGE=
#   scan      one vocabulary-scan shard over the 192 prompts: env NSHARDS (array index = shard)
#   keyscan   the chosen words and twins on every prompt -> key_parts/
#   assemble  stack the parts: key_parts -> key_words_scan.json, scan_parts -> scan.json (prompts covered)
#   gen       one generation shard: env NSHARDS (array index = shard)
#   genloop   a work-stealing worker: claims unfinished generation shards one at a time
#             (mkdir lock under data/runs/x110/genlocks, released on exit/TERM, reclaimed
#             when the owning job is gone) and runs each; shards are resumable per cell
#   keyscantopup  the key-word scan on the HP top-up prompts (prompts_topup.json), then its assembly
#                 -> key_words_scan_topup.json / key_words_shift_topup.npz
#   gentopup  one generation shard of the top-up prompts' cells: env NSHARDS (array index = shard)
#             -> generations/generations_topup_shard{k}of{n}.json
# Submit with an explicit --time and partition.
set -euo pipefail
B=<workdir>/x110
PY=<workdir>/x100/venv/bin/python
export HF_HOME=<workdir>/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error
cd "$B/code"
case "$STAGE" in
  scan) exec "$PY" x110_scan_gpu.py --shard "${SLURM_ARRAY_TASK_ID:-0}" --n-shards "${NSHARDS:-1}" ;;
  keyscan) exec "$PY" x110_scan_gpu.py --words key ;;
  assemble) "$PY" x110_scan_gpu.py --words key --assemble && exec "$PY" x110_scan_gpu.py --words all --assemble ;;
  gen) exec "$PY" x110_generate.py --shard "${SLURM_ARRAY_TASK_ID:-0}" --n-shards "${NSHARDS:-1}" ;;
  keyscantopup) "$PY" x110_scan_gpu.py --words key --prompts topup && exec "$PY" x110_scan_gpu.py --words key --prompts topup --assemble ;;
  gentopup) exec "$PY" x110_generate.py --prompts topup --shard "${SLURM_ARRAY_TASK_ID:-0}" --n-shards "${NSHARDS:-1}" ;;
  genloop)
    L="$B/data/runs/x110/genlocks"; mkdir -p "$L"
    me="$SLURM_JOB_ID"; held=""; child=""
    release() { if [ -n "$held" ] && [ "$(cat "$L/$held/owner" 2>/dev/null)" = "$me" ]; then rm -rf "$L/$held"; fi; held=""; }
    on_term() { echo "worker $me: TERM at $(date)"; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null; [ -n "$child" ] && wait "$child" || true; release; exit 143; }
    trap on_term TERM
    trap release EXIT
    if [ ! -e "$L/done_keyscan" ] && mkdir "$L/keyscan" 2>/dev/null; then
      echo "$me" > "$L/keyscan/owner"; held=keyscan
      echo "worker $me on $(hostname) runs the key-word scan at $(date)"
      "$PY" x110_scan_gpu.py --words key & child=$!
      rc=0; wait "$child" || rc=$?; child=""
      if [ "$rc" -ne 0 ]; then echo "worker $me: key-word scan exited $rc"; release; exit "$rc"; fi
      touch "$L/done_keyscan"; release
    fi
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
      echo "worker $me on $(hostname) takes shard $k at $(date)"
      "$PY" x110_generate.py --shard "$k" --n-shards "$NSHARDS" & child=$!
      rc=0; wait "$child" || rc=$?; child=""
      if [ "$rc" -ne 0 ]; then echo "worker $me: shard $k exited $rc"; release; exit "$rc"; fi
      touch "$L/done_$k"; release
    done
    echo "worker $me: no shards left at $(date)"
    ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
