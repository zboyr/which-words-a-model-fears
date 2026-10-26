#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -o <workdir>/x112/logs/%x_%j.out
# x112 GPU stages, one model per job (env TAG). env STAGE=
#   scan      the candidate list on the eight scan prompts (env MODE=cached|full) -> <tag>/table.*
#   keyscan   the 40 chosen words on the 32 held-out prompts -> <tag>/key_words_*
#   pool      fresh O cells for the requested held-out candidates -> <tag>/generations/generations_pool.json
#   orep      a second, selection-independent O cell of the 32 chosen prompts -> generations_orep.json
#   genloop   a work-stealing worker over the grid shards of the models listed (in order) in
#             data/runs/x112/gen_tags (or env GEN_TAGS_FILE; gen_tags_27b for --mem=140G workers),
#             re-read per worker: claims an unfinished (tag, shard) under a mkdir lock in
#             data/runs/x112/genlocks, released on exit/TERM, reclaimed when its job is gone;
#             shards are resumable per cell
#   pooltopup / oreptopup / keyscantopup / genlooptopup
#             the same four stages for the harmful top-up prompts (topup_request.json /
#             prompts_topup.json; outputs *_topup*); genlooptopup reads gen_tags_topup (or env
#             GEN_TAGS_FILE), <tag>/nshards_topup and locks under genlocks_topup
# Submit with an explicit --time and partition; qwen27b needs --mem=140G.
set -euo pipefail
B=<workdir>/x112
PY=<workdir>/x100/venv/bin/python
export HF_HOME=<workdir>/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error
cd "$B/code"
echo "job $SLURM_JOB_ID stage $STAGE on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) at $(date)"
case "$STAGE" in
  scan) exec "$PY" x112_scan_gpu.py --tag "$TAG" --words all --mode "${MODE:-cached}" ;;
  keyscan) exec "$PY" x112_scan_gpu.py --tag "$TAG" --words key --mode "${MODE:-cached}" ;;
  keyscantopup) exec "$PY" x112_scan_gpu.py --tag "$TAG" --words key --prompts topup --mode "${MODE:-cached}" ;;
  pool) exec "$PY" x112_generate.py --tag "$TAG" --stage pool ;;
  orep) exec "$PY" x112_generate.py --tag "$TAG" --stage orep ;;
  pooltopup) exec "$PY" x112_generate.py --tag "$TAG" --stage pooltopup ;;
  oreptopup) exec "$PY" x112_generate.py --tag "$TAG" --stage oreptopup ;;
  genloop|genlooptopup)
    if [ "$STAGE" = genlooptopup ]; then LOCK_DIR=genlocks_topup; NSHARDS_FILE=nshards_topup; GRID_STAGE=gridtopup; GEN_TAGS_FILE="${GEN_TAGS_FILE:-gen_tags_topup}"; else LOCK_DIR=genlocks; NSHARDS_FILE=nshards; GRID_STAGE=grid; GEN_TAGS_FILE="${GEN_TAGS_FILE:-gen_tags}"; fi
    L="$B/data/runs/x112/$LOCK_DIR"; mkdir -p "$L"
    me="$SLURM_JOB_ID"; held=""; child=""
    release() { if [ -n "$held" ] && [ "$(cat "$L/$held/owner" 2>/dev/null)" = "$me" ]; then rm -rf "$L/$held"; fi; held=""; }
    on_term() { echo "worker $me: TERM at $(date)"; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null; [ -n "$child" ] && wait "$child" || true; release; exit 143; }
    trap on_term TERM
    trap release EXIT
    for t in $(cat "$B/data/runs/x112/$GEN_TAGS_FILE"); do
      n=$(cat "$B/data/runs/x112/$t/$NSHARDS_FILE")
      for k in $(seq 0 $((n-1))); do
        id="${t}_$k"
        [ -e "$L/done_$id" ] && continue
        if ! mkdir "$L/$id" 2>/dev/null; then
          owner=$(cat "$L/$id/owner" 2>/dev/null || true)
          [ -z "$owner" ] && continue
          state=$(scontrol show job "$owner" 2>&1 || true)
          if echo "$state" | grep -q -E "JobState=(RUNNING|PENDING|CONFIGURING|COMPLETING|SUSPENDED|REQUEUED)|timed out|Unable to contact"; then continue; fi
          echo "worker $me: reclaiming $id from job $owner"
          rm -rf "$L/$id"; mkdir "$L/$id" 2>/dev/null || continue
        fi
        echo "$me" > "$L/$id/owner"; held=$id
        echo "worker $me on $(hostname) takes $id at $(date)"
        "$PY" x112_generate.py --tag "$t" --stage "$GRID_STAGE" --shard "$k" --n-shards "$n" & child=$!
        rc=0; wait "$child" || rc=$?; child=""
        if [ "$rc" -ne 0 ]; then echo "worker $me: $id exited $rc"; release; exit "$rc"; fi
        touch "$L/done_$id"; release
      done
    done
    echo "worker $me: no shards left at $(date)"
    ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
