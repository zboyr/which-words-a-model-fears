#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH -o <workdir>/x86/logs/%x_%j.out
# x86 GPU stages: manifest | calibrate | generate. generate requires CONDITION.
set -euo pipefail
B=<workdir>/x86
PY=<workdir>/x63/venv/bin/python
export HF_HOME=<workdir>/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd "$B/code"
case "$STAGE" in
  manifest)
    exec "$PY" x86_build_manifest.py
    ;;
  calibrate)
    exec "$PY" x86_calibrate_kl.py
    ;;
  generate)
    : "${CONDITION:?set CONDITION}"
    exec "$PY" x86_generate.py --condition "$CONDITION"
    ;;
  smoke)
    : "${CONDITION:?set CONDITION}"
    exec "$PY" x86_generate.py --condition "$CONDITION" --limit 4 \
      --out "<workdir>/x86/smoke/x86_${CONDITION}.json"
    ;;
  *)
    echo "unknown STAGE=$STAGE" >&2
    exit 2
    ;;
esac

