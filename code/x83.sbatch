#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -o <workdir>/x81/logs/%x_%j.out
# x83 GPU stages (shares the x81 working tree on the cluster). env STAGE: detect | gen_O | gen_S | gen_M | gen_E
set -eu
B=<workdir>/x81
PY=<workdir>/x63/venv/bin/python
export HF_HOME=<workdir>/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd $B/code
case "$STAGE" in
  detect) exec $PY x83_detect.py ;;
  gen_O|gen_S|gen_M|gen_E) exec $PY x83_generate.py --condition "${STAGE#gen_}" ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
