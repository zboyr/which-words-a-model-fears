#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -o <workdir>/x114/logs/%x_%j.out
# x114 GPU stages, one (tag, stage) per job. env TAG=<x101 tag, qwen included>, STAGE=
#   gate | gen_O | gen_S | gen_M | gen_E | gen_array (condition = COND_ORDER[$SLURM_ARRAY_TASK_ID])
#   extract | dose_<arm> (arm as in x105_common.ARMS) | dose_array (ARM_ORDER[$SLURM_ARRAY_TASK_ID])
# Submit with an explicit --time (the devel partitions default to 15 minutes and the batch
# partitions to an hour) and raise --mem to 140G for qwen27b. Every generation stage resumes
# by prompt id from its output path.
set -euo pipefail
B=<workdir>/x114
# The x100 venv is the surviving transformers-5.x environment x101, x103 and x105 ran in
# (torch 2.11, transformers 5.15, accelerate 1.15).
PY=<workdir>/x100/venv/bin/python
export HF_HOME=<workdir>/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error
COND_ORDER=(E O S M)   # E and O first: they are the arms the x105 stage doses
ARM_ORDER=(E_true_m150 O_true_p150 E_true_m075 O_true_p075 E_rand1_m150 E_rand2_m150 O_rand1_p150 E_randkl_m150)
cd "$B/code"
case "$STAGE" in
  gate) exec "$PY" x114_generate.py --tag "$TAG" --stage gate ;;
  gen_array) exec "$PY" x114_generate.py --tag "$TAG" --stage "${COND_ORDER[$SLURM_ARRAY_TASK_ID]}" ;;
  gen_*) exec "$PY" x114_generate.py --tag "$TAG" --stage "${STAGE#gen_}" ;;
  extract) exec "$PY" x114_extract.py --tag "$TAG" ;;
  dose_array) exec "$PY" x114_dosed.py --tag "$TAG" --arm "${ARM_ORDER[$SLURM_ARRAY_TASK_ID]}" ;;
  dose_*) exec "$PY" x114_dosed.py --tag "$TAG" --arm "${STAGE#dose_}" ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
