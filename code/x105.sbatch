#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -o <workdir>/x105/logs/%x_%j.out
# x105 GPU stages, one (tag, stage) per job. env TAG=<x101 tag, qwen included>,
# STAGE=kl | gen_<arm> (arm as in x105_common.ARMS, e.g. gen_E_true_m150) | gen_array
# (the arm is ARM_ORDER[$SLURM_ARRAY_TASK_ID]; submit one throttled array per model, e.g.
# --array=0-6%2, and index 7 — the KL-matched control — with a dependency on that model's
# kl job). Submit with an explicit --time (the devel partitions default to a 15-minute limit
# and the batch partitions to an hour); raise --mem at submit for qwen27b. Generation resumes
# by prompt id from its output path.
set -euo pipefail
B=<workdir>/x105
# The x100 venv is the surviving transformers-5.x environment x101 and x103 ran in
# (torch 2.11, transformers 5.15, accelerate 1.15).
PY=<workdir>/x100/venv/bin/python
export HF_HOME=<workdir>/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# keep the weight-loading progress bar out of the job logs
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error
# Main-dose arms first, so a partially finished model still carries the primary result; the
# KL-matched control is last because it waits on the model's own kl job.
ARM_ORDER=(E_true_m150 O_true_p150 E_true_m075 O_true_p075 E_rand1_m150 E_rand2_m150 O_rand1_p150 E_randkl_m150)
cd "$B/code"
case "$STAGE" in
  kl) exec "$PY" x105_calibrate_kl.py --tag "$TAG" ;;
  gen_array) exec "$PY" x105_generate.py --tag "$TAG" --arm "${ARM_ORDER[$SLURM_ARRAY_TASK_ID]}" ;;
  gen_*) exec "$PY" x105_generate.py --tag "$TAG" --arm "${STAGE#gen_}" ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
