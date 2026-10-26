#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3:00:00
#SBATCH -o <workdir>/x80/logs/%x_%j.out
# x80 Stages 1-2: paired probes. env: SEED
set -eu
B=<workdir>/x80
export HF_HOME=<workdir>/hf_cache
<workdir>/x63/venv/bin/python $B/code/x80_fit_paired_probes.py \
  --cache <workdir>/x63/features/qwen_A15k_v4_full/hidden_states.pt \
  --labels $B/inputs/x80_labels.npz --indices-dir $B/inputs/splits \
  --seed $SEED --out-dir $B/out${EXT:+_ext} --n-perm 5 --n-boot 50 ${EXT:+--l2-ext}
