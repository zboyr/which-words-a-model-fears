#!/bin/bash
#SBATCH -A <your-slurm-account>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -o <workdir>/x103/logs/%x_%j.out
# x103 boundary-state extraction, one model per job. env TAG=<x101 tag, qwen included>.
# Submit with -p <gpu-partition> and an explicit --time (the devel partitions default to a
# 15-minute limit); raise --mem at submit for qwen27b. 520 forward passes at batch size 1.
set -euo pipefail
B=<workdir>/x103
# The x100 venv is the surviving transformers-5.x environment x101 ran in
# (torch 2.11, transformers 5.15, accelerate 1.15).
PY=<workdir>/x100/venv/bin/python
export HF_HOME=<workdir>/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd "$B/code"
exec "$PY" x103_extract.py --tag "$TAG"
