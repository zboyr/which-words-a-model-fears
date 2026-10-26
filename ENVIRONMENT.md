# Environment

All GPU stages ran on CPython 3.12.14 with PyTorch 2.11.0 (CUDA 12.8 wheels,
`torch==2.11.0+cu128`) and transformers 5.15.0. The versions are recorded in the
`environment` block of every generation file under `data/runs/`. The pinned
files below are taken from the installed environments used for the runs.

| File | Packages | Stages |
|---|---|---|
| `requirements.txt` | 57 | GPU stages of x100–x116: sampled generation, hidden-state extraction, direction projection and vocabulary scans, activation interventions, attention analyses, the local GuardReasoner judge, tokenizer-only cohort building, and `code/tests/` |
| `requirements-eval.txt` | 152 | Earlier GPU stages x71–x86 (rank-one edits, probe fitting with scikit-learn, keyword-insertion and boundary-intervention runs) and the external benchmarks x87–x89 (OR-Bench, StrongREJECT, lm-eval GSM8K/IFEval) |
| `requirements-api.txt` | 13 | CPU-only stages: OpenRouter API calls (prompt drafter, refusal and word-uptake judges), cohort selection, statistics, bootstrap intervals and figures |

`requirements-eval.txt` is the generation stack of `requirements.txt` at the
preceding patch releases (accelerate 1.14.0, huggingface_hub 1.27.0, numpy
2.5.2) plus datasets 5.0.1, lm_eval 0.4.12, scikit-learn 1.9.0 and the
StrongREJECT evaluator package pinned to a GitHub commit. The API stages ran on
CPython 3.12 and need no GPU.

## Hardware

Generation and extraction ran on single GPUs, NVIDIA B200 (180 GB) and NVIDIA
RTX PRO 6000 Blackwell Server Edition (96 GB), with NVIDIA driver 595.71.05
(CUDA 13.2 runtime support). Every model fits on one GPU in bfloat16.

## Setup

With [uv](https://docs.astral.sh/uv/):

```bash
uv python install 3.12.14
uv venv --python 3.12.14 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

With pip:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Create separate virtual environments for `requirements-eval.txt` and
`requirements-api.txt` in the same way (for example `.venv-eval` and
`.venv-api`). The `--extra-index-url` line in the GPU files fetches the CUDA
12.8 build of PyTorch.

API stages read `OPENROUTER_API_KEY` from the `.env` file at the repository
root; a placeholder value works when every call is served from the shipped
caches. Models and datasets
download from the Hugging Face Hub; gated repositories (the Gemma models and
the StrongREJECT evaluator base) need their licence accepted and `HF_TOKEN`
set.
