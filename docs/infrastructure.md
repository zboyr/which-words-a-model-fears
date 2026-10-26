# Compute and API settings

The settings below are the ones the experiment write-ups in
[experiments/](experiments/) refer to. Package versions and setup steps are in
[ENVIRONMENT.md](../ENVIRONMENT.md); the step-by-step rerun of x113 and x116 is
in the [README](../README.md#reproducing-x113-and-x116).

## Hardware

- **GPU stages** (generation, hidden-state extraction, projections, vocabulary
  scans, the GuardReasoner-8B judge) ran on a Slurm cluster, one GPU per job:
  NVIDIA B200 (180 GB) or NVIDIA RTX PRO 6000 Blackwell Server Edition
  (96 GB). Every model fits on one GPU in bfloat16. The device of each
  generation shard is recorded in the `environment` block of its file.
- Memory per job: 80 GB by default in the sbatch files; Qwen3.5-27B jobs use
  `--mem=140G`. Each sbatch file reads `<your-slurm-account>` and `<workdir>`
  placeholders; set them before submitting, and pass `--time` and the
  partition at submit time.
- All `#SBATCH` directives sit at the top of each file; Slurm ignores a
  directive placed after the first shell command.
- **CPU stages** (cohort selection, prompt drafting, API judges, statistics,
  figures) need no GPU and ran on a workstation.
- Compute nodes had no internet access: model snapshots were downloaded once
  into `<workdir>/hf_cache`, and the GPU jobs run with `HF_HUB_OFFLINE=1`.

## Python

- GPU stages: CPython 3.12.14, `torch==2.11.0+cu128`, transformers 5.15.0,
  accelerate 1.15 ([requirements.txt](../requirements.txt)). The earlier
  probe-fitting and external-benchmark stages (x71–x89) use
  [requirements-eval.txt](../requirements-eval.txt).
- CPU and API stages: CPython 3.12 with
  [requirements-api.txt](../requirements-api.txt).
- Model loading needs `accelerate` installed (transformers 5.x
  `device_map="auto"`); a venv check should load a model, not only import the
  packages.

## Models

Every model is a public Hugging Face checkpoint loaded at a fixed revision.
The model id, the revision and the snapshot path of each model are recorded in
the frozen inputs of the experiment (for x106, x108, x113 and x116:
`data/runs/x106/x106_inputs.json`, field `models.<tag>`), and the snapshot
path is repeated as `model_source` in every generation and projection file.
Snapshot paths are written as `<workdir>/hf_cache/hub/models--<org>--<name>/snapshots/<revision>`.

| Tag | Model id |
|---|---|
| `qwen` | `Qwen/Qwen3.5-4B` |
| `qwen08b` | `Qwen/Qwen3.5-0.8B` |
| `qwen2b` | `Qwen/Qwen3.5-2B` |
| `qwen9b` | `Qwen/Qwen3.5-9B` |
| `qwen27b` | `Qwen/Qwen3.5-27B` |
| `llama` | `unsloth/Llama-3.2-3B-Instruct` |
| `gemma` | `google/gemma-4-e2b-it` |
| `phi` | `microsoft/Phi-4-mini-instruct` |

The GuardReasoner judge is `yueliu1999/GuardReasoner-8B` at revision
`f3f57a9c3e94c94dd029903b667aa144b7d7dfb8`. Gated repositories (the Gemma
models, the StrongREJECT evaluator base) need their licence accepted on the
downloading account and `HF_TOKEN` set for the download.

## Sampling contract

- Ten sampled responses per prompt-condition: temperature 1.0, top-p 0.9,
  top-k 50, `num_return_sequences=10`, 2,048 new tokens, and one retry at
  4,096 new tokens for a row whose responses were truncated. Thinking is off
  in the chat template (`enable_thinking=False` where the template has it).
- x106, x108, x113 and x116 seed every row with x106's `stable_seed` on
  (model tag, prompt id, text label), so a text byte-identical to an earlier
  arm's text samples with the same seed and its generation row is copied.
- Generators resume by cell or prompt id from their output file; a rerun into
  an existing output path continues it rather than starting over.

## API judges and drafter

All API calls go to OpenRouter (`https://openrouter.ai/api/v1/chat/completions`)
with `response_format: json_object`.

- **API key.** The API scripts read `OPENROUTER_API_KEY` from a `.env` file at
  the repository root (one line `OPENROUTER_API_KEY=...`). The file must exist
  even for a run served entirely from the shipped caches; any placeholder value
  works then.
- **`refusal2` judge** (`code/x81_judge_refusal.py`, the label of record for
  refusal): model `deepseek/deepseek-v4-flash-0731`, reasoning
  `{"effort": "low"}`, ten responses per call, attempt seeds 42, 1042, 2042,
  output validated as exactly ten booleans. `--per-response-fallback` judges
  one response per call for rows whose ten-label arrays stay malformed.
- **Word-uptake ("hijack") judge** (x110–x113): the same model with reasoning
  disabled (`reasoning: {"enabled": false}`), one call per cell with all ten
  responses; output `hijack_fast.jsonl` (x110–x112) or `uptake.jsonl` (x113).
  `refusal2` and the uptake judge run in separate loops at up to 24 threads.
- **Drafter** (x106, x108, x113; `code/x106_build_texts.py`): model
  `deepseek/deepseek-v4-flash-0731`, temperature 0, per-request seed,
  `max_tokens` 16,000. When the reasoning uses the whole budget without an
  answer, the next attempt runs with reasoning disabled.
- **Provider routing.** OpenRouter's default route serves a request from any
  upstream provider of the model. A judge script with a `--provider NAME`
  option adds `provider: {"order": [NAME], "allow_fallbacks": false}` to every
  request body and records the name as `judge_provider` on each record; model,
  prompt, parameters and seeds are unchanged. Routing per experiment:

  | Stage | Routing |
  |---|---|
  | x114 `refusal2` (`x114_judge_refusal.py --provider`) | pinned to DeepInfra |
  | x110 and x111 top-up passes (`--provider`; the records carrying `judge_provider`) | pinned to DeepInfra |
  | x110 and x111 original passes | default route |
  | x116 `refusal2` (`x116_judge.py`, default `--provider DeepInfra`) | pinned to DeepInfra |
  | x113 `refusal2` and uptake (`x113_judge.py`) | default route (no `provider` field) |
  | x106, x108, x113 drafter | default route (no `provider` field) |
  | x112 and all judging before x114 | default route |

  The default route for `deepseek/deepseek-v4-flash-0731` returned label
  arrays shorter than ten on long ten-response rows under 24 concurrent calls
  in October 2026, which is why x114 and later judging pin the provider; the
  pinned labels agree with the default route's successful ones on 0.993 of
  responses (x114). Two concurrent pinned judge loops are kept below 48
  threads in total.
- **Caching and resume.** Judges append one JSON record per row to a
  `.jsonl` file, skip rows already judged successfully, and retry failed rows
  on a resume pass (`--seed-offset` shifts the attempt seeds). Each record
  carries the sha256 of the generation row it labels. The drafter caches every
  exchange in a `.jsonl` keyed by the sha256 of the system prompt, user
  message and seed (`data/runs/x106/x106_substitute_drafts.jsonl`,
  `data/runs/x108/x108_substitute_drafts.jsonl`, `data/runs/x113/drafts*.jsonl`); a cached request is never sent again.
  Spend is logged per pass in `judge/cost_log.jsonl`.

## GPU hours

GPU totals per experiment on the cluster, from the write-ups:

| Experiment | GPU-hours |
|---|---|
| x100 | ≈ 14.7 |
| x101 | ≈ 19.5 |
| x103 | ≈ 0.2 |
| x104 | ≈ 10 (96,726 GuardReasoner judgments) |
| x105 | ≈ 54 |
| x106 | ≈ 142 (Stage 0 ≈ 2, generation ≈ 101, GuardReasoner ≈ 39) |
| x107 | ≈ 0.4 |
| x108 | ≈ 26.5 |
| x110 | ≈ 16 plus 1.5 for the top-up |
| x111 | ≈ 11 plus 2.3 for the top-up |
| x112 | 30.9 plus 2.3 for the top-up |
| x113 | 22.0 |
| x114 | ≈ 38.8 |
| x116 | 12.1 |

HF `generate` runs at roughly 215–255 ten-sample rows per GPU-hour on either
GPU class. API spend per experiment stayed under about $30 (x113: $13.12 in
all, of which drafter $2.46, `refusal2` $6.04, uptake $4.62).
