# Which Words a Model Fears: From a Refusal Direction to a Vocabulary-Wide Fear Lexicon

Boyuan Zhang, Mohammad Zandsalimy, Shanu Sushmita
Northeastern University

Paper: arXiv link to be added

This repository holds the code, experiment write-ups and figure scripts behind the paper. The
data (judged generations, intermediate results, hidden-state shards; about 9.6 GB) is hosted
separately as the Hugging Face dataset
[`vhboyr/which-words-a-model-fears`](https://huggingface.co/datasets/vhboyr/which-words-a-model-fears).
From the repository root, download it into `data/`:

```bash
hf download vhboyr/which-words-a-model-fears --repo-type dataset --local-dir data
```

This rebuilds the `data/` tree that the scripts and the layout below refer to. The `hf`
command comes with `huggingface_hub` (pinned in the requirements files). The download also
places the dataset card at `data/README.md`.

## Layout

| Path | Contents |
|---|---|
| `code/` | Experiment scripts (`xNN_*.py`, `xNN.sbatch`), shared helpers (`code/lib/`) and unit tests (`code/tests/`) |
| `data/runs/xNN/` | Per-experiment run artifacts: frozen prompts, texts, generations, judge labels, projections, tables; for x109–x113, x116, x117 and x118 also the final `results.{json,md}` (x110–x113 and x116 with `results_full.{json,md}` siblings; x117 with its pooled and x111-check siblings; x118 with `results_haiku.{json,md}`); x107 results under `data/runs/x107/*_analysis.{json,md}` |
| `data/results/` | Final result JSON (and figures) of x80, x83, x85–x89, x100–x106, x108, x114 and x115 |
| `data/datasets/` | The frozen harmful prompt set (`x61_majority_harmful_v1`: prompts, manifest, exclusions) |
| `data/pools/` | Source prompt pools the cohort builders read |
| `data/labels/`, `data/judged/` | Per-model label files and judged generations of the earlier experiments |
| `docs/experiments/` | One write-up per experiment ([index](docs/experiments/README.md)): design, numbers and checks; the tables they link live in `docs/experiments/json/` |
| `docs/infrastructure.md` | Hardware, sampling contract, judge and drafter settings, provider routing, GPU hours |
| `figures/` | Plotting scripts and the final figure PDFs used in the paper |
| `requirements*.txt`, `ENVIRONMENT.md` | Pinned Python environments and setup |

Everything under `data/` comes from the Hugging Face dataset (see the download command above),
including the large binary files (`data/runs/x85/features/*.pt`, `data/runs/x100/features/*.pt`,
`data/runs/x107/*.npz`, `data/pools/cp_15k_clustered_v2.json`).

## Experiment index

| ID | Topic |
|---|---|
| x80 | Refusal/harm contrast direction on Qwen |
| x81 | Self-recoverable lexical trigger (generation and refusal-judge modules shared by later experiments) |
| x83 | Keyword insertion into external benign prompts |
| x85 | Crossed keyword x prompt latent decomposition (direction fitting cohort) |
| x86 | Crossed keyword boundary intervention |
| x87 | External safety and over-refusal evaluation (OR-Bench, StrongREJECT, GSM8K, IFEval) |
| x88 / x89 | Eight-model dose transfer, positive-dose extension |
| x100 | Lexical "courage" words |
| x101 | Cross-model keyword insertion |
| x102 | OR-Bench judge agreement (API and open-weight judges, XSTest human labels) |
| x103 / x105 | Cross-model direction projection and causal specificity |
| x104 | GuardReasoner-8B re-judge of the OR-Bench responses |
| x106 / x108 | Natural lexical carriers in benign and toxic prompts |
| x107 | Attention read vs. late-MLP write |
| x109 | Vocabulary scan of the boundary direction |
| x110 / x111 / x112 | Table courage words, table dose-response, cross-model tables |
| x113 / x116 | Cross-model table substitution and random-synonym control |
| x117 | Table choice replicated on prompts that chose none of the tables |
| x118 | Meaning preservation of the substituted prompts, read by two independent judges |
| x114 | Leak-free insertion top-up |
| x115 | Direction geometry |

Cluster-specific paths and accounts are replaced by placeholders (`<workdir>`, `<your-slurm-account>`).
Model weights are public Hugging Face checkpoints and are not included.

## Reproducing x113 and x116

Environment: [ENVIRONMENT.md](ENVIRONMENT.md). The CPU path needs
[requirements-api.txt](requirements-api.txt); the GPU stages need
[requirements.txt](requirements.txt). Hardware, sampling, judge settings and
provider routing are in [docs/infrastructure.md](docs/infrastructure.md).

Run every command from the repository root. The API scripts read
`OPENROUTER_API_KEY` from a `.env` file at the repository root; the file must
exist for the CPU path too, and a placeholder value works there because every
drafter call is served from the shipped cache:

```bash
echo "OPENROUTER_API_KEY=placeholder" > .env
```

`HF_TOKEN` is needed only to download the gated Gemma model for a full rerun.

### CPU-only path from the shipped artifacts

The stages rewrite the files they produce under `data/runs/x113/` and
`data/runs/x116/` from the shipped inputs (x106 cohort and labels, x111 and
x112 tables, the drafter caches, the shipped generations and judge labels).

```bash
# x113
python code/x113_select.py                 # prompts.json (core, llama_extra, xstest)
python code/x113_nominate.py --tables      # tables/<tag>.npz, tables.json
python code/x113_nominate.py               # <tag>/nominations.json
python code/x113_build_texts.py --tags qwen qwen08b qwen2b qwen9b qwen27b \
    --drafts data/runs/x113/drafts_a.jsonl
python code/x113_build_texts.py --tags llama gemma phi \
    --drafts data/runs/x113/drafts_b.jsonl --read-drafts data/runs/x113/drafts_a.jsonl
python code/x113_exclusions.py             # x85_exclusions.json (needs the texts)
python code/x113_analyze.py                # results.{json,md}, figures
python code/x113_analyze.py --full         # results_full.{json,md}, figures

# x116 (reads the x113 texts, labels and exclusions above)
python code/x116_build_texts.py            # <tag>/texts.json (R_rand, no API call)
python code/x116_analyze.py                # results.{json,md}
python code/x116_analyze.py --full         # results_full.{json,md}
```

The two `x113_build_texts.py` calls use the draft cache of each model group as
the run did: `drafts_a.jsonl` for the five Qwen models, `drafts_b.jsonl` with
`drafts_a.jsonl` as fallback for Llama, Gemma and Phi. `drafts.jsonl` is the
concatenation of the two. With these caches no drafter request reaches the
API and every `texts.json` reproduces its recorded `texts_sha256`; the
regenerated `results.json` and `results_full.json` of x113 and x116 match the
shipped files in every value.

### Full rerun

1. **Workspace.** The sbatch files expect a copy of this repository at
   `<workdir>/x113` (code in `<workdir>/x113/code`, data in
   `<workdir>/x113/data`), the GPU venv at `<workdir>/x100/venv`
   (built from `requirements.txt`) and the Hugging Face cache at
   `<workdir>/hf_cache`. Replace `<workdir>` and `<your-slurm-account>` in
   `code/x113.sbatch` and `code/x116.sbatch`.
2. **Models.** The model id and revision of each tag are recorded in
   `data/runs/x106/x106_inputs.json` (`models.<tag>.model_id`,
   `models.<tag>.revision`; the table is also in
   [docs/infrastructure.md](docs/infrastructure.md#models)). Download each at
   that revision into the cache, then point the recorded snapshot paths at it:

   ```bash
   export HF_HOME=<workdir>/hf_cache
   python - <<'PY'
   import json
   from huggingface_hub import snapshot_download
   m = json.load(open("data/runs/x106/x106_inputs.json"))["models"]
   for tag, e in m.items():
       print(tag, snapshot_download(e["model_id"], revision=e["revision"]))
   PY
   sed -i 's#<workdir>#/absolute/path/to/workdir#g' data/runs/x106/x106_inputs.json
   ```

   `snapshot` then resolves to
   `<workdir>/hf_cache/hub/models--<org>--<name>/snapshots/<revision>`, the
   path the generation and projection scripts load.
3. **Fresh outputs.** Generators and judges resume from existing files, so
   move the shipped `data/runs/x113/<tag>/{generations,judge}` and
   `data/runs/x116/<tag>/{generations,judge}` aside before a rerun.
4. **x113 stages.**
   - CPU: `x113_select.py`, `x113_nominate.py --tables`, `x113_nominate.py`,
     `x113_build_texts.py` (as above; a cache miss calls the drafter with a
     real `OPENROUTER_API_KEY`), `x113_exclusions.py`.
   - GPU, one model per job (`TAG` one of `qwen qwen08b qwen2b qwen9b qwen27b
     llama gemma phi`; `--mem=140G` for `qwen27b`):

     ```bash
     sbatch --time=02:00:00 --partition=<partition> --export=ALL,TAG=qwen,STAGE=project code/x113.sbatch
     sbatch --time=08:00:00 --partition=<partition> --array=0-7 --export=ALL,TAG=qwen,STAGE=gen,NSHARDS=8 code/x113.sbatch
     ```

     `STAGE=genloop` (with `NSHARDS=8`) runs a work-stealing worker over the
     same eight shards instead of a fixed array.
   - API: `python code/x113_judge.py --tag <tag> --judge refusal2` and
     `python code/x113_judge.py --tag <tag> --judge uptake`, then
     `x113_analyze.py` and `x113_analyze.py --full`.
5. **x116 stages.** `x116_build_texts.py` (CPU); per model
   `sbatch ... --export=ALL,TAG=<tag>,STAGE=genloop,NSHARDS=8,PROJECT=1 code/x116.sbatch`
   (generation, then the projection) or `STAGE=project` alone;
   `python code/x116_judge.py --tag <tag>`; `x116_analyze.py` and
   `x116_analyze.py --full`.

### OpenRouter provider routing

- **x113 drafter** (`x113_build_texts.py`, through `x106_build_texts.call_json`):
  default OpenRouter route; the request body has no `provider` field. Model
  `deepseek/deepseek-v4-flash-0731`, temperature 0, seeded.
- **x113 judges** (`x113_judge.py`, `refusal2` and uptake): default route; no
  `provider` field, no `judge_provider` on the records.
- **x116**: no drafter call (the candidates are x113's). The `refusal2` judge
  (`x116_judge.py`) pins the upstream to DeepInfra by default
  (`--provider DeepInfra`, request body
  `provider: {"order": ["DeepInfra"], "allow_fallbacks": false}`, the same pin
  as x114) and records `judge_provider` on every record.

### x118 from the shipped pairs and judge labels

Both passes recompute end to end from the shipped files; the `refusal2` labels
they read are x106's, x108's and x113's, and the regenerated results match the
shipped `results.json` and `results_haiku.json` in every value:

```bash
python code/x118_analyze.py                # data/runs/x118/results.{json,md}
python code/x118_analyze.py \
    --judge-file data/runs/x118/judge/meaning_haiku.jsonl \
    --retest-file data/runs/x118/judge/meaning_haiku_retest.jsonl \
    --compare-with data/runs/x118/judge/meaning.jsonl \
    --out-stem results_haiku --allow-unlabelled 2
```

`python code/x118_collect_pairs.py` rebuilds `pairs.jsonl` from the x106, x108
and x113 texts, and `python code/x118_judge.py` (optionally
`--judge-model anthropic/claude-haiku-5.5`, `--retest 0.1`) calls the
judge again; both need a real `OPENROUTER_API_KEY`.

### x117 from the shipped artifacts

x117 ships its prompts, words, key-word scan, the 48 generation shards (2,624
cells), judge labels and results, so the analysis recomputes from the shipped
files and matches the shipped `results.json` in every value:

```bash
python code/x117_analyze.py                    # data/runs/x117/results.{json,md}
python code/x117_analyze.py --runs x111,x117   # results_pooled_x111_x117.{json,md}
python code/x117_analyze.py --runs x111        # results_check_x111.{json,md}
```

The last call runs x117's loader on x111's cohort and reproduces x111's
primary numbers; the tables and the 40 words are x111's throughout
(`data/runs/x111/tables.{json,npz}`). A full rerun is x111's pipeline under
`--run x117` (move the shipped `data/runs/x117/{generations,judge}` aside
first; replace `<partition>`):

```bash
python code/x117_select.py --prompts       # prompts.json, words.json
sbatch --time=04:00:00 --partition=<partition> --array=0-15 \
    --export=ALL,STAGE=genloop,NSHARDS=48 code/x117.sbatch
python code/x111_judge.py --run x117 --judge refusal2 --provider GMICloud
```

### x102 and x104 from the shipped labels

```bash
python code/x102_analyze.py                # data/results/x102_judge_agreement.json and the deltas figure
python code/x104_analyze.py --skip-merge   # data/results/x104_*.json from the shipped label file
```

## Citation

```bibtex
@misc{zhang2026whichwords,
  title         = {Which Words a Model Fears: From a Refusal Direction to a Vocabulary-Wide Fear Lexicon},
  author        = {Zhang, Boyuan and Zandsalimy, Mohammad and Sushmita, Shanu},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv}
}
```
