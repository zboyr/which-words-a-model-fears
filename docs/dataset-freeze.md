# The frozen prompt set and sampling contract

Parent: [Experiment index](experiments/README.md).

Frozen 2026-08-17, before any experiment in this project ran. Every cohort
in this paper — the keyword-insertion cohort, the crossed prompt–word grid,
the intervention cells, and the benign calibration and sealed benign sets —
is drawn from the prompt set described here, and every generation follows
the sampling contract below. Everything in it is verified against the
artifacts.

## Composition

**13,527 harmful + 10,500 benign = 24,027 rows.**

### Harmful side (real, continuous risk target T)

- Source pool: the 15,000 `source == "toxic"` rows of
  `data/pools/cp_15k_clustered_v2.json`, sorted by id (positional contract
  below). That file is too large for git and is kept on disk only; its
  provenance is [the mining line](experiments/dataset-mining.md) and
  [the scale-out](experiments/dataset-scaleout-validity.md).
- Retention: a three-judge majority vote (deepseek / glm / luna judges;
  majority over collapsed P/X codes) retained 13,527 rows. Canonical
  artifact: `data/datasets/swse_x61_majority_harmful_v1/` — `prompts.jsonl`
  (retained), `excluded.jsonl` (all 1,473 dropped rows with per-judge codes
  and exclusion_reason: 1,398 dropped_majority_PX + 75
  all_collapsed_different), `manifest.json` (counts and builder). The
  builder script belongs to the original probe project and is not kept
  here; its output is.
- Enforcement in the pipeline: the 1,473 non-retained rows remain inside the
  30,000-row feature caches but carry `False` in `data/x61_mask_tbg.npy`,
  which the overlay builder (`code/x66_build_v4v5_overlays.py`) bakes into
  every label overlay cache and `code/x66_split_cache_masked.py` drops
  before any split. No cohort in this paper ever sees them.

### Benign side (T = 0 by construction)

- Source pool: the 15,000 `source == "hard"` rows of
  `data/pools/cp_15k_benign.json`, sorted by id.
- Subset: the FIRST 10,500 rows of the fixed
  `np.random.RandomState(42).permutation` of the benign rows
  (`code/x66_split_cache_masked.py --benign_n`). Subsets are nested across N
  by construction.
- The count 10,500 was selected on one model and one split by maximizing a
  probe-minus-lexical-baseline separation gap, then reused unchanged
  everywhere.
- This side supplies the benign calibration prompts, the sealed benign
  over-refusal set, and the benign-KL budget prompts of the interventions.

## Positional contract (all caches and splits)

Harmful rows first (sorted by id), then benign rows (sorted by id) — the
`code/extract_features_v2.py` contract. All label arrays, masks, and splits
index into this fixed 30,000-row order; the freeze is realized as (the
retention mask) + (benign seed-42-permutation prefix), never as a physically
re-written cache.

## Labels on the harmful side

Built per model from the judge outputs
`data/judged/x64_{refusal2,comply2}_qwen_s42.jsonl` by the label-building
stage of the original probe campaign; the resulting frozen label files are
checked in under `data/labels/{tag}_labels_v4.json` and `_v5.json` and are
read, never rebuilt, here. Judge: deepseek/deepseek-v4-flash-0731,
reasoning effort low, verbatim `refusal2` + `comply2` prompts, pass seed 42.
Coverage at freeze: 13,527/13,527 non-failed on BOTH methods, zero duplicate
ids. The label constructs themselves are owned by
[refusal and compliance surface labels](experiments/labels-refusal-comply-x58.md).

The refusal rate this paper supervises on — `r` = the fraction of ten
sampled responses judged to contain an explicit refusal — is the `refusal2`
side of that file; the v4 target is a blend built from it. T is continuous
everywhere (never binarized for training); benign rows are exactly 0.

## Response side and sampling contract

The frozen responses behind the labels: 10 responses per prompt on the
13,527 harmful prompts, T = 1.0, top_p 0.9, top_k 50, 2,048-token first
pass, whole-row 4,096 retry, still-truncated-at-4,096 kept and judged as-is.
The generation payload is `data/prompts/x63_prompts13527.json`; the raw
responses are 3.1 GB and live on disk outside git.

**Every generation experiment in this project reuses that contract
verbatim** (same temperature, top-p, top-k, token budgets, retry rule, and
the keep-truncated policy), with per-experiment seed namespaces, so a new
arm is comparable to the historical record and to the other arms. The
intervention and external-benchmark runs are the exception by design: they
generate greedily; the owning experiment file states which.

## Split protocol frozen with the prompt set

Split seeds 42/123/7 (`code/x66_split_cache_masked.py`, mask + benign_n
applied before splitting). **This paper uses seed 42 only.** Its harmful
evaluation split (1,353 prompts) is the universe every prompt cohort here is
drawn from, and its benign train/test portions supply the calibration and
sealed benign sets. The seed-42 index files are checked in under
`data/runs/x66/`.

## Where the physical artifacts live

- Seed-42 split indices: `data/runs/x66/qwen_seed42_{train,test}_indices.json`
  and the same for the other model tags.
- Feature caches (30,000 rows each) are 46 GB and live outside git; see
  [infrastructure.md](infrastructure.md).
