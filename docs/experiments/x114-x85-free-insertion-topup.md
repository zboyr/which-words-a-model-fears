# x114: an x85-free top-up of the cross-model insertion cohort, so x103 and x105 rest on 136 units instead of 64

Parent: [Experiment index](README.md). A second cohort for
[x101](x101-cross-model-keyword-insertion.md)'s design, built so that
[x103](x103-cross-model-direction-insertion-projection.md) and
[x105](x105-cross-model-direction-causal-specificity.md) can be read on prompts
and words the per-model boundary directions were never fit on without losing
sample size.

The directions those two experiments project onto and write were fit on the
[x85](x85-crossed-keyword-prompt-latent-decomposition.md) crossed cohort (96
prompts × 24 harm words), and the x101 insertion cohort shares 47 of those
prompts and 17 of those words, so the leak-free analysis ran on 64 of x101's 130
units. x114 adds **72 units on campaign prompts that are neither x83 nor x85
prompts, with words drawn from the x83 lexicon minus x85's 48**, runs x101's
generation, gate and judge contract on them in all eight models, and then x103's
boundary read and x105's eight dosed arms. The primary cohort of x103 and x105 is
now **the 64 x101 leak-free units plus these 72**.

## Design, as run

- **Prompt pool.** The 13,527 campaign prompts the eight per-model v4 label
  files cover (`data/labels/<tag>_labels_v4.json`), kept where the historical
  `refusal2` rate is ≤ 0.8 in **every** one of the eight models (x101's
  `MAX_HISTORICAL_RATE` rule, selection reading historical labels only) and the
  id is neither one of x83's 321 cohort prompts nor one of x85's 96 (the x85
  prompts are a subset of x83's): **1,220 eligible**. x83's cohort was drawn
  from the x66 seed-42 test split, so every one of the 1,220 lies outside that
  split; the split is only the x66 probes' and nothing in x103 or x105 reads it.
  Per-model eligibility over the whole pool is recorded in the cohort file.
- **Words.** The x83 lexicon (99 words) minus x85's 24 harm words; x85's 24
  neutral twins are not lexicon words, so none of them is in it either —
  **75 words remain**.
- **Draw.** One `numpy.random.default_rng(114)`: 72 prompt ids without
  replacement from the sorted eligible ids, then one word per prompt, uniform
  over the pool in frozen lexicon order, in ascending id order. No
  stratification. 45 distinct words, 13 categories (largest
  `illegality_markers` 13, `fraud_property_crime` 9, `chemical_biological` 8,
  `drugs` 8). The O/S/M/E strings are x83's `insert_word` rule, and the builder
  re-derives every insertion and its offset from the original.
- **Models.** All eight, from the frozen x88 snapshots x101/x103/x105 ran on
  (still in the shared the cluster cache); **Qwen3.5-4B runs fresh here**, where x101
  imported its rows from x83. No preflight is rerun: x101's preflight found no
  fixed pre-answer tokens in any template, so the gate scoring offset is 0
  everywhere, and the x114 manifest carries those constants
  (`data/runs/x114/x114_manifest.json`). Every generation file records the
  scaffold its own job rendered, with its sha256.
- **Detection gate.** x83's template, surface variants, statistic and threshold
  (`P_detect > 0.90`), per model with its own template.
- **Generation.** x101's contract (T = 1.0, top-p 0.9, top-k 50, 10 sequences,
  2,048 tokens, one 4,096 retry), seeds per (tag, prompt, condition) in an
  `x114` namespace, base 114. 288 prompt-conditions per model.
- **Judge.** The x64 `refusal2` contract exactly as x101 ran it
  (`deepseek/deepseek-v4-flash-0731`, reasoning low, seeds 42/1042/2042), run
  locally.
- **Boundary read.** x103's stage, unchanged: a forward hook on the decoder
  block the x88 boundary write hooks, the block output at the two trailing
  prefill positions, 288 forward passes per model, with x103's own in-job site
  audit (the read is bitwise `output_hidden_states[block + 1]` at those
  positions, the x88 write moves it by exactly `dose · μ` to bf16 rounding, it
  fires once on prefill, removing it restores the forward). Every rendering is
  checked against the `rendered_chat_sha256` the x114 generation row recorded.
- **Dosed arms.** x105's eight arms at x105's doses (E at −0.75 and −1.5, O at
  +0.75 and +1.5, two norm-matched orthogonal random directions on E at −1.5,
  one of them on O at +1.5, and one KL-matched random direction on E), with
  x105's own per-model control directions and benign-KL calibration reused
  unchanged — the controls are deterministic in (tag, draw, position) and the
  calibration is measured on the 256 frozen x86 benign prompts, so neither
  depends on the cohort. Dose 0 is not regenerated: the x114 base O and E arms
  are the dose-0 arms, and every dosed arm reuses that arm's own x114 seed
  (x86's common-random-number contract).
- **Exclusion check.** The x105 exclusion rule
  (`x103_common.leakage_sets`) is applied to the x114 cohort as a check and
  excludes **0 of 72** units, as construction requires
  (`data/runs/x114/x114_x85_exclusions.json`).
- **Analysis.** `code/x114_analyze.py` for the behavioural table below (x101's
  estimator, 10,000-replicate two-stage paired bootstrap, Holm within model and
  over the 24 model × position tests). The direction results are x103's and
  x105's own analyses in their `extended` cohort mode
  (`code/x114_cohorts.py`), which pools these units with the 64 x101 leak-free
  ones and keeps every bootstrap, permutation and Holm scheme as it was.

## Results

All 2,304 base prompt-conditions carry a successful `refusal2` record (one
phi / E row needed a fourth pass after transport timeouts). The judge route
needed one operational change: OpenRouter's default upstream for the judge
model (OpenInference, fp4) returned label arrays shorter than ten on 15–90 % of
the long rows during this run, with the same model, prompt and parameters that
had failed on 1 of 3,640 rows in x101, while every other upstream returned ten
labels on 11–12 of 12 probe rows; all x114 judgments are therefore served by one
pinned upstream (DeepInfra, fp8; `judge_provider` on every record), and on the
583 rows the default route had labelled successfully before the switch the two
agree on 0.993 of responses (0.961 of rows). Judge spend ≈ $3.6 for the base
and dosed arms together on the pinned route, ≈ $1 more on the default route
before the switch.

Table: [The inserted word's effect on the x114 units, per model and position](json/x114-x85-free-insertion-topup__insertion-deltas.json)
— ungated `refusal2` deltas against each model's regenerated original, with
the gate admissions; the end position raises refusal in every model, from
+0.076 (Phi) to +0.501 (Qwen3.5-9B), Holm p over the 24 tests < 1e-4
everywhere except Phi (1.6e-3), and the start and middle are null on
Qwen3.5-0.8B (+0.025, +0.008) and Phi (+0.031, +0.031), as in x101.

The units behave like x101's. End-position deltas per model (x101's 130-unit
values in parentheses): Qwen3.5-0.8B +0.108 (+0.099), 2B +0.375 (+0.38), 4B
+0.340 (+0.33 on x83's rows), 9B +0.501 (+0.43), 27B +0.475 (+0.489), Llama
+0.276 (+0.29), Gemma +0.138 (+0.18), Phi +0.076 (+0.11); the middle is the
weakest position in every model and the end the strongest in all but
Qwen3.5-4B and 27B, where start and end are within noise, and the three
off-family models start from the floor (regenerated O 0.017–0.025, 63–67 of
72 prompts at 0/10). The run-era diagnostic (regenerated O − historical O) is
within ±0.032 for every model with every 95 % interval covering zero (largest
Qwen3.5-4B +0.032 [−0.010, +0.075]). The detection gate admits the same
pattern as x101: at E 65 / 3 / 5 / 69 / 71 / 29 / 71 / 56 of 72 for 4B / 0.8B /
2B / 9B / 27B / Llama / Gemma / Phi, at S and M nothing on 0.8B, 2B, Llama and
Phi (≤ 3), and the greedy answer names the word at E on 0.69 (Llama) to 1.00 of
the prompts. Generation integrity as in x101: Qwen3.5-0.8B's responses average
1,130–1,220 tokens with 23–29 retry rows and 12–15 responses still truncated
per arm; every other model has ≤ 7 retry rows and ≤ 10 truncated responses per
arm.

The direction results on these units are in the x103 and x105 write-ups,
whose primary cohort they now form together with the 64 x101 leak-free units
(136 units, 408 x103 cells per model).

## Artifacts

- Frozen cohort: `data/runs/x114/x114_cohort.json`; manifest
  `x114_manifest.json`; exclusion check `x114_x85_exclusions.json`; frozen
  downstream inputs `x114_inputs.json`.
- Per model: `data/runs/x114/<tag>/x114_detect.json`,
  `x114_generations_{O,S,M,E}.json` and the eight `x114_generations_<arm>.json`,
  `x114_refusal_*.jsonl`,
  `x114_projections.json`, `x114_boundary_deltas.npz`.
- Result: `data/results/x114_insertion_topup.json`.
- Code: `code/x114_common.py`, `x114_build_cohort.py`, `x114_generate.py`,
  `x114_judge_refusal.py`, `x114_analyze.py`, `x114_prepare.py`,
  `x114_extract.py`, `x114_dosed.py`, `x114_cohorts.py`, `x114.sbatch`; tests
  `code/tests/test_x114_cohort_topup.py`.
