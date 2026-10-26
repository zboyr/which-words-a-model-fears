# x105: in all eight models, writing the model's own boundary direction removes the inserted word's refusal and creates it on the clean prompt, dose-ordered and direction-specific, on 136 prompt–word units the direction was never fit on

Parent: [Experiment index](README.md). The causal counterpart of
[x103](x103-cross-model-direction-insertion-projection.md): x101 showed the
lexical reaction in all eight models, x103 showed that the shift an inserted
word makes along each model's own boundary direction predicts that model's
refusal rise, and x105 writes the direction to test whether it is the
mechanism. The intervention, the direction and the dose units are
[x88](x88-eight-model-deep-dose-transfer.md)'s; the texts, the sampling
contract and the judge are [x101](x101-cross-model-keyword-insertion.md)'s,
on x101's cohort and on [x114](x114-x85-free-insertion-topup.md)'s top-up; the
control construction is
[x86](x86-crossed-keyword-boundary-intervention.md)'s. The design is recorded
here rather than in a plan file, by author decision.

On the 136 units whose prompt and inserted word are both outside the x85
cohort the directions were fit on (the 64 such x101 units and the 72 x114
units, the same units x103 reports), writing the model's own consensus
boundary direction at dose −1.5 on the end-inserted texts removes **+0.087
(Phi) to +0.279 (Qwen3.5-2B)** of `refusal2`, and at +1.5 on the original
texts it creates **+0.040 (Llama) to +0.387 (Qwen3.5-2B)**; both effects are
present in **every one of the eight models**, every 95 % interval excludes
zero, and every Holm-adjusted p over the seven transfer models is 7 × 10⁻⁴ in
all 21 primary tests. The movement is dose-ordered in all eight (half dose
always smaller and positive at the point estimates; bootstrap monotonicity
0.990–1.000). Norm-matched orthogonal random directions at the same dose move
refusal by −0.022 to +0.023, so the target beats them by **+0.086 to +0.301**;
random directions re-scaled to the same benign KL — 1.7× to 4.7× the dose —
move refusal on Qwen3.5-9B (+0.115) and 27B (+0.142) and by at most +0.041
elsewhere, and the target still beats them by +0.055 to +0.299 everywhere. The
write removes **41 % (Llama) to 92 % (Qwen3.5-0.8B, Phi)** of the
insertion-induced rise. Nothing here is routed to
[../headline-numbers.md](../headline-numbers.md).

## Question

x86 established on Qwen3.5-4B that the crossed-keyword boundary shift is
causally usable and direction-specific; x88 carried the write to seven more
models at a single negative dose with no control direction. Does the causal
claim itself transfer? In each transfer model:

1. **Bidirectional.** Does a negative dose on the end-inserted (E) texts
   remove the insertion-induced `refusal2` rise, and a positive dose on the
   original (O) texts create one?
2. **Dose.** Is the movement ordered over dose magnitude on both signs?
3. **Specificity.** Does the model's own direction beat random directions of
   the same norm, and random directions matched on benign KL?

## Design, as run

- **Cohorts and texts.** Two frozen insertion cohorts, two of their four texts
  per prompt: the original **O** and the end-inserted **E**. The end is the one
  position whose rise x101 and x114 found in every model, so the question "did
  the write remove the rise" is posed identically everywhere. The 130-prompt
  x101 cohort (`data/runs/x101/x101_cohort.json`) was generated and judged in
  every arm in the original x105 run; the 72-prompt x114 cohort
  (`data/runs/x114/x114_cohort.json`) in the x114 run, under the identical arm
  table. Every text is re-rendered with the model's own template and checked
  against the `rendered_chat_sha256` its base generation row recorded, so the
  prompt entering the model is byte-identical to the one the dose-0 responses
  came from (the Llama template's date line is recovered from that hash, as in
  x103).
- **Cohort modes.** The direction each model writes was fit on the x85 crossed
  cohort (96 prompts × 24 harm words, cohort sha256 `93784a93…`). The x101
  cohort shares 47 of those prompts and 17 of those harm words (32 units carry
  one); the 66 units with either are excluded and 64 retained, as frozen in
  `data/runs/x105/x105_x85_exclusions.json` (`2305ef73…`, built by
  `code/x105_exclusions.py` from `x103_common.leakage_sets`). The x114 cohort
  was drawn outside x83's prompts (and so x85's) and outside x85's 48 words,
  and the same rule applied to it as a check excludes none
  (`data/runs/x114/x114_x85_exclusions.json`). The analysis runs in three
  modes (`code/x114_cohorts.py`, `--cohort`): `extended` — the 64 x101 units
  pooled with the 72 x114 units, **the primary cohort and the canonical result
  files** (136 prompts per model); `x101_leak_free` — the 64 x101 units alone,
  the `_x101only` siblings; `full` — all 130 x101 units, the `_full` siblings.
  Both exclusion artifacts are re-derived and checked at load and their sha256
  recorded in the result's `cohort` block.
- **Models.** All eight x101 / x88 snapshots: `qwen08b`, `qwen2b`, `qwen9b`,
  `qwen27b`, `llama`, `gemma`, `phi` as the transfer set, and Qwen3.5-4B
  (`qwen`) as the pipeline reference row, outside the Holm family as in x103.
- **Direction and site.** Per model the x88 consensus (mean of the four fold
  directions) from `data/runs/x88/<tag>/x88_directions.npz`, written by x88's
  prefill-only `ExternalBoundaryHook` at the two trailing prefill positions of
  the decoder block the frozen relative-depth rule `round(29/33 · n_blocks)`
  selects; for Qwen3.5-4B the x87 consensus of the x86 fold directions at
  x86's own block 29. Direction norms 0.90 (0.8B) to 42.6 (27B).
- **Doses.** Multiples of the model's own mean boundary shift — the x86/x88
  unit. On E: **−0.75 and −1.5**; on O: **+0.75 and +1.5**. The main magnitude
  1.5 is exactly x88's transfer dose and x89's positive-dose point on 4B.
  **Dose 0 is not regenerated:** the base arms are each cohort's own O and E
  generations and judgments (x101's for the x101 units, x83's for Qwen3.5-4B
  on them, x114's for the x114 units), and every dosed arm reuses that arm's
  per-(tag, prompt, text) seed, so the arms share one random-number stream
  (x86's common-random-number contract).
- **Controls.** Per model, two **norm-matched orthogonal random directions**
  built by `x86_common.orthogonal_random_null` (Gaussian, projected off the
  true direction, rescaled to its norm; seed namespace `x105|<tag>|<draw>`),
  written on E at −1.5, and one of them again on O at +1.5. Plus one
  **KL-matched random direction** on E: on the 256 frozen x86 benign
  calibration prompts rendered with this model's template, the dose of draw 1
  is bisected until its mean `KL(base ‖ written)` at the boundary equals the
  true direction's at −1.5 (x86's matching rule, tolerance
  `max(5e-5, 5 % of target)`), and that dose is written. The control directions
  are deterministic in (tag, draw, position) and the calibration is a
  model-level measurement, so the x114 arms reuse both artifacts unchanged,
  pinned by sha256 in `data/runs/x114/x114_inputs.json`.
- **Arms.** Eight generated arms per model per cohort — `E_true_m075`,
  `E_true_m150`, `O_true_p075`, `O_true_p150`, `E_rand1_m150`,
  `E_rand2_m150`, `E_randkl_m150`, `O_rand1_p150` — plus the two reused dose-0
  arms: 64 generated arms on the x101 cohort (8,320 prompt-conditions) and 64
  on the x114 cohort (4,608), 129,280 responses in total.
- **Generation.** x101's contract unchanged: T = 1.0, top-p 0.9, top-k 50, 10
  sequences, 2,048 tokens with one 4,096-token retry, one prompt per call.
- **Write-site audit, in every job, before the first row.** On two texts: the
  hooked read is bitwise equal to `output_hidden_states[block + 1]` at the two
  boundary positions; the hook moves that tensor by exactly `dose · μ` to bf16
  rounding (cosine ≥ 0.999, norm ratio within 1 %); it fires once on prefill
  and never on a decode step; the logits change; removing it restores the
  forward exactly. A failing audit aborts the job before any cell is written.
  Per-row hook counters are recorded and a write that fires on more than one
  prefill aborts the row. All 128 audits passed (minimum cosine 0.9999, exact
  restore), and every generated row recorded exactly one prefill write.
- **Judge.** The x64 `refusal2` contract exactly as x101 ran it
  (`deepseek/deepseek-v4-flash-0731`, reasoning low, seeds 42/1042/2042), run
  locally. The base arms are never re-judged. On the x101 cohort 8,320 of 8,320
  prompt-conditions carry a successful `refusal2` record (one row, llama /
  `E_true_m150` prompt 43137, returned unparseable JSON once and converged on
  the next pass; judge spend ≈ $14); on the x114 cohort 4,608 of 4,608, served
  by one pinned upstream (DeepInfra; the x114 write-up records why) after
  three resume passes for transport timeouts, ≈ $2.4.
- **Analysis** (`code/x105_analyze.py`). Per model, on the retained prompts
  with successful labels in every arm, the two cohorts' per-arm label matrices
  stacked on the pooled prompt set: `suppress_E(d) = mean_i[r_i(E,0) −
  r_i(E,−d)]`, `induce_O(d) = mean_i[r_i(O,+d) − r_i(O,0)]`, their mean
  (`bidirectional`), the fraction of the insertion-induced rise removed, the
  same suppression under the random and KL-matched directions, the
  target-minus-control differences, and the oriented dose slope over
  magnitudes {0, 0.75, 1.5} (x86's per-cell fixed-effect slope). One
  10,000-replicate two-stage paired bootstrap (prompts, then the ten binary
  responses within each arm) carries every quantity, so a difference between
  contrasts is an interval on the same prompt sample. Holm over the seven
  transfer models within each of the three primary tests
  (`suppress_E_full`, `induce_O_full`, `true_minus_random_E`). Seeds,
  bootstrap and Holm are the same in all three cohort modes.

## Results

n = 136 prompts in every model and every arm (64 x101 + 72 x114).

Table: [Condition means (`refusal2`) per model and arm](json/x105-cross-model-direction-causal-specificity__condition-means.json) — mean `refusal2` on O at doses 0 / +0.75 / +1.5 and under a random direction, and on E at 0 / −0.75 / −1.5 and under the norm-matched and KL-matched random directions; e.g. Qwen3.5-2B E falls from 0.580 to 0.301 at −1.5 while its random arms stay at 0.624 and 0.599.

Table: [Primary tests: suppression on E, induction on O, target − random](json/x105-cross-model-direction-causal-specificity__primary-tests.json) — `suppress E` at −1.5, `induce O` at +1.5 and target − random on E per model with 95 % intervals and Holm p; every interval excludes zero on the predicted side, and the transfer-model Holm p is 7 × 10⁻⁴ in all 21 tests.

Table: [Dose, the KL-matched control, and the share of the rise removed](json/x105-cross-model-direction-causal-specificity__dose-and-controls.json) — half-dose effects, oriented dose slope, bootstrap monotonicity, target − KL-matched random, the random direction on O, and the share of the rise removed; the oriented dose slope's interval excludes zero in every model, monotonicity is 0.990–1.000 and the share removed runs from 0.41 (Llama) to 0.92 (0.8B, Phi).

**What the random directions did on their own.** The two norm-matched draws at
−1.5 move `refusal2` on the inserted texts by −0.022 (Qwen3.5-2B) to +0.023
(Gemma); only Gemma's interval excludes zero (+0.023 [+0.004, +0.046]). On the
original texts the random direction at +1.5 moves refusal by −0.050 (2B, the
one interval excluding zero, on the wrong side) to +0.024, the rest covering
zero. The KL-matched draws are the harder control: matched doses are −2.5
(0.8B) to −7.0 (27B), i.e. 1.7× to 4.7× the target dose, and on Qwen3.5-9B
and 27B they remove +0.115 and +0.142 of refusal on their own, against the
target's +0.223 and +0.249; the target − KL-matched interval excludes zero in
every model (narrowest on 0.8B, +0.055 [+0.028, +0.083]).

Table: [Benign KL of the transferred dose and the matched control](json/x105-cross-model-direction-causal-specificity__benign-kl.json) — direction norms, benign KL of the −1.5 write, the KL-matched random dose and its KL, and the random draws' cosine to the target; matched doses are 1.7× (0.8B) to 4.7× (27B) the target dose and every cosine is ≤ 6e-17.

Benign KL rises monotonically with dose magnitude on both signs in every model
(a model-level calibration on the 256 benign prompts, the same in every cohort
mode and shared by both cohorts' arms).

**Insertion rise on the retained units, and output integrity.** The base arms
are each cohort's own rows; on the 136 units the end insertion raises
`refusal2` by +0.105 (0.8B), +0.354 (2B), +0.344 (4B), +0.469 (9B), +0.465
(27B), +0.268 (Llama), +0.166 (Gemma), +0.094 (Phi). On the full x101 cohort
the same base arms reproduce x101's end-position delta to the digit in seven
models and within 0.004 on Qwen3.5-27B (x105 keeps all 130 prompts where x101
reported 129; the cross-check is kept in the `_full` file). Across the 128
generated arms on the retained units the worst truncation rate after the
4,096-token retry is 0.022 (Qwen3.5-0.8B, whose responses are the longest),
the smallest median response length is 935 characters, and no arm has more
than 0.6 % of responses dominated by a repeated 4-gram.

Figure: `data/results/x105_cross_model_direction_causality.pdf` (per model the
suppression on the inserted texts against both random controls, the induction on
the original texts against its control, and the dose response).

## Reading

- The reference row reproduces the home model's behaviour through this
  pipeline (+0.272 / +0.338).
- **Effect size tracks the model in the same order as x101 and x103.** On the
  original texts the mid and large Qwen models move most (+0.25 to +0.39)
  and Qwen3.5-0.8B and the three off-family models least (+0.04 to +0.11);
  on the inserted texts the Qwen models ≥ 2B remove +0.22 to +0.28 and the
  others +0.09 to +0.11 — the ordering x103 found for the projection–refusal
  association and x101 and x114 for the behaviour itself.
- **The write undoes most of the word.** At −1.5 the residual rise over the
  original prompt is +0.007 (Phi), +0.009 (0.8B), +0.052 (Gemma), +0.072 (4B),
  +0.075 (2B), +0.159 (Llama), +0.216 (27B), +0.246 (9B) — on five of eight
  models more than two thirds of the inserted word's effect is gone, and on
  Qwen3.5-0.8B and Phi the dosed inserted text refuses within 0.01 of the
  original.
- **Both halves are the same axis.** The oriented dose slope is positive in
  every model with its interval excluding zero, and the half dose is always
  between zero and the full dose at the point estimates, so suppression on
  the inserted texts and induction on the clean ones are two signs of one
  quantity, as x86 found on the home model.
- **Neither cohort alone is doing the work.** **64-unit cohort** (the x101
  leak-free units alone;
  `data/results/x105_cross_model_direction_causality_x101only.json`, from
  `python3 code/x105_analyze.py --cohort x101_leak_free`): removal +0.088
  (Llama) to +0.273 (2B), creation +0.044 (Llama) to +0.386 (2B), target −
  norm-matched random +0.073 to +0.292, Holm p 7 × 10⁻⁴ except the creation
  test on Llama and Gemma (1.6 × 10⁻³), verdict `holds` in all eight. **Full
  cohort** (all 130 x101 prompts;
  `data/results/x105_cross_model_direction_causality_full.json`, from
  `--cohort full`): removal +0.087 (Phi) to +0.286 (2B), creation +0.049
  (Llama) to +0.357 (2B), target − norm-matched random +0.073 to +0.290,
  target − KL-matched random +0.065 to +0.281, share of the rise removed 0.41
  (Llama) to 1.00 (0.8B), monotonicity 0.988–1.000, every Holm p 7 × 10⁻⁴,
  verdict `holds` in all eight. The top-up moves every point estimate by at
  most 0.066 against the 64-unit values (largest: Gemma removal +0.180 →
  +0.114, 4B creation +0.380 → +0.338, 9B removal +0.188 → +0.223), in both
  directions, halves the interval widths, brings every transfer-model Holm p
  to 7 × 10⁻⁴, and changes no verdict.

## Artifacts

- Result: `data/results/x105_cross_model_direction_causality.json` (the 136
  leak-free units; the `cohort` block records both cohorts' counts and the
  sha256 of both exclusion artifacts), figure
  `x105_cross_model_direction_causality.{pdf,png}`; siblings
  `x105_cross_model_direction_causality_x101only.{json,pdf,png}` (64 units)
  and `x105_cross_model_direction_causality_full.{json,pdf,png}` (130 units).
- Exclusion sets: `data/runs/x105/x105_x85_exclusions.json` (`2305ef73…`): the
  47 x85 prompt ids and 17 x85 harm words inside the x101 cohort, the rule,
  the x85 cohort's sha256 and the 64 retained unit ids;
  `data/runs/x114/x114_x85_exclusions.json` (0 of 72 excluded).
- Frozen inputs: `data/runs/x105/x105_inputs.json` (`732788e5…`; per model the snapshot,
  the hook block and boundary ids, the direction artifact and its sha256, the
  direction norms, the control-direction artifact and its diagnostics, the
  rendered-chat sha256 of the 260 texts, the arm table and the dose units);
  `data/runs/x114/x114_inputs.json` (the same for the 144 x114 texts, plus
  the sha256 of the reused control directions and KL calibration).
- Per model: `data/runs/x105/<tag>/x105_control_directions.npz`,
  `x105_kl_calibration.json`, `x105_generations_<arm>.json` (with its
  write-site audit) and `x105_refusal_<arm>.jsonl`;
  `data/runs/x114/<tag>/x114_generations_<arm>.json` and
  `x114_refusal_<arm>.jsonl`. The generation files (≈ 300 MB for x105,
  ≈ 170 MB for x114) are shipped in this repository.
- Code: `code/x105_common.py`, `x105_prepare.py`, `x105_calibrate_kl.py`,
  `x105_generate.py`, `x105_judge_refusal.py`, `x105_exclusions.py`,
  `x105_analyze.py` (`--cohort {extended,x101_leak_free,full,all}`),
  `x105.sbatch`; the x114 stages `code/x114_prepare.py`, `x114_dosed.py`,
  `x114_judge_refusal.py`, `x114.sbatch` and the mode table
  `code/x114_cohorts.py`; tests `code/tests/test_x105_direction_causality.py`
  (7) and `test_x114_cohort_topup.py` (9), in the 127-test suite. The hook,
  directions, model registry and chat rendering are imported from x88 / x87 /
  x103, the cohorts, renderer check, sampling engine, seeds and judge from
  x101 / x114 / x83 / x81, and the control construction and KL-matching rule
  from x86.
