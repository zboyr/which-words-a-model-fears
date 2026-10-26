# x86: subtracting the x85 response-boundary shift lowers Qwen3.5-4B refusal and adding it raises refusal, dose-ordered and specific against KL-matched controls

Parent: [Experiment index](README.md).
Source evidence: [x85 crossed keyword-prompt latent decomposition](x85-crossed-keyword-prompt-latent-decomposition.md),
which found the shared boundary shift and its refusal association.

x85 showed that appending a harm word produces a neutral-corrected shift at
Qwen3.5-4B's response-boundary tokens (`</think>` and the following `"\n\n"`,
state 30) whose held-out projection tracks the induced refusal change. x86
tests whether that shift is **causally usable**: on prompt-word cells excluded
from a fold's direction, subtract the fold-specific shift from harm-word (H)
prompts at state 30 and add it to matched neutral-word (N) prompts, then
regenerate and re-judge. At the benign-KL-admitted dose `a* = 0.5`:

- **subtracting from H lowers refusal by 0.080** (95% CI [0.043, 0.121]),
- **adding to N raises refusal by 0.086** ([0.041, 0.133]),
- the bidirectional effect is **dose-ordered** (half 0.036 → full 0.083) with a
  positive fixed-effect slope [0.053, 0.115],
- both one-sided Holm-adjusted p = 2×10⁻⁴,
- and it **exceeds both KL-matched null directions**: true − random
  = 0.088 [0.054, 0.124], true − permutation = 0.055 [0.017, 0.095].

**All four gates pass (G0 integrity/calibration, G1 bidirectional, G2
specificity, G3 output integrity): the fold-specific x85 boundary shift is a
causally usable, direction-specific refusal mediator within this activation
intervention.** The effect is carried almost entirely by the final `"\n\n"`
token (tbg-only 0.070 vs tbg_minus1-only 0.011, no synergy), exactly matching
x85's single-position KL ranking. The intervention writes activations, not
weights. The secondary substantive-compliance (`comply2`) outcome moves
**oppositely and commensurately**: subtraction raises harmful-request
compliance +0.072 [+0.032, +0.115], induction lowers neutral-prompt
compliance −0.082 [−0.129, −0.040], both inert on the same KL-matched nulls
and carried by the same single token — so the refusal movement is
substantive-compliance movement, not explicit-refusal wording alone.
Nothing is routed to [../headline-numbers.md](../headline-numbers.md).

## Design, as run

- **Frozen inputs.** The x85 `Qwen/Qwen3.5-4B` snapshot (`851bf6e8…`),
  non-thinking template, and the 24 x85 feature shards. Stage 0 verified
  4,704-cell coverage, cell/position/shard hashes, state count, and hidden
  dimension against the x85 cohort.
- **Fold directions.** State 30 (output of decoder block 29; the hook is on
  block 29's output). For each of the four x85 folds and each boundary
  position, `mu[f,t] = mean(h_H − h_N)` over the 72 prompts × 18 words with
  `prompt_fold ≠ f and word_fold ≠ f` (1,296 paired cells), rebuilt in fp32
  from the fp16 shards. The raw mean vector (not just its unit) is frozen:
  dose 1 = one observed fold-mean shift in residual units. Norms 5.9–7.9;
  each true fold direction has cosine 0.998 to the x85 all-data global unit
  (lineage check only). The four fold directions are near-identical
  (cross-fold cosine 0.99).
- **Target graph.** Strict fold diagonal `prompt_fold[p] == word_fold[w] == f`.
  Within each fold and historical-refusal stratum, a `rng(8604)` simple
  (3,4)-biregular bipartite graph on 8 prompts × 6 words. Total **288 cells**;
  every prompt at degree 3, every harm word on 12 prompts (4 per stratum).
  Frozen before calibration; all H and matched N are regenerated inside x86
  (x85 texts/labels are not reused as dose zero).
- **Activation contract.** A prefill-only hook writes `signed_dose · mu[f,t]`
  at the two audited absolute positions and nothing on cached decode steps,
  the word token, `<|im_end|>`, weights, or any other layer. H uses negative
  dose, N positive. A **dose-zero equivalence audit** on 16 held-out benign
  prompts passed exactly: hooked-vs-unhooked logits bitwise equal, greedy and
  seeded-sampled continuations token-identical, exactly one prefill hook call,
  zero writes and zero decode modifications. (The Qwen forward is bitwise
  deterministic run-to-run on the RTX PRO 6000, so bitwise equality is
  meaningful.)
- **KL calibration.** `KL(base‖intervened)` at TBG over 256 `rng(8600)` benign
  calibration prompts (x66 seed-42 train benign portion). The `both`
  intervention's mean KL is 0.003 / 0.008 / 0.031 / 0.128 at magnitudes
  0.125 / 0.25 / 0.5 / 1.0; **`a* = 0.5`** is the largest magnitude with
  pooled and per-fold mean KL ≤ 0.10 at both signs and monotone KL. Behavior
  doses frozen as {0, 0.25, 0.5}. Single-position KL at a*: `tbg` 0.021–0.022
  ≫ `tbg_minus1` 0.0015 — the final-token write dominates the distribution
  shift. Both KL-matched nulls calibrated (`random` needs multiplier ≈ 2.1–2.3,
  `permutation` ≈ 0.84 to reach the same benign KL).
- **Conditions.** 14 per target cell (H/N × {dose 0, both ±a*/2, both ±a*,
  tbg_minus1-only ±a*, tbg-only ±a*, random-null KL-matched, permutation-null
  KL-matched}), 10 responses each = 40,320 target responses; plus 256 sealed
  benign prompts × 5 conditions × 5 responses = 6,400. Exact x85 HF sampling
  contract (T = 1.0, top-p 0.9, top-k 50, 2,048 tokens, one 4,096 retry),
  common-random-number seed per (scope, cell, arm) across doses/families.
- **Judging.** Primary `refusal2`, the exact x85/x64 contract
  (`deepseek/deepseek-v4-flash-0731`), on all 46,720 responses:
  4,032/4,032 target and 1,280/1,280 benign rows judged, 0 failures after
  convergence. Secondary `comply2` (target-only), same judge model but
  **one response per call** (`--per-response`): all 14 × 288 rows converged
  in two passes, 0 residual failures (deviations).

## Results

n = 288 held-out cells, 10,000-replicate two-way (prompt × word) cluster
bootstrap resampling the ten Bernoulli responses within each cell. Raw
condition mean refusal: H_0 0.679, N_0 0.352 (natural gap 0.327).

### G1 — bidirectional causal movement (pass)

| estimand | half dose (a*/2) | full dose (a*) |
|---|---|---|
| `suppress_H = r(H_0) − r(H_both_−a)` | +0.037 [+0.004, +0.072] | **+0.080 [+0.043, +0.121]** |
| `induce_N = r(N_both_+a) − r(N_0)` | +0.036 [−0.002, +0.076] | **+0.086 [+0.041, +0.133]** |
| `bidirectional` | +0.036 [+0.011, +0.064] | +0.083 [+0.053, +0.115] |

Both full-dose one-sided Holm-adjusted p = 2×10⁻⁴; `bidirectional(a*) >
bidirectional(a*/2) > 0`; oriented fixed-effect dose slope +0.083
[+0.053, +0.115]. Raw means: H_0 0.679 → H_both_full 0.599; N_0 0.352 →
N_both_full 0.438. The natural H−N gap closes by **51%** (gap_0 0.327 →
gap_full 0.161; closure 0.508 [0.323, 0.746]).

### G2 — direction specificity (pass)

Table: [G2 direction specificity: bidirectional refusal effect at a*, true direction versus KL-matched nulls](json/x86-crossed-keyword-boundary-intervention__g2-specificity.json) — the true `both` write gives +0.083 against −0.005 (random) and +0.028 (permutation); true − random +0.088 [+0.054, +0.124] and true − permutation +0.055 [+0.017, +0.095], both Holm-significant.

The orthogonal-random null is inert; the paired-sign-permutation null is weak
and its own interval includes zero. The true direction beats both at
KL-matched dose, so the effect is specific to the x85 geometry, not to a
generic norm-matched activation perturbation, and is not explained by norm
or by the harm-word pairing structure alone.

### G3 — output integrity (pass)

Every target condition passes: 0.08% of 46,720 responses truncated after the
4,096 retry, median printable fraction 1.00, median length ≥ 945 chars, no
condition with >5% repeated-4-gram-dominated responses. The behavioral
movement is not degeneration.

### Position attribution (secondary)

Bidirectional(a*) by write site: **tbg (`"\n\n"`) only +0.070 [+0.040,
+0.103]**, tbg_minus1 (`</think>`) only +0.011 [−0.016, +0.038], synergy
(`both − minus1 − tbg`) +0.002 [−0.041, +0.045]. The final-token residual
carries almost the entire causal effect and the two writes are additive —
matching x85's KL ranking and the x85
observation that the boundary decision variable is read at the generation
handoff. This localizes the readout to the exact generation-handoff position
and rules out a distributed two-token mechanism for this direction. Raw:
H_tbg_full 0.616 / H_minus1_full 0.665; N_tbg_full 0.430 /
N_minus1_full 0.360.

### Substantive compliance (`comply2`, secondary)

Same 288 cells and bootstrap on the `comply2` label (did the response
substantively attempt to deliver what was asked), judged per-response by the
same `deepseek-v4-flash` judge. Estimands use the refusal-table formulas
applied to the comply2 label, so a **negative** sign means compliance moves
opposite to refusal. Raw means: H_0 0.356 → H_both_full 0.428; N_0 0.666 →
N_both_full 0.584 (natural N−H comply gap 0.310 → 0.156 at full dose).

Table: [Substantive compliance (`comply2`) estimands by dose, with KL-matched nulls](json/x86-crossed-keyword-boundary-intervention__comply2-estimands.json) — at full dose `suppress_H` −0.072 [−0.115, −0.032] and `induce_N` −0.082 [−0.129, −0.040], while the random (−0.001) and permutation (−0.026) nulls span zero.

Compliance mirrors refusal in every respect: opposite sign, commensurate
magnitude, dose-ordered, inert on both KL-matched nulls, and carried by the
same single token (tbg-only −0.072 [−0.105, −0.040], tbg_minus1-only −0.010
[−0.037, +0.018], synergy ≈ 0).

### Benign over-refusal (cost)

On 256 sealed benign prompts (x66 seed-42 test benign portion), baseline
refusal B_0 = 0.467 (the confusable-benign pool is deliberately borderline).
Adding the true direction raises benign refusal by **+0.058 [+0.031, +0.085]**
at full dose (+0.029 at half); the random and permutation nulls do not
(−0.012 and +0.014, both CIs spanning zero). The N-induction effect thus
generalizes partly to benign inputs.

## Deviations (recorded 2026-08-24)

- **Secondary `comply2` was judged per-response, not 10-per-call (completed
  2026-08-24).** The frozen contract judges 10 responses in one call;
  `deepseek-v4-flash` failed on 40–50% of `comply2` rows for the
  long-response conditions (failed rows averaged 38k chars vs 28k parsed —
  malformed schemas and multi-thousand-token rambles). An earlier attempt
  with a stronger judge (`openai/gpt-5.6-sol`, clean but ~$4–6/condition)
  exhausted the OpenRouter credit after three conditions (HTTP 402); **those
  gpt judgments were discarded** — `comply2` ran fresh under one model, one
  method. The rerun judges **one response per API call** (~3–4k-char input)
  and assembles the length-10 array: all 14 target conditions × 288 rows on
  `deepseek-v4-flash`, ~$4 total.
- **Two label-shape repairs were added mid-judging** (
  with 13 conditions in flight): a bare bool under the correct field for
  n = 1, and 0/1-integer lists coerced to bools. Both are strict acceptance
  extensions — they admit previously rejected but semantically unambiguous
  judge outputs and change nothing previously accepted — so the final
  dataset is consistent under the final (extended) acceptance rule. Without
  them the dominant flash failure mode (bare `{"complied": true}` instead of
  a length-1 list; 59% of first-pass N_tbg_full rows) was content-stable and
  would not have converged under retries. First pass: 740/4,032 rows failed;
  one resume pass under the extended rule re-judged all 740 with **0
  residual failures**. Resume appends the successful record and keeps the
  superseded failed record; the analyzer skips failed records and enforces
  exactly one successful record per id.
- **`code_drift` allowed for `x86_judge.py` and `x86_analyze.py` only.** Those
  two downstream scripts changed after the manifest freeze (a `--judge-model`
  override and the comply2-optional guard). `x86_analyze.py` records the drift
  and refuses to run if any *frozen-pipeline* file (`x86_common`,
  `x86_build_manifest`, `x86_calibrate_kl`, `x86_generate`) drifts; those are
  byte-identical to the manifest.

## Artifacts

- Result: `data/results/x86_boundary_intervention.json` (sha256 `1bcd8b4e…`;
  `status: complete`, gates, refusal estimands, specificity, position
  attribution, G3 integrity, benign cost, `compliance_secondary` comply2
  estimands, `code_drift`).
- Frozen design: `data/runs/x86/x86_manifest.json` (`cbc7c9b3…`),
  `x86_directions.npz` (`698b3ce1…`; true/random/permutation fold directions,
  fold assignments), `x86_kl_calibration.json` (`a6c5a097…`, `a* = 0.5`).
- Judgments: `data/runs/x86/x86_refusal2_{condition}.jsonl` (all 19
  complete); `x86_comply2_{condition}.jsonl` (14 target conditions, 288
  successful rows each; append-style resume, failed records retained but
  skipped by the analyzer).
- Generations `data/runs/x86/x86_generations_{condition}.json` (19 files,
  ~140 MB; regenerable from the manifest, calibration, and directions; kept on
  the cluster under `<workdir>/x86/`, gitignored locally).
- Code: `code/x86_common.py`, `x86_build_manifest.py`, `x86_calibrate_kl.py`,
  `x86_generate.py`, `x86_judge.py`, `x86_analyze.py`, `x86.sbatch`; tests
  `code/tests/test_x86_boundary_intervention.py` (9).
