# x101: one inserted harm keyword raises refusal in all eight transfer models, by +0.04 to +0.49

Parent: [Experiment index](README.md). Cross-model extension of
[x83](x83-external-keyword-insertion.md), whose single-model result it
reproduces on Qwen3.5-4B and carries to the other seven models of the
[eight-model transfer](x88-eight-model-deep-dose-transfer.md).

One shared set of 130 harmful prompts, each with the word x83 assigned it
inserted at the start (S), middle (M) or end (E), regenerated per model with
x83's sampling contract and judged with the x64 `refusal2` contract. On the
identical 130 prompts, the inserted word raises `refusal2` relative to the
contemporaneously regenerated original in **every one of the eight models at
the end position** (+0.10 on Qwen3.5-0.8B to +0.49 on Qwen3.5-27B, all
Holm-adjusted p ≤ 0.003 over the 24 model × position tests) and in seven of
eight at the start and middle; the exception is Qwen3.5-0.8B, where S and M
are null (+0.03, +0.00). Within the Qwen family the effect grows with size
from 0.8B to 2B and is then flat from 2B to 27B; the three off-family models
(Llama-3.2-3B, gemma-4-e2b, Phi-4-mini) start from a near-floor baseline
(O = 0.03–0.04) and gain +0.11 to +0.29 at the end. Edges beat the middle in
every model that can tell them apart. The recoverability gate admits almost
nothing below 9B at S and M, so the gated per-model table is sparse there and
the ungated table is the like-for-like comparison. [x114](x114-x85-free-insertion-topup.md)
adds 72 further units of this design on prompts and words outside the x85
direction-fitting cohort, run in all eight models, so that
[x103](x103-cross-model-direction-insertion-projection.md) and
[x105](x105-cross-model-direction-causal-specificity.md) rest on 136 leak-free
units. Nothing here is routed to
[../headline-numbers.md](../headline-numbers.md).

## Design, as run

- **Cohort (shared by all eight models).** The 321 x83 prompts restricted to
  those with historical `refusal2` rate ≤ 0.8 in every model's
  `data/labels/{tag}_labels_v4.json` (eligible per model: qwen 321, qwen08b
  319, qwen2b 273, qwen9b 231, qwen27b 188, llama 278, gemma 293, phi 309;
  intersection **130**). Each prompt keeps the word
  x83 assigned it and x83's exact O/S/M/E strings; the builder re-derives
  every insertion from `x83_common.insert_word` and fails on any drift.
  75 distinct words, 13 categories (largest: `fraud_property_crime` 24,
  `cybercrime` 20, `illegality_markers` 14, `chemical_biological` 13).
  Selection reads historical labels only.
- **Models.** Seven run fresh from the x88 frozen snapshots (`qwen08b`
  0.8B, `qwen2b` 2B, `qwen9b` 9B, `qwen27b` 27B, `llama`
  Llama-3.2-3B-Instruct, `gemma` gemma-4-e2b-it, `phi` Phi-4-mini-instruct
  on the native implementation); **Qwen3.5-4B (`qwen`) is not rerun** — its
  rows are the x83 detection, generations and judgments restricted to the
  130 prompts (x83 ran under the same contract on the 321-prompt superset).
  Deviation in the compute setup: the x88 venv and snapshots had been swept
  from the cluster scratch, so the frozen revisions were re-downloaded and the jobs
  ran in the x100 venv ([../infrastructure.md](../infrastructure.md)).
- **Chat rendering.** x88's `format_user_chat` (user turn only,
  `add_generation_prompt`, `enable_thinking=False`). A per-model preflight
  with one real model load confirmed on four prompts that the scaffold
  around the user turn is a template constant and recorded its sha256 in
  `x101_manifest.json`: the Qwen3.5 templates carry an *empty*
  `<think>\n\n</think>` block (identical to the campaign renderer that
  produced the historical labels); Llama, gemma and Phi have no thinking
  block. The Llama scaffold is the only one that differs from the campaign
  renderer, and only in the template's auto-inserted "Today Date" line
  (not pinned, as in x88; the rendered-chat hash of every row is recorded).
- **Detection gate.** x83's template, surface variants, statistic and
  threshold (`P_detect > 0.90`) run on each model with its own template.
  The preflight greedy-decoded the detection prompt for four distinct-word
  probes at S and E; the longest common token prefix (the template's fixed
  pre-answer emission) was **empty for every model**, so the gate is scored
  at the first generated token everywhere (`scoring_offset` 0 in the
  manifest). Admitted per model (S / M / E; all three): qwen 21 / 43 / 111
  (13), qwen08b 0 / 0 / 11 (0), qwen2b 0 / 0 / 6 (0), qwen9b 23 / 83 / 125
  (18), qwen27b 117 / 105 / 127 (95), llama 0 / 2 / 57 (0), gemma 39 / 95 /
  125 (30), phi 0 / 3 / 100 (0). The greedy answer names the word at E for
  73–100 % of prompts in every model, but at S the small models answer with
  the prompt's own first word (`the`, `a`, `The device.`) and at M they
  spread mass (median `P_detect` at S: 0.00 for 0.8B / 2B / llama / phi,
  0.17 gemma, 0.36 9B, 0.55 4B, 0.99 27B).
- **Generation.** x83 contract (T = 1.0, top-p 0.9, top-k 50, 10 sequences,
  2,048 tokens, one 4,096 retry), seeds per (tag, prompt, condition) in an
  `x101` namespace, base 101. 130 rows per arm per model, 36,400 responses in
  total counting the imported qwen rows. Retry rows were rare except on
  0.8B (48–55 per arm, 13–22 responses still truncated after retry; its
  responses average 1,100 tokens) and 2B (4–8 per arm, ≤ 1 truncated);
  every other model had ≤ 2 retries per arm and nothing truncated.
- **Judge.** Exact x64 `refusal2` contract as x83 ran it
  (`deepseek/deepseek-v4-flash-0731`, reasoning low, seeds 42/1042/2042),
  run locally. 3,639 of 3,640 fresh prompt-conditions judged (four rows
  needed a re-run); **one row, qwen27b/M prompt 21446, returned malformed
  JSON on all 36 attempts** (the ten responses are all explicit refusals).
  The analysis drops that prompt from qwen27b in all four conditions so
  every contrast stays paired (n = 129 for qwen27b); its other labels are
  O 8/10, S 9/10, E 8/10. Total judge spend ≈ $1.9.
- **Analysis** (`code/x101_analyze.py`; x83's functions imported). Per
  model: `delta_pos = mean_i(r_pos,i − r_O,i)` on that model's admitted
  prompts, 10,000-replicate two-stage paired bootstrap, Holm over the three
  positions; ungated (all 130), common-admitted, non-admitted, floor
  prompts, per-category (≥ 20 admitted), Spearman moderators and the
  run-era diagnostic against that model's own historical labels. Across
  models: Holm over all 18 gated tests with n > 0 and over all 24 ungated
  tests.

## Results

Table: [Condition means over the 130 prompts](json/x101-cross-model-keyword-insertion__condition-means.json) — mean `refusal2` per model in O, S, M and E, regenerated O with historical O in parentheses; E is the highest condition in every model except 4B (S 0.589, E 0.584), peaking at 0.707 on qwen27b.

The run-era diagnostic (regenerated O − historical O on the same prompts)
is within ±0.02 for every model, with every 95 % interval covering 0.

Table: [Gated deltas on each model's admitted prompts](json/x101-cross-model-keyword-insertion__gated-deltas.json) — 24 model × position rows with n admitted, mean rates, delta, CI and Holm p within model and over the 18 gated tests; every populated cell is positive and significant except qwen08b E (+0.127, n = 11), llama M (n = 2) and phi M (n = 3), with gemma S the weakest significant cell (Holm p (18) 0.042).

Table: [Ungated deltas on the identical 130 prompts in every model](json/x101-cross-model-keyword-insertion__ungated-deltas.json) — S, M and E deltas with CIs and Holm p over 24 tests; E runs from +0.099 (qwen08b) to +0.489 (qwen27b), and only qwen08b's S and M are null.

Figure: `data/results/x101_cross_model_insertion.pdf` (gated and ungated
deltas per model per position with 95 % intervals, in the style of the
paper's transfer figure).

Effect size scales with the
model, not monotonically with the baseline: the off-family models and the
0.8B model start from O ≈ 0.03–0.10, so their absolute deltas (+0.10 to
+0.29 at E) are smaller than the Qwen ≥ 2B models' (+0.37 to +0.49) despite
more head-room; in relative terms llama's E rate rises tenfold. Within Qwen
the jump is from 0.8B (S and M null, E +0.10) to 2B (+0.21 / +0.12 / +0.38),
after which 2B–27B are flat at S and M and rise mildly at E (+0.38 → +0.49).
Being bigger does not make the model less sensitive to the word; the largest
model has the largest end effect. At S and M the recoverability gate admits
0–3 prompts for 0.8B, 2B, llama and phi; where both sets exist
the admitted and non-admitted deltas agree, as in x83, and `P_detect` is at
most weakly correlated with the per-prompt delta, so the ungated table is
the comparison to read.

**Secondary.**

- *Common cohort* (admitted at all three positions; only four models have
  one): qwen n = 13: S +0.49, M +0.34, E +0.51; qwen9b n = 18: +0.48 /
  +0.28 / +0.64; qwen27b n = 95: +0.36 / +0.22 / +0.49; gemma n = 30:
  +0.12 / +0.13 / +0.21. Position differences on the same prompts:
  M − S = −0.15 [−0.31, −0.02] (4B), −0.21 [−0.36, −0.04] (9B), −0.14
  [−0.21, −0.07] (27B); E − M = +0.17 [+0.05, +0.30], +0.37 [+0.18, +0.55],
  +0.27 [+0.19, +0.36]; E − S = +0.02 [−0.08, +0.12] (4B), +0.16 [0.00,
  +0.33] (9B), +0.13 [+0.08, +0.19] (27B). On gemma the three positions are
  within noise of each other (M − S = +0.02, E − M = +0.08 [−0.01, +0.18]).
  In the ungated table the middle is the weakest position in all eight
  models and the end the strongest in seven (4B: S ≈ E).
- *Non-admitted prompts* carry the same effect as admitted ones wherever
  both sets are populated: 4B S +0.35 (n = 109) vs +0.48 admitted, M +0.20
  (87) vs +0.24; 9B S +0.31 (107) vs +0.49, M +0.13 (47) vs +0.25; 27B M
  +0.18 (24) vs +0.22; gemma S +0.08 (91) vs +0.12. Below 9B at S and M the
  "non-admitted" set *is* the cohort.
- *Floor prompts* (regenerated O = 0/10; 55–78 per Qwen model, 105–118 for
  the off-family models): the inserted word pushes to r ≥ 0.5 at S / M / E
  26 / 13 / 22 of 57 (4B), 1 / 0 / 7 of 74 (0.8B), 10 / 7 / 24 of 55 (2B),
  25 / 15 / 37 of 78 (9B), 29 / 10 / 40 of 72 (27B), 9 / 6 / 34 of 117
  (llama), 14 / 8 / 22 of 118 (gemma), 6 / 3 / 9 of 105 (phi).
- *Moderators.* `P_detect` vs per-prompt delta (Spearman, all 130): |ρ| ≤
  0.19 everywhere except 9B at E (ρ = +0.29, p = 0.001) and 4B at E
  (+0.22, p = 0.013); 24 tests. Historical `r_O` vs delta on
  admitted prompts: between −0.37 and +0.38, no consistent sign.
- *Per category* (≥ 20 admitted): `fraud_property_crime` at E is positive
  on 4B (+0.31, n = 21), 9B (+0.50, 22) and 27B (+0.43, 24; also at S
  +0.34, 22), `cybercrime` at E on 27B (+0.40, 20); on gemma at E both
  categories are null (+0.02, −0.02), i.e. gemma's +0.18 at E is carried by
  the other eleven categories.

## Artifacts

- Result: `data/results/x101_cross_model_insertion.json` (sha256
  `4eab58b8…`), figure `x101_cross_model_insertion.{pdf,png}`.
- Frozen cohort: `data/runs/x101/x101_cohort.json` (`118a27c6…`,
  `cohort_sha256` `8ef7b8a2…`); manifest `x101_manifest.json`
  (`6cf6d551…`; per-model snapshot, revision, scaffold text and sha256,
  gate offset, environment); per-model preflights
  `data/runs/x101/<tag>/x101_preflight.json`, detection `x101_detect.json`,
  generations `x101_generations_{O,S,M,E}.json`, judgments
  `x101_refusal_{O,S,M,E}.jsonl` (the generation files total ≈ 140 MB).
- Code: `code/x101_common.py`, `x101_build_cohort.py`, `x101_preflight.py`,
  `x101_detect.py`, `x101_generate.py`, `x101_judge_refusal.py`,
  `x101_analyze.py`, `x101.sbatch`; tests
  `code/tests/test_x101_cross_model_insertion.py` (8, in the 66-test
  suite). Insertion rules, gate scoring, sampling engine, judge and
  bootstrap are imported from x83 / x81, the model registry and chat
  rendering from x88.
