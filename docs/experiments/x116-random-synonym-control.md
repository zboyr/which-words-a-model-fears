# x116: with x113's nominee fixed, a synonym drawn at random from the same admissible candidates removes 0.67–0.89 of what the table-lowest synonym removes; the table's choice adds +0.017 to +0.078 of `refusal2`, significant in three of the seven transfer models

Parent: [Experiment index](README.md). A control arm for
[x113](x113-cross-model-table-substitution.md). x113 replaces each prompt's
table-nominated word by the accepted drafter candidate that the model's table
places **lowest** (S_T), yet within a model the table's shift numbers do not
order the per-prompt effect (Spearman −0.09 to +0.16). x116 asks how much of
x113's effect comes from choosing the table-lowest synonym rather than any
harm-free synonym of the same word.

On the same x85-free Hard-1K units, with the same nominee, x113's drafter
candidates and x113's splice, **R_rand** (one accepted candidate other than
S_T's, drawn uniformly at random) removes **+0.243 (Qwen3.5-4B), +0.120
(0.8B), +0.199 (2B), +0.208 (9B), +0.216 (27B), +0.035 (Llama), +0.187
(Gemma), +0.170 (Phi)** of `refusal2`, Holm p ≤ 2 × 10⁻³ over the seven
transfer models. S_T removes +0.052 to +0.283 on the same units, so R_rand
reaches 0.67 (Llama) to 0.89 (9B) of S_T's effect. The paired **r(R_rand) −
r(S_T) is +0.017 to +0.078, positive in all eight models** (mean +0.041). Its
interval excludes zero in Qwen3.5-2B (+0.047), 27B (+0.067), Gemma (+0.078)
and Phi (+0.036); after Holm over the seven transfer models it stays
significant in 2B, 27B and Gemma (one-sided p 0.042, 0.002, 7 × 10⁻⁴; Phi
0.061). Most of the effect therefore comes from replacing the nominated word
at all, and the table-lowest choice adds a small, consistent increment. Per
unit, the size of the table-shift gap between the two candidates barely
predicts that increment (Spearman −0.05 to +0.21; the interval excludes zero
only in Gemma). The boundary projection shows the same ordering: R_rand
lowers it by 0.40–0.74 sd against S_T's 0.61–0.87, so R_rand ends 0.12–0.21 sd
above S_T in every model, and that projection gap tracks the per-unit refusal
gap at Spearman +0.26 to +0.62.

## Question

Given the nominee, does it matter which harm-free synonym replaces it? If
the table's lowest pick matters, R_rand should remove clearly less refusal
than S_T. If any admissible synonym does as well, r(R_rand) − r(S_T) should be
zero.

## Design, as run

- **Units, nominees, candidates.** x113's 397 units per model, the nominee
  from x113's `nominations.json`, and the candidates already stored in x113's
  `texts.json` (no redrafting: x113's drafter output, DeepSeek-V4-Flash at
  temperature 0, about 9.5 candidates per unit, 7.1–7.4 accepted by x113's
  structural checks). x113's S_T text was verified to be the splice of its
  chosen candidate on every unit of every model.
- **R_rand** (`x116_build_texts.py`). The pool is the unit's accepted
  candidates (the same hedge-phrase or content span x113 used), minus the
  table-lowest one S_T used, minus a candidate whose spliced text equals S_T
  (2 units in all), minus a candidate carrying one of x85's 48 words (1–9 per
  model, so that x113's frozen x85 exclusion stays the complete unit filter).
  One candidate is drawn uniformly with `numpy.random.default_rng(seed)` and
  seed = the first 8 bytes of sha256(`x116|116|<tag>|<unit id>`), so each
  model and unit has its own fixed seed that does not depend on order. Mean pool
  size is 6.1–6.4. The text is x113's `splice` (with the hedge-separator fix).
  56–82 R_rand texts per model are hedge-phrase substitutions. The candidate's
  table shift is x113's record. The shift gap shift(R_rand) − shift(S_T) is
  ≥ 0 by construction and defined on 340–354 units per model; it averages
  +0.26 to +0.46 sd.
- **Arm counts.** All 397 units: R_rand exists for 378 / 367 / 375 / 369 / 370 /
  372 / 372 / 367 (Qwen3.5-4B, 0.8B, 2B, 9B, 27B, Llama, Gemma, Phi). 16–25
  units have no S_T (x113), and 2–8 more have S_T but no other admissible
  candidate. x85-free: 329–338 of the 335–341 units with S_T have an R_rand
  (2–6 do not). The Hard-1K analysis has 241–249 units per model and XSTest
  88–90.
- **Generation** (`x116_generate.py`, `x116.sbatch genloop`). This is x113's
  contract: x106 rendering, x101 sampling (T = 1.0, top-p 0.9, top-k 50, ten
  samples, 2,048 tokens with one 4,096 retry), seed x106 `stable_seed(tag,
  prompt id, "R_rand")`. It ran as work-stealing workers over eight resumable
  shards per model in the x113 cluster tree. There were 2,970 cells (29,700
  responses), 0–17 retry rows per model, and truncation after the retry of
  at most 0.0025.
- **Judge** (`x116_judge.py`). x113's `refusal2` (`x81_judge_refusal.judge_row`,
  ten responses per call) with the OpenRouter upstream pinned to DeepInfra
  with no fallback (`x114_judge_refusal.pin_provider`, recorded as
  `judge_provider`). 23 rows failed on the first pass, all recovered by one
  resume pass with shifted attempt seeds, and none needed the per-response
  fallback. O is x106's `O_base` rows and S_T is x113's rows, both reused.
- **Representation** (`x116_project.py`). O and R_rand were read at x106's
  site with x113's reader on the model's own direction, in the model's x106
  Hard-1K sd unit. The site audit passed on every model. The re-read O matches
  x113's O projection to 0 sd on seven models and 0.09 sd on Qwen3.5-9B. The
  R_rand change is taken against x113's O.
- **Statistics** (`x116_analyze.py`). This is x113's prompt-paired two-stage
  bootstrap (prompts within stratum, then the ten binary responses of each
  arm) with 10,000 replicates and seed 11,600, on the x85-free units with both
  arms (`--full`: every unit). Holm runs over the seven transfer models per
  contrast, one-sided for Δ_S > 0, Δ_rand > 0 and r(R_rand) − r(S_T) > 0, and
  two-sided for r(R_rand) − r(S_T). Qwen3.5-4B is the reference row. The
  per-unit Spearman correlations use 2,000 unit-bootstrap replicates.

## Results

### Hard-1K, x85-free

Table: [Hard-1K: S_T against R_rand](json/x116-random-synonym-control__hard1k.json) — Δ_S +0.052 to +0.283 and Δ_rand +0.035 to +0.243 on the same 241–249 units per model (every Δ_rand Holm p ≤ 2e−3); Δ_rand / Δ_S 0.67 (Llama) to 0.89 (Qwen3.5-9B); r(R_rand) − r(S_T) +0.017 (Llama) to +0.078 (Gemma), Holm one-sided 0.042 (2B), 0.002 (27B), 7e−4 (Gemma), 0.061 (Phi), 0.157 (0.8B, 9B, Llama); Qwen3.5-4B +0.033 [−0.004, +0.072], one-sided p 0.042 unadjusted.

| | Δ_S | Δ_rand | r(R_rand) − r(S_T) |
|---|---|---|---|
| mean over the eight models | +0.213 | +0.172 | +0.041 |

Per unit, R_rand refuses more than S_T on 0.26–0.45 of units and less on
0.11–0.26 (Llama 0.06 / 0.06, where most units are at zero in both arms).
The table-shift gap between the two picked candidates and the per-unit
refusal gap correlate at Spearman −0.05 (Llama) to +0.21 (Gemma, [+0.07,
+0.33]). The interval excludes zero only in Gemma, and the correlation is no
stronger on the units with r(O) ≥ 0.5 (−0.04 to +0.26). The table's choice
helps on average, but how far the table separates the two candidates does not
say by how much on a given prompt, which matches x113's finding that the
table's numbers do not rank per-prompt effects.

### XSTest, x85-free

Table: [XSTest: S_T against R_rand](json/x116-random-synonym-control__xstest.json) — at base rates 0.006–0.200, Δ_rand (+0.005 to +0.105) is as large as Δ_S (+0.006 to +0.135); r(R_rand) − r(S_T) is −0.006 to +0.031 and no Holm-adjusted p is below 0.05.

### Representation, Hard-1K

Table: [Projection of R_rand against S_T](json/x116-random-synonym-control__projection.json) — R_rand lowers the model's own boundary projection by 0.40 (Gemma) to 0.74 (Qwen3.5-4B) sd against S_T's 0.61–0.87, so R_rand ends +0.12 to +0.21 sd above S_T with every interval excluding zero; the per-unit projection gap tracks the refusal gap (Spearman +0.26 Llama to +0.62 Qwen3.5-2B), while the table-shift gap predicts the projection gap only weakly (+0.03 to +0.25).

### Whole population (`--full`)

Table: [Hard-1K, no x85 exclusion](json/x116-random-synonym-control__hard1k-full.json) — 272–281 units per model; r(R_rand) − r(S_T) +0.017 to +0.068, Holm one-sided significant in Qwen3.5-2B (+0.058), 27B (+0.061) and Gemma (+0.068); Δ_S moves by at most 0.017 against the x85-free set.

## Reading

- **The nominee carries most of the effect.** With the word fixed, replacing it
  by any admissible harm-free synonym, chosen without the table, removes
  0.67–0.89 of the refusal that the table-lowest synonym removes, and Δ_rand
  is Holm-significant in all seven transfer models. x113's effect therefore
  comes mostly from which word is replaced, not which synonym replaces it.
- **The table's synonym choice is a small, consistent addition.** It adds
  +0.017 to +0.078 of `refusal2` and points the same way in all eight models,
  with intervals excluding zero in four and Holm-significant in three
  (Qwen3.5-2B, 27B, Gemma). The projection agrees: in every model the
  table-lowest synonym lowers the boundary projection 0.12–0.21 sd more than
  a random one.
- **The table orders candidates on average, not unit by unit.** The per-unit
  shift gap barely predicts the per-unit refusal gap, as in x113, where the
  shift numbers did not rank per-prompt effects. A useful per-unit signal is
  the projection of the actual substituted text, not the context-free table
  entry of the synonym.

## Deviations from the plan

- The random pool excludes a candidate that carries an x85 word (1–9 per
  model), in addition to the S_T candidate itself. Otherwise new units would
  have had to leave the x85-free set. It also excludes the 2 candidates whose
  text would equal S_T. Accepted candidates without a table entry stay in the
  pool; their units have no shift gap and drop out of the gap correlation only.
- Generation shared x113's cluster tree (`<workdir>/x113`, data
  under `data/runs/x116/`) instead of a separate x116 tree.
- The account's fair-share priority kept Llama and Phi workers pending for
  about 3 h. Their pending jobs were opened to the remaining GPU partitions
  (resumable per cell), and idle pending workers of finished models were
  cancelled (9 jobs, zero runtime).
- Judging ran per model as each model's generation finished, not after all
  eight.
- The plan said nothing on the uptake judge; x116 judged `refusal2` only.

## Artifacts

- Results: `data/runs/x116/results.json` (per model `arm_counts_*`,
  `integrity`, `populations` core / llama_extra / hard1k / xstest / all with
  estimands, shift-gap and representation records, `per_unit`;
  `cross_model` with Holm), `results.md`; whole population
  `results_full.{json,md}`.
- Per model under `data/runs/x116/<tag>/`: `texts.json` (every unit's pool,
  the draw seed and index, the R_rand text and sha256, the S_T substitute and
  both shifts), `projections.json`, `judge/refusal2.jsonl`,
  `judge/cost_log.jsonl`; `generations/generations_shard{0..7}of8.json`
  (gitignored, kept on the cluster under
  `<workdir>/x113/data/runs/x116/<tag>/generations/`).
- Code: `code/x116_common.py`, `x116_build_texts.py`, `x116_generate.py`,
  `x116_project.py`, `x116_judge.py` (`--provider`, default DeepInfra),
  `x116_analyze.py` (`--full`), `x116.sbatch` (`genloop`, `project`).
- Compute: 12.1 GPU-hours (26 jobs: 17 completed, 9 idle pending workers
  cancelled), generation and projections. API spend $1.36 for `refusal2`
  (2,993 calls, DeepInfra route); no drafter calls.
