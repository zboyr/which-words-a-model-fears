# x113: a vocabulary table read off each model once, never touched by the prompt, picks the word that carries a natural over-refusal and names the synonym that removes it — +0.05 (Llama) to +0.29 (Qwen3.5-27B) of `refusal2` outside the direction's fitting vocabulary, as much as x106's direction-nominated substitution

Parent: [Experiment index](README.md). Joins the two lines of this project:
[x106](x106-natural-benign-lexical-carrier.md) lets the boundary direction
nominate the carrier word of a refused benign prompt by deleting each word in
turn, while [x109](x109-benign-appended-word-vocabulary-scan.md),
[x111](x111-vocabulary-table-dose-response.md) and
[x112](x112-cross-model-vocabulary-table.md) build, per model, a context-free
table over 71,583 words from the shift each causes when appended to a handful
of over-refused prompts. x113 asks whether the table can do x106's job.

On the Hard-1K prompts most of the eight models refuse (275 refused at or above
0.5 by at least five models, plus the 22 further cohort prompts Llama-3.2-3B
refuses at all) and x106's 100 safe XSTest prompts, each model's own table
nominates the highest-shift content word of each prompt, an external drafter
returns up to ten meaning-preserving substitutes for it, and the accepted
candidate the same table places lowest replaces it in place (**S_T**); the
control replaces the prompt's table-nearest-zero inert word by a synonym
(**C_T**). Units whose nomination or text touches the x85 crossed cohort — the
96 prompts crossed with 24 harm words and 24 neutral twins that every model's
direction was fitted on — are excluded, leaving 349–352 of the 397 per model.
**S_T removes +0.278 (Qwen3.5-4B), +0.145 (0.8B), +0.246 (2B), +0.232 (9B),
+0.287 (27B), +0.052 (Llama), +0.261 (Gemma), +0.204 (Phi) of `refusal2` on the
Hard-1K prompts, Holm p = 7 × 10⁻⁴ over the seven transfer models, while C_T
moves it by +0.011 to +0.050.** On the same prompts x106's direction-nominated
substitution removes +0.060 to +0.281, and the paired difference `r(S_T) − r(S)`
covers zero in seven of eight models (Qwen3.5-0.8B −0.033 [−0.067, −0.000]): a
table read once per model matches a per-prompt deletion pass of 20–50 forward
passes. The table names x106's exact top word on 0.32 (Gemma) to 0.42
(Qwen3.5-27B) of the prompts and puts it in its own top three on 0.60–0.67;
where the two agree S_T removes +0.39 to +0.51 and where they differ +0.04 to
+0.16. The substituted text's projection on the model's own direction falls
0.61–0.88 sd against C_T's 0.00–0.08 and tracks the per-prompt refusal change
at Spearman +0.31 (Llama) to +0.80 (4B). On XSTest, where the models refuse
0.005–0.222 at base, S_T removes +0.006 to +0.131. Keeping the x85 units
(`--full`) moves no model's Δ_S by more than 0.017.

## Question

x106 found the carrier word by deleting every word of the prompt in turn and
reading the direction — a per-prompt search. x109 / x111 / x112 found that a
word's position in a fixed per-model table predicts the change in refusal it
causes when appended to a held-out prompt. x113 asks whether that table, read
off once per model and never shown the prompt, can (1) nominate the word x106's
direction nominates, (2) name a substitute whose in-place replacement removes
the refusal, and (3) leave refusal alone when a table-nearest-zero word of the
same prompt is replaced instead.

## Design, as run

- **Models and contract.** The eight x106 models at their x106 snapshots, x106
  rendering (`x106_common.render_chat`, Llama with its stamped date), x101
  sampling (T = 1.0, top-p 0.9, top-k 50, ten samples per cell, 2,048 new
  tokens with one 4,096 retry), seeds `x106_common.stable_seed(tag, prompt id,
  x106 text label)` so S_T uses x106's S seed and C_T its C seed.
- **Population, 397 units** (`data/runs/x113/prompts.json`, recomputed from
  x106's `O_base` `refusal2` rows). **core**, 275: x106 cohort Hard-1K prompts
  with `refusal2` ≥ 0.5 in at least five of the eight models; per-model r(O)
  Qwen3.5-4B 0.95, 0.8B 0.39, 2B 0.69, 9B 0.96, 27B 0.91, Llama 0.09, Gemma
  0.91, Phi 0.52. **llama_extra**, 22: the remaining cohort Hard-1K prompts with
  Llama `refusal2` > 0, descending (all 22 that exist; Llama r(O) 0.42).
  **xstest**, 100: x106's safe prompts of the homonyms, figurative language,
  safe contexts and safe targets categories.
- **Tables** (`data/runs/x113/tables/<tag>.npz`). Qwen3.5-4B: x111's BP table.
  The seven others: the x112 scan summed over the two boundary tokens, averaged
  over the eight scan prompts, divided by the model's x112 sd unit — reproducing
  x112's own `table.npz` to 2 × 10⁻⁷ sd and its percentiles exactly. Candidate
  list: x109's 71,583 words (`words_sha256` `e68c9703…`, the list x110 / x111 /
  x112 share), percentiles over x111's 47,371 eligible words.
- **Stage 1, nomination** (`x113_nominate.py`). Per prompt every whitespace
  word, edge punctuation stripped, lower-cased; x106's stop words excluded;
  exact lookup, then an inflection fallback (possessive `'s`, plural `-s` /
  `-es` / `-ies`, `-ed`, `-ing` with the final `e` or doubled consonant
  restored), then for a hyphenated word its highest-shift part. 0.984 of content
  words are found in every model (the rest are contractions, numerals and
  misspellings). The nominee is the highest-shift content word: median
  within-prompt z 2.4–3.4, median table percentile 92–96.
- **Stage 2, texts** (`x113_build_texts.py`, per model). x106's drafter
  (`deepseek/deepseek-v4-flash-0731`, temperature 0, seeded, x106's reasoning
  budget and no-reasoning retry) returns up to ten meaning-preserving
  substitutes with no harm association (9.4–9.6 per unit, 7.1–7.4 of them
  passing the checks); a candidate's table shift is its word's, or for a phrase
  the highest among its words after stop words are removed; the accepted
  candidate with the **lowest** shift in that model's table is used, which is
  the drafter's first choice only 8–12 % of the time. x106's hedge rule is
  unchanged (59–83 units per model): the phrase the drafter quotes is replaced
  by the lowest-shift accepted neutral phrase, and the splice keeps the
  separator the span carried. C_T replaces the content word outside the nominee
  whose table shift is nearest zero within x106's Zipf band, by the drafter's
  synonym. x106's structural checks apply verbatim (no x83 lexicon word, no x106
  or x113 nominated word, substitute ≠ nominee, unique hedge span, no new
  repeated adjacent word), with up to three redrafts; 16–25 units per model have
  no S_T because the drafter judged the nominee to have no neutral
  meaning-preserving substitute. Mean shift: nominee +0.26 to +0.77 sd,
  substitute −0.16 to −0.45, drop +0.51 to +1.04, control word −0.18 to +0.01.
  3–11 S_T texts and 42–61 C_T texts per model are byte-identical to x106's S /
  C text of the same prompt, and copy x106's generations and `refusal2` labels.
- **Stage 3, generation** (`x113_generate.py`, `x113.sbatch`). O is x106's
  `O_base` rows, reused. 6,186 S_T / C_T cells over the eight models, of which
  458 copied from x106 and 5,728 generated (57,280 responses) by work-stealing
  Slurm workers over eight resumable shards per model, Qwen3.5-27B at
  `--mem=140G`; 207 cells were regenerated after the splice fix below. Retry
  rows 0–37 per model; truncation after the retry at most 0.0014.
- **Stage 4, judges** (local, `x113_judge.py`). `refusal2`, the x64 contract
  (ten responses per call), on every S_T and C_T cell: one row needed a second
  pass with shifted attempt seeds, none the per-response fallback. **Uptake**,
  once per cell with its ten responses — x110's delimited prompt adapted to
  "does the response use the substituted word or phrase, or treat it as part of
  the request", reasoning disabled per
  [infrastructure.md](../infrastructure.md): 6,186 cells, zero failures. The two
  judges ran in separate loops at 24 threads.
- **Stage 5, representation** (`x113_project.py`). O, S_T and C_T read at
  x106's site through x106's reader on the model's own direction, summed over
  the two trailing prefill positions, in the model's x106 Hard-1K sd unit; the
  site audit passed on every model and the O projections reproduce x106's Stage
  0a values to at most 0.118 sd.
- **Statistics.** Prompt-paired two-stage bootstrap, 10,000 replicates (prompts
  resampled within stratum, then the ten binary responses), seed 11,300; Holm
  within the (Δ_S, Δ_C) pair and over the seven transfer models per contrast,
  Qwen3.5-4B the reference row as in x103 / x105 / x106.

### Exclusion: the x85 fitting vocabulary

The directions these tables are read on were fit in x85 on 96 prompts crossed
with 24 harm words and 24 neutral controls (`code/x85_common.py` `WORD_PAIRS`),
so those 48 words are in-sample for the table: the table ranks them from the
texts the direction was fitted on, and the nomination can inherit that ranking.
None of the 397 x113 prompts is an x85 prompt (zero overlap by text sha256 with
`x85_cohort.json`, `cohort_sha256` `93784a93…`), and neither the x111 BP table's
48 scan prompts nor x112's eight contain one, so the leak is lexical only.

A unit leaves the primary analysis when its table nominee is one of the 48
words, when its prompt contains an x85 harm word anywhere, or when either arm's
substitute, control word or synonym is an x85 word. Per model that removes 45–48
of the 397 units: 33–40 x85-harm-word nominees (`illegal` 11–15, `bomb` 7–9,
`harmful`, `malware`, `murder`, `stalking`, `phishing`, `rape`), 2 neutral-word
nominees on Llama, 43 prompts carrying an x85 harm word, 0–1 S_T substitutes and
2–4 C_T control words. **Leak-free n per model:** Qwen3.5-4B 350, 0.8B 351, 2B
352, 9B 352, 27B 351, Llama 349, Gemma 351, Phi 351 (core 235–236, the 22 Llama
add-on prompts untouched, XSTest 92–94). The exclusion is frozen in
`data/runs/x113/x85_exclusions.json`; `x113_analyze.py` applies it by default and
`--full` writes the whole-population `_full` siblings. **The full population
moves no model's Δ_S by more than 0.017** (Qwen3.5-0.8B +0.162 against +0.145;
the other seven by at most 0.012), and leaves the nomination agreement, the
projection change and the uptake ordering unchanged. The excluded units are
reported on their own as the `x85_excluded` population: Δ_S there is +0.023
(Llama) to +0.297 (Qwen3.5-2B), of the same size as the primary estimate.

## Results

### The substitution, Hard-1K (`refusal2`, x85 units excluded)

| model | n S_T | r(O) | r(S_T) | r(C_T) | Δ_S [95% CI] | Holm over transfer | Δ_C [95% CI] | S_T − C_T | x106 Δ_S, same prompts | r(S_T) − r(S) paired |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B | 251 | 0.929 | 0.654 | 0.893 | +0.278 [+0.229, +0.326] | ref | +0.037 [+0.015, +0.060] | +0.237 | +0.281 [+0.228, +0.334] | +0.023 [−0.031, +0.080] |
| Qwen3.5-0.8B | 246 | 0.331 | 0.178 | 0.295 | +0.145 [+0.110, +0.181] | 7e−4 | +0.036 [+0.004, +0.069] | +0.103 | +0.104 [+0.069, +0.141] | −0.033 [−0.067, −0.000] |
| Qwen3.5-2B | 249 | 0.629 | 0.380 | 0.611 | +0.246 [+0.203, +0.288] | 7e−4 | +0.018 [−0.014, +0.051] | +0.228 | +0.225 [+0.179, +0.273] | −0.013 [−0.063, +0.038] |
| Qwen3.5-9B | 247 | 0.937 | 0.704 | 0.914 | +0.232 [+0.185, +0.280] | 7e−4 | +0.023 [+0.004, +0.043] | +0.209 | +0.259 [+0.206, +0.314] | +0.027 [−0.024, +0.080] |
| Qwen3.5-27B | 249 | 0.897 | 0.608 | 0.847 | +0.287 [+0.236, +0.340] | 7e−4 | +0.050 [+0.025, +0.076] | +0.237 | +0.265 [+0.210, +0.320] | −0.014 [−0.070, +0.043] |
| Llama-3.2-3B | 252 | 0.127 | 0.075 | 0.116 | +0.052 [+0.027, +0.078] | 7e−4 | +0.011 [−0.012, +0.034] | +0.041 | +0.060 [+0.029, +0.092] | +0.017 [−0.009, +0.045] |
| Gemma-4-E2B | 250 | 0.879 | 0.620 | 0.855 | +0.261 [+0.210, +0.314] | 7e−4 | +0.023 [−0.000, +0.049] | +0.238 | +0.267 [+0.211, +0.325] | −0.000 [−0.059, +0.060] |
| Phi-4-mini | 249 | 0.481 | 0.278 | 0.448 | +0.204 [+0.158, +0.251] | 7e−4 | +0.033 [−0.001, +0.068] | +0.175 | +0.186 [+0.136, +0.235] | +0.014 [−0.038, +0.065] |

Every Δ_S interval excludes zero and Holm within the (Δ_S, Δ_C) pair is
2 × 10⁻⁴ in every model; of the eight Δ_C estimates four have intervals
excluding zero, all at or below +0.050. Among the units that have an S_T,
`refusal2` falls below 0.5 for 0.25 (Qwen3.5-9B) to 0.92 (Llama) of them; 2–5 %
of units have no S_T. Across the eight models Δ_S tracks r(O) at Spearman +0.74:
Llama, refusing 0.127 of the population at base, has the least to remove.

### By population

| model | core Δ_S | Llama add-on Δ_S | XSTest r(O) | XSTest Δ_S | XSTest Δ_C |
|---|---|---|---|---|---|
| Qwen3.5-4B | +0.278 [+0.227, +0.329] | +0.276 [+0.124, +0.436] | 0.095 | +0.058 [+0.014, +0.109] | −0.024 |
| Qwen3.5-0.8B | +0.160 [+0.122, +0.198] | −0.014 [−0.073, +0.038] | 0.068 | +0.060 [+0.022, +0.102] | −0.003 |
| Qwen3.5-2B | +0.262 [+0.218, +0.307] | +0.060 [−0.030, +0.150] | 0.115 | +0.082 [+0.035, +0.134] | −0.026 |
| Qwen3.5-9B | +0.239 [+0.190, +0.291] | +0.152 [+0.038, +0.273] | 0.133 | +0.103 [+0.052, +0.164] | +0.003 |
| Qwen3.5-27B | +0.290 [+0.238, +0.344] | +0.257 [+0.071, +0.459] | 0.085 | +0.056 [+0.011, +0.107] | −0.022 |
| Llama-3.2-3B | +0.039 [+0.014, +0.065] | +0.186 [+0.068, +0.314] | 0.005 | +0.006 [+0.000, +0.018] | −0.018 |
| Gemma-4-E2B | +0.284 [+0.230, +0.341] | +0.005 [−0.127, +0.138] | 0.222 | +0.131 [+0.059, +0.210] | −0.007 |
| Phi-4-mini | +0.219 [+0.169, +0.269] | +0.055 [−0.032, +0.141] | 0.149 | +0.112 [+0.053, +0.176] | −0.013 |

The Llama add-on prompts are the ones Llama itself refuses (r(O) 0.42, Δ_S
+0.186 [+0.068, +0.314]); for the models that barely refuse them the interval
covers zero. On XSTest seven of eight Δ_S intervals exclude zero while every Δ_C
estimate is at or below zero.

### Nomination against x106

| model | nominee = x106 top word | nominee inside the x106 unit | x106 top word's table rank (median) | x106 top word in table top 3 | agreement with the Qwen3.5-4B nominee | x112 table ρ with the Qwen3.5-4B BP table |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | 0.372 | 0.403 | 2 | 0.663 | – | – |
| Qwen3.5-0.8B | 0.384 | 0.391 | 2 | 0.636 | 0.574 | +0.58 |
| Qwen3.5-2B | 0.403 | 0.419 | 2 | 0.643 | 0.643 | +0.70 |
| Qwen3.5-9B | 0.376 | 0.388 | 2 | 0.640 | 0.640 | +0.76 |
| Qwen3.5-27B | 0.419 | 0.422 | 2 | 0.674 | 0.620 | +0.73 |
| Llama-3.2-3B | 0.346 | 0.374 | 3 | 0.619 | 0.490 | +0.45 |
| Gemma-4-E2B | 0.323 | 0.323 | 2 | 0.603 | 0.479 | +0.46 |
| Phi-4-mini | 0.403 | 0.422 | 2 | 0.605 | 0.516 | +0.36 |

The seven transfer models' agreement with Qwen3.5-4B's nominee tracks how well
their x112 table correlates with the Qwen3.5-4B BP table (Spearman +0.75 over
the seven models).

**Where the two nominations agree, the substitution is twice as strong.** On the
prompts where the table's nominee is x106's top word Δ_S is +0.512 (4B), +0.157
(0.8B), +0.389 (2B), +0.407 (9B), +0.491 (27B), +0.080 (Llama), +0.472 (Gemma),
+0.393 (Phi); where they differ, +0.142, +0.138, +0.153, +0.132, +0.148, +0.037,
+0.161, +0.082. x106's own substitution splits the same way on the same prompts
(+0.066 to +0.434 agreeing, +0.057 to +0.222 differing), so the split is a
property of the prompts, not of which nomination rule is used. Δ_C is flat in
both halves (+0.004 to +0.066).

Per prompt, the nominee's table shift and the shift drop barely order the effect
within a model (Spearman −0.09 to +0.16 for Δ_S, the same range for Δ_C): the
table says which word, not how much that prompt will move.

### Representation

| model | proj(O) (sd) | proj(S_T) − proj(O) [95% CI] | proj(C_T) − proj(O) | share of S_T below O | ρ(projection change, refusal change) S_T / C_T |
|---|---|---|---|---|---|
| Qwen3.5-4B | +3.09 | −0.86 [−0.99, −0.73] | −0.06 | 0.748 | +0.80 / +0.46 |
| Qwen3.5-0.8B | +3.37 | −0.72 [−0.81, −0.64] | −0.05 | 0.837 | +0.54 / +0.49 |
| Qwen3.5-2B | +2.81 | −0.74 [−0.84, −0.64] | −0.03 | 0.783 | +0.66 / +0.33 |
| Qwen3.5-9B | +3.26 | −0.76 [−0.89, −0.62] | −0.04 | 0.748 | +0.76 / +0.34 |
| Qwen3.5-27B | +3.28 | −0.88 [−1.05, −0.71] | −0.08 | 0.756 | +0.79 / +0.40 |
| Llama-3.2-3B | +0.87 | −0.75 [−0.88, −0.63] | −0.06 | 0.774 | +0.31 / +0.29 |
| Gemma-4-E2B | −0.04 | −0.61 [−0.72, −0.50] | −0.00 | 0.732 | +0.71 / +0.36 |
| Phi-4-mini | +4.57 | −0.80 [−0.94, −0.65] | −0.05 | 0.733 | +0.75 / +0.46 |

x106's direction-nominated substitution lowered the projection by 0.43–0.88 sd
on the same models, so the table's substitute moves the state as far as the
direction's own choice does.

### Uptake

The substituted word or phrase is used, or treated as part of the request, in
0.445 (Gemma) to 0.701 (Llama) of S_T responses and 0.383 to 0.617 of C_T
responses. `refusal2` is lower where it is taken up — S_T 0.025 vs 0.191
(Llama), 0.117 vs 0.467 (Phi), 0.432 vs 0.771 (Gemma), 0.631 vs 0.679
(Qwen3.5-4B) — and the same ordering holds on C_T.

Figures: `data/runs/x113/substitution.pdf` (per model Δ_S, Δ_C and x106's Δ_S;
the agreement split; the projection change) and `nomination.pdf` (agreement with
x106; Δ_S by population); the `_full` siblings carry the whole population.

## Reading

- **The carrier word can be found without looking at the prompt.** A table of
  71,583 context-free word shifts, read off once per model from eight prompts,
  nominates a word whose in-place replacement removes as much refusal as x106's
  per-prompt deletion pass over the same prompts, in all eight models — and it
  does so after every word the direction was fitted on has been removed from the
  population.
- **The nominations differ but the effect does not.** The table picks x106's
  exact word about 40 % of the time and puts it in its top three about two
  thirds of the time; where the rules agree the effect is roughly +0.45, where
  they disagree about +0.15 for both rules. Natural prompts often carry more
  than one word the model reacts to.
- **It is the word, not the edit.** Replacing the table-nearest-zero word of the
  same prompt by a synonym moves `refusal2` by at most +0.050 and the projection
  by at most 0.08 sd.
- **Effect size follows the base rate.** Δ_S against r(O) across models is
  Spearman +0.74; Llama refuses 0.127 of the population and loses +0.052, the
  models refusing 0.63–0.94 lose +0.20 to +0.29.

## Deviations from the plan

- The plan analysed all 397 units. The primary analysis excludes the units whose
  nomination or text touches the x85 crossed cohort, the cohort the directions
  were fitted on (`data/runs/x113/x85_exclusions.json`, 45–48 per model), as
  x103 / x105 / x110 / x111 / x112 do; the whole-population analysis is kept as
  `results_full.json` / `results_full.md` and the `_full` figures from
  `x113_analyze.py --full`, with the same estimators, seeds and draw order.
- The plan's Stage 1 ranked only exact and inflected lookups. 17–20 prompts per
  model have a hyphenated content word (`non-destructive`, `self-harm`) with no
  entry of its own; those take the highest-shift part after stop words are
  removed, recorded as `hyphen:` in the nomination rows.
- The first build spliced a hedge substitute over the span the drafter quoted,
  which often opens with its separator (", ensuring … no one"), so the neutral
  phrase was glued to the preceding word in 20–37 S_T texts per model. The
  splice now carries the span's leading and trailing separators (or inserts a
  space), the texts were rebuilt from the same drafter cache without re-drafting,
  and the 207 affected S_T cells (qwen 25, qwen08b 37, qwen2b 28, qwen9b 27,
  qwen27b 20, llama 20, gemma 26, phi 24) were regenerated with their unchanged
  seeds and re-judged by both judges; no C_T text was affected. Every headline
  number is from the rebuilt texts.
- The plan assumed the control word would always exist; one Qwen3.5-4B prompt
  has no admissible control (the drafter's only synonym carried an x83 lexicon
  word), so that model has 396 C_T texts and the other seven have 397.
- Generation ran as work-stealing Slurm workers over eight resumable shards per
  model (x110's `genloop`), not as one array per model, so that any freed GPU
  could be used.
- The drafter cache is keyed by the hash of the request, so models that nominate
  the same word of the same prompt share one call; 2,077 exchanges over two
  concurrent passes (`drafts_a.jsonl` for the five Qwen tags, `drafts_b.jsonl`
  for Llama / Gemma / Phi, 1,827 distinct requests) covered the eight models'
  3,176 units.

## Artifacts

- Results: `data/runs/x113/results.json` (per model `populations` for core /
  llama_extra / hard1k / xstest / all and the `x85_excluded` units,
  `x85_exclusion`, `nomination`, `integrity`, `per_prompt`; `cross_model`),
  `results.md`, `substitution.{pdf,png}`, `nomination.{pdf,png}`; whole
  population `results_full.{json,md}`, `substitution_full.{pdf,png}`,
  `nomination_full.{pdf,png}`.
- Exclusion: `data/runs/x113/x85_exclusions.json` (the rule, the x85 cohort file
  and `cohort_sha256`, the 48 words, the x85 prompt sha256s, and per model every
  excluded unit id with its reasons and the retained counts per stratum).
- Design: `data/runs/x113/prompts.json` (397 units with the selection rules,
  stratum, per-model r(O) and the x106 unit; `units_sha256` `e6961abf…`),
  `tables.json` + `tables/<tag>.npz` (the eight tables with percentiles and
  eligibility), `drafts_a.jsonl` + `drafts_b.jsonl` (every drafter exchange).
- Per model under `data/runs/x113/<tag>/`: `nominations.json` (every word with
  its table entry, the nominee, the x106 agreement), `texts.json` (every text
  with its sha256, the nominee, all candidates with their table shift, the
  chosen substitute, the control and its shift, and whether the text is x106's),
  `projections.json`, `judge/refusal2.jsonl`, `judge/uptake.jsonl`,
  `judge/cost_log.jsonl`; `generations/generations_shard{0..7}of8.json`
  (≈ 170 MB in all, shipped in this repository).
- Code: `code/x113_common.py`, `x113_select.py`, `x113_nominate.py`
  (`--tables`), `x113_build_texts.py`, `x113_exclusions.py`, `x113_generate.py`,
  `x113_project.py`, `x113_judge.py` (`--judge refusal2|uptake`),
  `x113_analyze.py` (`--full`), `x113.sbatch` (`project`, `gen`, `genloop`);
  the cluster tree `<workdir>/x113/`.
- Compute: 22.0 GPU-hours (generation 21.6 including the 207 cells regenerated
  after the splice fix, projections 0.35 over the two passes), 96 jobs, all
  completed. Wall clock 22:12 UTC to 05:18 UTC, about 7 h from the start to the
  finished analysis including the text fix. API spend $13.12: drafter $2.46
  (2,077 exchanges, 9.8 M tokens), `refusal2` $6.04 (5,935 calls, 458 labels
  copied from x106), uptake $4.62 (6,393 calls).
