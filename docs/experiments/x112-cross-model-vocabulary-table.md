# x112: on each of the seven other models, a word's position in the model's own eight-prompt vocabulary table orders its change in refusal on held-out prompts, from −0.11 at the bottom percentile to +0.36 at the top averaged over models; the own table ranks the 40 words at ρ +0.54 to +0.96 and above the Qwen3.5-4B table on five of seven

Parent: [Experiment index](README.md). Carries
[x111](x111-vocabulary-table-dose-response.md)'s held-out dose-response
check to the seven transfer models of
[x106](x106-natural-benign-lexical-carrier.md).

x111 found on Qwen3.5-4B that a word's position in a vocabulary table
scanned on over-refused Hard-1K prompts (BP) predicts the change in
`refusal2` it causes when appended to held-out prompts the model refuses
about half the time. x112 builds the same kind of table on each of the
seven other models from eight x100 BP prompts (the eight whose own word
ranking agrees best with x111's BP table), with the model's own x106
direction, read block and sd unit, over x109's 71,583 candidate words. On
each model it then draws 40 words stratified jointly on the model's own
table and the Qwen3.5-4B BP table, appends each to 32 held-out prompts (16
benign Hard-1K, 16 harmful) whose fresh twenty-sample `refusal2` sits near
0.5, and measures Δ_EO = r(E) − r(O). Tables, words and prompts were fixed
before any held-out prompt was scanned; the in-context scan of the 40 words
on the 32 prompts was run afterwards, as a readout. Every number below is
over held-out prompts and words outside the x85 crossed cohort the
directions were fitted on: the 13 harmful prompts of that cohort over four
models are excluded and the stratum is refilled by 13 prompts chosen by the
same rule, and one Llama word is excluded (Design, "Held-out prompts",
"Top-up" and "Words"); the pre-top-up and whole-grid values are in two
notes at the end of the Results.

**On every model the change in refusal rises with the table position.**
Averaged over the seven models (pooled prompts), the concordant bins run
−0.114, −0.121, −0.004, +0.047, +0.102, +0.133, +0.249, +0.361 from the
bottom percentile to the top; top minus bottom +0.475 [+0.444, +0.507]. Per
model, top minus bottom is +0.32 (Qwen3.5-27B) to +0.71 (Qwen3.5-0.8B),
every interval above zero, with 5 to 7 of 7 adjacent bin pairs in order
(7/7 on Phi-4-mini, Qwen3.5-0.8B and Qwen3.5-9B). The bottom bins are
style and register words (bookish, poetic, playfully, metaphorically,
allegorical) and lower refusal on six models; the top bins are sexual
terms, slurs, violence and assassination words and raise it by +0.30 to
+0.45 on every model. **Each model's own table ranks the words, and on five of seven it
ranks them better than the Qwen3.5-4B table.** Over the 40 words (39 on
Llama), ρ(own table, Δ_EO) is +0.78 Gemma, +0.54 Llama, +0.91 Phi, +0.96
Qwen3.5-0.8B, +0.82 Qwen3.5-2B, +0.83 Qwen3.5-9B, +0.78 Qwen3.5-27B
(permutation p ≤ 0.0004 each); the Qwen3.5-4B BP table reaches +0.59 to
+0.85. The own table is higher by +0.19 [+0.12, +0.25] on Phi-4-mini, +0.22
[+0.16, +0.28] on Qwen3.5-0.8B, +0.08 [+0.02, +0.13] on Qwen3.5-27B, +0.07
[−0.03, +0.14] on Gemma and +0.06 [+0.00, +0.14] on Qwen3.5-2B; it is lower
by 0.02 to 0.05 (intervals across zero) on Llama-3.2-3B and Qwen3.5-9B. On
the concordant words the two tables agree (ρ +0.62 to +0.97 for both); the
difference sits in the discordant words, where across the 112 (model, word)
pairs the own table's line is closer to the observed Δ_EO for 63 and the
Qwen table's for 49. The tables themselves agree across models at Spearman
+0.36 (Phi) to +0.76 (Qwen3.5-9B) with the Qwen3.5-4B BP table over the
47,371 eligible words, and share 212 to 279 of their top 474 words with it
(chance 4.7). The in-context shift of the 40 words on the held-out prompts
ranks them at +0.66 to +0.97 per word.

## Design, as run

- **Models and contract.** The seven x106 transfer models (Gemma-4-E2B,
  Llama-3.2-3B, Phi-4-mini, Qwen3.5-0.8B / 2B / 9B / 27B) at their x106
  snapshots, x106 rendering (`x106_common.render_chat`, Llama with its
  stamped date; every O rendering hashes to the rendered chat its source
  run (x106, x101 or x108) recorded), each model's x106 direction (`x106_common.load_mu`), hook block
  (`x106_inputs.json`) and Hard-1K sd unit (population sd of the x106
  Stage 0a projections over the 1,319 Hard-1K prompts: 12.85, 3.77, 18.84,
  1.35, 4.83, 30.83, 41.11). x81/x85 sampling (temperature 1.0, top-p 0.9,
  top-k 50, 2,048 new tokens with a 4,096 retry of the whole cell), twenty
  samples per cell, seeds `sha256(x112|112|tag|prompt id|condition)`.
- **Scan prompts.** x111's `representative_prompts.json`, table BP, top
  eight: b00393, b00520, b00490, b00127, b00711, b00124, b00335, b01314
  (texts from `x100_cohort.json`), the same eight on every model. These
  eight were chosen for the agreement of Qwen3.5-4B's per-prompt word
  ranking with the x111 BP table, so the between-model table agreement
  below (Spearman with the Qwen3.5-4B table) is a figure on selected
  prompts.
- **Scan.** The x109 candidate list (`words_sha256` `e68c9703…`, same order
  as x111) appended as `text + " " + w`, read at the hook block's output at
  the two boundary tokens, summed, through a cached prefix: P + text is
  prefilled once and every per-layer tensor state of the cache up to the
  read block (attention keys and values; DeltaNet conv and recurrent states
  on the Qwen3.5 models; Gemma-4's 15 cache layers, sliding-window and
  full) is repeated to the batch. Each word's continuation ids are read off
  the tokenized full text, so no junction approximation is made; no word
  on any model tokenized differently at the junction. Table = mean over the
  eight prompts in the model's sd unit; percentiles over x111's 47,371
  eligible words.
- **Scan audit, every prompt of every model.** O projection cached vs full
  prefill: at most 0.035 sd per token (Gemma), 0.006–0.028 elsewhere. On
  128 sampled words per prompt the cached shift against batched full
  prefill: Spearman 0.981–0.996 on Gemma (sample shift sd 0.19–0.48),
  0.994–1.000 on the other six; cached against batch-1 full prefill
  differs by at most 0.059 sd per token on Gemma and 0.024–0.039 elsewhere,
  the same size as batched full prefill against batch 1 (0.048 Gemma,
  0.000–0.050 elsewhere). Every model ran on the cached path. Scan time per
  model (eight prompts): Gemma 1.6 min, Phi 1.7, Llama 2.5, Qwen3.5-2B 5.0,
  0.8B 8.9, 9B 25.6, 27B 70.2 (batch 256 after an OOM at 1,024).
- **Held-out prompts, 32 per model.** Benign: the model's x106 O_base
  Hard-1K rows (476); harmful: the union of its x101 O rows (130, x83
  cohort) and x108 O_base rows (719, OR-Bench Toxic), historical
  `refusal2` over ten samples; x100's HP / BP, x109's 40 and the eight scan
  prompts dropped (32 benign, 13 harmful). Candidates ordered by
  |rate − 0.5| inside [0.3, 0.7] with seeded ties; the nearest 24 per pool
  got a fresh O cell (twenty samples) and `refusal2`, the 16 with fresh
  rate in the band nearest 0.5 were kept, the next candidates were run
  only for a pool still short. Gemma's harmful pool held 15 eligible
  prompts in [0.3, 0.7] and used [0.2, 0.8] (26 candidates). Llama's benign
  pool held 17 in [0.3, 0.7]; after all 17 were run and 5 landed in band it
  was widened to [0.2, 0.8] (13 more run, 16 in band). Gemma benign (6
  more) and Qwen3.5-9B benign (4 more) were each extended once; Llama
  harmful 6 more. Fresh O cells run: 54, 60, 48, 48, 48, 52, 48 (358). Fresh
  versus historical rate over the judged candidates: Spearman −0.18 to
  +0.72. The harmful prompts are mostly OR-Bench Toxic (13–16 of 16) on
  Gemma, Llama, Phi and the two small Qwens, and mostly x83 (12 and 10 of
  16) on Qwen3.5-9B and 27B. The selection rule did not drop the 96
  prompts of the x85 crossed cohort, on whose texts every model's
  direction was fitted (the x88 consensus direction that `load_mu` reads;
  [x85](x85-crossed-keyword-prompt-latent-decomposition.md),
  [x88](x88-eight-model-deep-dose-transfer.md)); the x83 harmful rows that
  are x85 prompts are excluded from the analysis after the run: Gemma 2
  (`x83:20006`, `x83:24708`), Qwen3.5-0.8B 1 (`x83:44349`), Qwen3.5-9B 4
  (`x83:22450`, `21446`, `3493`, `24939`), Qwen3.5-27B 6 (`x83:489`,
  `24677`, `21504`, `23009`, `44087`, `3493`); none on Llama, Phi or
  Qwen3.5-2B, and no benign prompt. Prompts analysed per model (benign /
  harmful): Gemma 30 (16 / 14), Llama 32 (16 / 16), Phi 32 (16 / 16),
  Qwen3.5-0.8B 31 (16 / 15), Qwen3.5-2B 32 (16 / 16), Qwen3.5-9B 28 (16 /
  12), Qwen3.5-27B 26 (16 / 10). The exclusion is frozen in
  `data/runs/x112/x85_exclusions.json` with the x85 cohort's sha256, and
  `x112_analyze.py` applies it by default.
- **Top-up of the harmful stratum.** On the four models that lost
  prompts, the harmful stratum was refilled to 16 prompts outside the x85
  cohort by the selection rule above (`x112_select.py --topup-request`,
  `--topup-prompts`; `<tag>/topup_request.json`, `prompts_topup.json`;
  `prompts.json` is unchanged): the candidates of `pool_candidates.json`
  that are neither chosen nor x85 prompts, inside the pool's band, in the
  pool's |rate − 0.5| order; a candidate whose pool-stage O cell had been
  judged in band is reused first, and for the shortfall left the next
  unjudged candidates got a fresh O cell (over-draw max(2 × shortfall, 4),
  bounded by the candidates left), of which the in-band ones nearest 0.5
  were kept. Added, with the fresh rate: Qwen3.5-27B six from an over-draw
  of twelve (eleven in band; `x83:21306` 0.50, `orbench_toxic:00449`
  0.50, `orbench_toxic:00441` 0.50, `x83:24554` 0.50,
  `orbench_toxic:00504` 0.55, `x83:43137` 0.45); Qwen3.5-9B three
  pool-stage cells (`x83:44860` 0.65, `orbench_toxic:00541` 0.65,
  `x83:24267` 0.30) and one of an over-draw of four (`xstest:v2-138` 0.55;
  two in band, 0.15 and 0.25 out); Gemma one pool-stage cell
  (`orbench_toxic:00430` 0.20) and one of the last two candidates of its
  [0.2, 0.8] pool (`x83:4376` 0.20; the other 0.85); Qwen3.5-0.8B one
  pool-stage cell (`orbench_toxic:00132` 0.70), no over-draw. Fresh O
  cells run for the top-up: 18. Every model is analysed over 16 benign and
  16 harmful prompts; the added prompt's O cell is the one that selected
  it, as for the originals, and it got the same 40 E cells, an `Orep`
  cell, both judges and the in-context scan (`key_words_scan_topup.json`).
  The harmful prompts are now x108 rows (OR-Bench Toxic, with the XSTest
  contrast prompts x108 carried) 14, 16, 15, 15, 15, 6, 9 of 16 on Gemma,
  Llama, Phi, Qwen3.5-0.8B, 2B, 9B and 27B, the rest x83; XSTest among
  the x108 rows 2, 3, 1, 3, 0, 3, 0.
- **O cells.** The fresh O cell that selected a prompt is its O cell in the
  grid. A second O cell per chosen prompt (condition `Orep`, its own seed,
  224 cells) was generated afterwards and is reported beside it.
- **Words, 40 per model.** x111's rules with the model's own table in place
  of the benign table and the Qwen3.5-4B BP table in place of BP: three
  concordant words from each of eight joint percentile bins and four from
  each discordant quadrant (Qwen ≤ p5 with own ≥ p25; own ≤ p5 with Qwen ≥
  p25; Qwen ≥ p95 with own ≤ p75; own ≥ p95 with Qwen ≤ p75), seed 112, a
  seeded walk per stratum skipping proper nouns (words capitalised in
  standard English: personal and place names, nationalities and languages,
  organisations, brands, fictional names; 84 judged over the seven walks)
  and stem-sharing words (two drops). Every stratum filled on every model.
  Token counts are the model's own. On Llama one chosen word, `rape`
  (concordant bin [99,100]), is one of the 24 x85 harm words and is
  excluded like the x85 prompts: Llama is analysed over 39 words, its top
  bin holds two, and Holm runs over 39. No other chosen word on any model
  is an x85 harm word or neutral twin.
- **Cells.** Per model 32 original prompts × (O, 40 E_w) = 1,312 cells × 20
  samples; 9,184 cells and 183,680 responses generated over the seven
  models in the original run, plus the 224 `Orep` cells and the 134 pool O
  cells not kept; the top-up added 13 prompts × 40 E_w = 520 E cells, 13
  `Orep` cells and 18 pool O cells (one top-up cell regenerated at 4,096
  tokens, on Qwen3.5-27B; no response truncated or empty); 9,152 cells
  enter the analysis (224 O, 8,928 E). No cell is missing. Cells regenerated at 4,096 tokens in the original run:
  Qwen3.5-0.8B 206 (106 responses still truncated), Qwen3.5-9B 48 (11),
  Qwen3.5-2B 7, Phi and 27B 1, none on Gemma and Llama. Mean generated
  tokens O / E over the analysed cells: Gemma 445 / 379, Llama 295 / 288,
  Phi 207 / 192, 0.8B 603 / 646, 2B 434 / 482, 9B 518 / 463, 27B 396 /
  386.
- **In-context scan.** The same reader on the 40 words × 32 held-out
  prompts per model, run after words and prompts were fixed; cached vs full
  O at most 0.065 sd per token; the 13 top-up prompts were scanned the same
  way after their selection (cached vs full O at most 0.028 sd per token,
  on Qwen3.5-27B).
- **Judges.** `refusal2`, x64 contract (`deepseek/deepseek-v4-flash-0731`
  via OpenRouter), each cell judged as two units of ten; every unit of the
  grid, the pool, the top-up and `Orep` judged, none by the per-response
  fallback. Hijack (x111's prompt, reasoning disabled) on all 9,480 E cells
  (18,960 units).
- **Statistics.** x111's: prompt-level paired bootstrap, 10,000 draws per
  model (pooled: resampled within each population); permutation p over
  10,000 word permutations; Holm over the 40 words (39 on Llama); partial ρ
  with token count and log wordfreq rank; per-cell ceiling from twenty
  random half splits with Spearman-Brown; discordant verdict against each
  table's least-squares line over the 24 concordant words (23 on Llama).
  The bootstrap, permutation and split seeds are those of the whole grid;
  only the prompt set is smaller. Across models the bin curve is the mean
  of the seven per-model curves, its interval from the mean of the seven
  independent bootstrap draws; ρ over all 280 (model, word) points uses
  percentiles.

## Results

### Dose-response over the concordant bins (`refusal2`, Δ_EO, pooled prompts, three words per bin)

Table: [Dose-response over the concordant bins](json/x112-cross-model-vocabulary-table__dose-response-bins.json) — per-bin Δ_EO for the mean of seven models and per model, adjacent pairs in order and top − bottom; the mean curve runs −0.114 to +0.361, top − bottom +0.475 [+0.444, +0.507], per model +0.324 [+0.240, +0.412] (Qwen3.5-27B) to +0.709 [+0.669, +0.750] (Qwen3.5-0.8B).

Adjacent pairs in order: 6/7 on the mean curve (the [0,1) / [1,5) pair),
per model 5, 6, 7, 7, 6, 7, 5 (Gemma to Qwen3.5-27B). Per population, the
mean of seven runs from −0.16 / −0.07 (benign / harmful) at the bottom to
+0.39 / +0.34 at the top, top minus bottom +0.548 [+0.508, +0.590] benign
and +0.403 [+0.355, +0.452] harmful, 6/7 and 7/7 adjacent pairs in order.
On Qwen3.5-27B the harmful prompts (16, nine of them OR-Bench Toxic) sit
above zero in every bin (+0.22 to +0.41) and the two bottom bins do not
lower refusal; top minus bottom +0.159 [+0.023, +0.308] with 5/7 pairs in
order; its own-table ρ on the harmful prompts alone is +0.46 [+0.13,
+0.66] (permutation p 0.003), the Qwen table's +0.43 [+0.09, +0.62] (p
0.005), against +0.83 / +0.76 on its benign prompts. Figure:
`data/runs/x112/dose_response.png`.

Table: [The words, by bin](json/x112-cross-model-vocabulary-table__words-by-bin.json) — the 24 concordant words per model from the bottom bin to the top, sexual terms and slurs given by category; the bottom bins are register words (bookish, poetic, playfully, metaphorically, allegorical), the top bins sexual terms, slurs and violence words.

### Own table vs Qwen3.5-4B BP table: ρ(table shift, per-word Δ_EO)

Table: [ρ over the 40 words, own table against the Qwen3.5-4B BP table](json/x112-cross-model-vocabulary-table__own-vs-qwen-rho.json) — pooled ρ with intervals for both tables, their paired difference, ρ per population and the in-context shift; own +0.54 (Llama) to +0.96 (Qwen3.5-0.8B), Qwen +0.59 to +0.85, own − Qwen +0.19 on Phi and +0.22 on Qwen3.5-0.8B.

Every permutation p is at most 0.0004. Over the 280 (model, word) points on
the percentile scale, ρ is +0.70 for the own tables and +0.63 for the Qwen
table (mean per-model ρ +0.80 vs +0.72). Words with pooled Δ_EO
Holm-significant at 0.05: 25, 11 (of 39), 25, 23, 17, 17, 31 of 40.

Table: [ρ on word subsets, partial ρ and the per-cell ceiling](json/x112-cross-model-vocabulary-table__rho-subsets-and-ceiling.json) — ρ on the concordant words (+0.62 to +0.97, the two tables within 0.01 of each other), on the middle four bins, partial ρ given token count and log frequency (own +0.45 to +0.96), and per-cell ρ own / Qwen / in-context against the split-half ceiling.

Figure: `data/runs/x112/tables_vs_effect.png`.

### Discordant words

Table: [Discordant words by quadrant](json/x112-cross-model-vocabulary-table__discordant-quadrants.json) — pooled Δ_EO per quadrant (four words each), which table's line comes closer per word and the side match; words high in the own table only +0.10 to +0.35 on every model.

Words high in the model's own table only raise refusal on every model
(+0.10 to +0.36, mean over models +0.22); words high in the Qwen table only
raise it on six (not on Qwen3.5-0.8B; mean +0.15). Over the 112 discordant
(model, word) pairs the own table's line is closer for 63 and the Qwen
table's for 49; side match 78 vs 74. The own table wins clearly where its ρ
lead is large: Qwen3.5-0.8B 14 of 16, Phi 11 of 16.

### Between-model table agreement (47,371 eligible words)

Table: [Between-model table agreement](json/x112-cross-model-vocabulary-table__table-agreement.json) — Spearman of each table with the Qwen3.5-4B BP table (+0.36 Phi to +0.76 Qwen3.5-9B) and the words shared at the two ends (212–279 of the top 474, chance 4.7).

Among the seven tables Spearman runs from +0.29 (Phi vs Qwen3.5-27B) to
+0.81 (Qwen3.5-9B vs 27B); the Qwen3.5 family agrees most with itself and
with the Qwen3.5-4B table. 60 words sit in the top 1 % and 25 in the bottom
1 % of all eight tables (the seven plus the Qwen3.5-4B BP table). The
Qwen3.5-4B benign table (x109) agrees with the seven at +0.17 to +0.45.
Per-model table spread (sd units, eligible words, p1 / median / p99): Gemma
−0.51 / +0.04 / +0.57, Llama −1.06 / −0.14 / +1.39, Phi −0.81 / −0.23 /
+1.14, 0.8B −1.10 / +0.01 / +1.29, 2B −1.10 / +0.11 / +1.07, 9B −1.08 /
−0.10 / +0.69, 27B −0.98 / +0.00 / +0.74. The tables are scan outputs and
do not involve the held-out prompts. Full matrix:
`data/runs/x112/table_comparison.json`.

### r(O) and the O replicate

Table: [r(O) of the selection cell and of the replicate](json/x112-cross-model-vocabulary-table__r-o-and-replicate.json) — per population and the pooled O − replicate per model; the mean over models of O − replicate is −0.022, Qwen3.5-9B −0.073 [−0.113, −0.033] the largest.

A shift of the O level moves every bin of a model by the same amount and
leaves the bin order, top minus bottom and every ρ unchanged;
`results.json` carries Δ against the replicate per bin
(`delta_E_minus_Orep`).

### Hijack (reasoning-disabled label, all E cells)

The word is taken up most at the two ends of the table on most models
(bottom bin 0.17–0.42, top bin 0.02–0.68, middle bins mostly below 0.2);
on Phi it is taken up in the bottom two bins (0.37, 0.20) and almost never
elsewhere (≤ 0.07). Per-word hijack rate against Δ_EO: −0.42 (Phi) to
+0.25 (Qwen3.5-9B). Per-bin rates and `refusal2` given hijack are in
`results.md`.

### The pre-top-up cohort and the whole grid

With the 13 x85 prompts excluded and nothing added (`x112_analyze.py
--no-topup`; `data/runs/x112/results_pre_topup.json`,
`results_pre_topup.md`, `dose_response_pre_topup.png`,
`tables_vs_effect_pre_topup.png`; 26 to 32 prompts per model): seven-model
top − bottom +0.482 [+0.451, +0.514], per model +0.337 [+0.252, +0.424]
(Qwen3.5-27B, 26 prompts) to +0.712 (Qwen3.5-0.8B), 4 to 7 of 7 adjacent
pairs in order (4/7 on Qwen3.5-27B); ρ own +0.54 to +0.96, Qwen +0.59 to
+0.85, own higher on the same five models; discordant closer 64 / 48; the
harmful mean of seven top − bottom +0.398 [+0.349, +0.446], 6/7 in order.
Qwen3.5-27B's ten harmful prompts alone: top − bottom +0.093 [−0.062,
+0.257], own-table ρ +0.36 [−0.12, +0.62] (permutation p 0.023), Qwen
table +0.30 [−0.16, +0.56] (p 0.061).

With the 13 x85 prompts and the Llama word kept (`--full`;
`data/runs/x112/results_full.json`, `results_full.md`,
`dose_response_full.png`, `tables_vs_effect_full.png`; the original 32
prompts per model): seven-model top − bottom +0.492 [+0.461, +0.523], per
model +0.370 (Qwen3.5-27B) to +0.714 (Qwen3.5-0.8B), 5 to 7 of 7 adjacent
pairs in order; ρ own +0.58 to +0.97, Qwen +0.62 to +0.86, own higher on
the same five models; discordant closer 62 / 50; Qwen3.5-27B's 16 original
harmful prompts: top − bottom +0.251 [+0.097, +0.405], own-table ρ +0.62
[+0.37, +0.75]. Phi-4-mini and Qwen3.5-2B have no excluded row and are
identical in the three cohorts; Llama-3.2-3B differs only by the word.

## Artifacts

- Results (x85 rows excluded, harmful stratum topped up):
  `data/runs/x112/results.json` (`cohort`, `exclusions`, `topup` with the
  rule and per model the added ids, fresh rates and pool-stage reuse;
  `models[]` with `excluded` and `topup` where applicable, `r_O`,
  `dose_response`, `rho`, `noise_ceiling_per_cell`, `discordant`,
  `hijack_by_group`, `rate_matrix`; `pooled_over_models`;
  `table_comparison`), `results.md`, `dose_response.png`,
  `tables_vs_effect.png`, `table_comparison.json` (scan tables only, no
  held-out prompt enters it). Pre-top-up (the retained original prompts
  alone, `--no-topup`): `results_pre_topup.json`, `results_pre_topup.md`,
  `dose_response_pre_topup.png`, `tables_vs_effect_pre_topup.png`. The
  original 32 prompts per model, unfiltered (`--full`): `results_full.json`,
  `results_full.md`, `dose_response_full.png`, `tables_vs_effect_full.png`.
- Exclusion: `data/runs/x112/x85_exclusions.json` (per-model prompt ids
  and words, the rule, the x85 cohort file and `cohort_sha256`).
- Inputs: `data/runs/x112/scan_prompts.json`, `candidates.json`.
- Per model, `data/runs/x112/<tag>/`: `scan.json` + `scan_shift.npz`
  (`[8, 71583, 2]`, per-prompt audit, cache layout), `table.json` +
  `table.npz` (own table, percentiles, token counts); `pool_candidates.json`,
  `pool_request.json`, `prompts.json` (32 prompts, pool summaries);
  `words.json` (strata, 40 words, draw log); `key_words_scan.json` +
  `key_words_shift.npz` (`[32, 40, 2]`); `generations/`
  (`generations_pool.json`, `generations_orep.json`,
  `generations_shard{0..31}of32.json`); `judge/refusal2.jsonl`,
  `judge/hijack_fast.jsonl`, `judge/cost_log.jsonl`. On Gemma, Qwen3.5-0.8B,
  9B and 27B also the top-up: `topup_request.json` (the spares, the reused
  and the over-drawn candidates), `prompts_topup.json` (the added prompts,
  their fresh rates and source, provenance), `key_words_scan_topup.json` +
  `key_words_shift_topup.npz` (`[n_added, 40, 2]`), `nshards_topup`,
  `generations/generations_pool_topup.json`, `generations_orep_topup.json`,
  `generations_topup_shard{k}of{n}.json` (n = 2, or 6 on Qwen3.5-27B); the
  judged rows sit in the same `judge/` files.
- Code: `code/x112_common.py`, `x112_select.py` (`--scan-inputs`, `--pool`,
  `--request`, `--prompts`, `--words`, `--compare`, `--topup-request`,
  `--topup-prompts`), `x112_scan_gpu.py` (`--words all|key`, `--prompts
  original|topup`, `--mode cached|full`, `--assemble`; the scan and the
  in-context key-word scan), `x112_generate.py` (`--stage
  pool|orep|grid|pooltopup|oreptopup|gridtopup`), `x112_judge.py`,
  `x112_analyze.py` (default: the exclusion and the top-up; `--no-topup`:
  the exclusion alone; `--full`: the original 32 prompts, unfiltered),
  `x112.sbatch` (`scan`, `keyscan`, `pool`, `orep`, `genloop` and the
  `*topup` counterparts); the cluster tree `<workdir>/x112/`.
- Compute, original run: 30.9 GPU-hours, 4 h wall from first submission to
  the last job (13:31–17:30): scans 2.0 GPU-h, pool O cells 2.0, key
  scans 0.1, grid generation 25.5 (Qwen3.5-27B 17.0 at `--mem=140G`, the
  other six 8.5, on work-stealing workers over 32 resumable shards per
  model), `Orep` 1.3. Judge spend $29.87 (`refusal2` $17.46, hijack
  $12.41). Top-up: 2.3 GPU-hours, 1 h 21 min wall (21:43–23:05,
  fifteen GPU jobs on the GPU partitions): pool O cells 0.14 GPU-h,
  grid generation 2.05 (Qwen3.5-27B 0.88 on three `--mem=140G` workers
  over six shards; the other three models 1.18, of which Qwen3.5-9B's 160
  cells 0.80 on an RTX PRO 6000), `Orep` 0.07, key scans 0.02. Judge spend
  $1.06 (`refusal2` $0.62, hijack $0.44).
