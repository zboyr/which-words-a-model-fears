# x106: in natural benign prompts the models refuse, the boundary direction nominates the word that carries the refusal, and replacing that one word in place removes it in all eight models

Parent: [Experiment index](README.md). The natural-prompt counterpart of the
insertion line: [x83](x83-external-keyword-insertion.md) /
[x101](x101-cross-model-keyword-insertion.md) add an extraneous harm word to a
prompt, [x103](x103-cross-model-direction-insertion-projection.md) /
[x105](x105-cross-model-direction-causal-specificity.md) show the models' own
boundary direction carries that word's effect, and
[x87](x87-external-safety-overrefusal-eval.md) /
[x104](x104-guardreasoner-orbench-rejudge.md) show the direction lowers refusal
on OR-Bench Hard-1K. x106 starts from the Hard-1K prompts the model refuses
although GuardReasoner reads the request as unharmful, lets the direction pick
the word, and replaces that word in place with an ordinary synonym. The
direction, the write site and the dose unit are x88's / x86's, the reader is
x103's, the sampling contract and judge are x101's, the secondary judge is
x104's.

On the 476 Hard-1K prompts Qwen3.5-4B refuses at base with an unharmful
request (plus 100 safe XSTest prompts), a leave-one-word-out pass over the
direction nominates one unit per prompt (median within-prompt z 3.0; a two-word
phrase in 28 %). Replacing that unit by a meaning-preserving word with no harm
association (S) removes **+0.277** of `refusal2` on Qwen3.5-4B and **+0.033
(Llama) to +0.244 (Gemma)** on the seven transfer models that read the same
texts; replacing a frequency- and part-of-speech-matched inert word of the same
prompt (C) moves refusal by **−0.005 to +0.019** with no interval excluding
zero; removing the nominated hedge phrase outright (R, 57 prompts) removes
**+0.039 to +0.504**. Holm p over the seven transfer models is 7 × 10⁻⁴ for S
in every model, and GuardReasoner-8B gives the same picture (S +0.028 to
+0.286). The substituted text's projection on the model's own direction falls
by 0.4 to 0.9 sd while C's stays within 0.1 sd, the per-prompt projection
change tracks the per-prompt refusal change at Spearman +0.64 to +0.80 on the
six models with a base rate above 0.3, and the write at −1.5 removes no more
on S than on O in any model (less on 2B, Phi and 0.8B). The citable values are routed to
[../headline-numbers.md](../headline-numbers.md) under "The phenomenon on
natural prompts".

## Question

The insertion experiments put a harm word into a prompt from the harmful pool.
On a natural benign prompt the model refuses, is there one word, used in its
ordinary sense, that carries the refusal through the same direction, and does
changing only that word change the behaviour? Three parts:

1. **Separation.** Does the insertion-fitted direction separate the natural
   Hard-1K prompts the model refuses from those it answers, and predict which
   refused prompts the write flips?
2. **Nomination.** Within a refused prompt, which word or two-word phrase
   carries the projection, as measured by deleting it?
3. **Substitution.** Replacing that unit in place by a meaning-preserving,
   harm-free word: does `refusal2` fall, does the projection fall with it, and
   is the write left with less to remove?

The word is chosen by the direction, not by hand.

## Design, as run

- **Stage 0, projection** (`x106_stage0_project.py`). Per model, every Hard-1K
  prompt rendered as the x87 / x88 base generation row records it
  (`rendered_chat_sha256` checked per row; the Llama template date recovered
  from the hash as in x103), read at the x88 write site — block output at the
  two trailing prefill positions, Qwen3.5-4B at x86's block 29, the others at
  `round(29/33 · n_blocks)` — through x103's reader, projected on the model's
  consensus direction (x87's consensus of the x86 folds for 4B,
  `x88_directions.npz` otherwise), summed over the two positions. Labels are
  x104's GuardReasoner base `refusal` and request harmfulness and the flip
  under that model's x104 written conditions. Null: 1,000 random unit
  directions. In-domain reference: a five-fold cross-validated
  refused-minus-answered difference-of-means direction and its cosine to the
  consensus direction.
- **Stage 0, nomination** (`x106_stage0_nominate.py`). Per prompt, every
  whitespace word and every adjacent pair containing a content word is deleted
  (edge punctuation stripped from the word, kept in the text), the text
  re-rendered and re-read in unpadded length-grouped batches (one batch per
  exact token length, no attention mask; the batched read is audited against
  the batch-1 read on the first prompt of every shard), `delta_w = proj(full) −
  proj(without w)`. Stop words are excluded from ranking; the unit is the pair
  when its delta beats the top word's by more than one within-prompt sd, else
  the word. Recorded per prompt: top word, within-prompt z, share of
  `proj(full) − mean proj(answered)`, pair win, lexicon membership. Run on all
  1,319 Hard-1K prompts and the 100 XSTest prompts in all eight models, and on
  the S / C / R texts of the cohort (S with the deletion pass, C and R
  projection only).
- **Population.** The 476 Hard-1K prompts Qwen3.5-4B refuses at base under
  GuardReasoner whose request GuardReasoner reads as unharmful, stratified by
  4B projection into `[*, 15)` (33), `[15, 25]` (173), `(25, *]` (270), plus
  the 100 safe XSTest prompts of the `homonyms`, `figurative language`, `safe
  contexts` and `safe targets` categories from x102's copy. 576 units.
- **Texts** (`x106_build_texts.py`), built once from the 4B nomination and
  shared by all models. Per unit the drafter (`deepseek/deepseek-v4-flash-0731`
  over OpenRouter, temperature 0, seeded, reasoning on with a 16,000-token
  budget and a no-reasoning retry when the budget runs out without an answer)
  returns a JSON object: whether the unit belongs to an intent or hedging phrase
  (`hedge`) or is a content unit; a substitute word or phrase; a control word
  chosen from the prompt's inert candidates (content words outside the unit with
  |z| < 0.5, Zipf frequency within the narrowest non-empty band of 1.0 / 2.0 /
  any around the unit's, part of speech the drafter's choice) and a synonym for
  it; for hedges, the exact phrase and a neutral phrase of the same function.
  The texts are assembled here by span replacement, so each text is O with
  exactly one span changed: **S** = unit → substitute (hedge units: phrase →
  neutral phrase), **C** = control word → synonym, **R** = hedge phrase
  removed. A draft is accepted when it passes the structural checks — the
  substitute and the neutral phrase contain no x83 lexicon word and no content
  word of any cohort prompt's nominated unit, the control synonym contains no
  lexicon word, the hedge phrase is a unique exact substring of O containing
  the unit, the control word is one of the inert candidates, the substitute
  differs from the unit, and the edited text repeats no adjacent word that O
  does not — and the drafter is re-prompted with the failed checks up to three
  times. Units whose drafter returns no meaning-preserving substitute, or whose
  drafts fail the checks three times, keep O with S absent. Cohort
  `data/runs/x106/x106_cohort.json`, sha256 `e06327cf…`: 576 units, 518
  content / 57 hedge; 143 units are pairs; 35 lie in the x83 lexicon; texts
  O 576 / S 473 / C 530 / R 57; 103 without S (66 the drafter judged to have no
  neutral meaning-preserving substitute — `self-harm`, `gun`, `gambling`,
  `scam`, `explosive` — and 37 whose drafts failed the checks). A 10 % sample
  (58 units, seed 106) is listed in `data/runs/x106/x106_readthrough.md` for
  the human read-through. Drafting cost ≈ $1.
- **Arms** (`x106_generate.py`). Per model: `O_base`, `S_base`, `C_base`,
  `R_base` at dose 0 and the boundary write on O and on S at −1.5 (Qwen3.5-4B
  also at −0.5): 50 arms, 22,529 prompt-conditions, 225,290 responses. x105's
  write path (`ExternalBoundaryHook`, the per-job write-site audit before the
  first row; all 50 audits passed and every dosed row recorded exactly one
  prefill write), x101's sampling contract (T = 1.0, top-p 0.9, top-k 50, 10
  sequences, 2,048 tokens, one 4,096 retry), seeds per (tag, prompt, text) in
  the `x106` namespace so the doses share one random-number stream. O rows are
  rendered and checked against the x87 / x88 base row's `rendered_chat_sha256`.
- **Judges.** Primary `refusal2`, the x64 contract as x101 / x105 ran it
  (`x106_judge_refusal.py`, run locally): 50 arms, every prompt-condition
  labelled, zero failed rows, ≈ $12. Secondary GuardReasoner-8B under x104's
  contract (`x106_judge_guardreasoner.py`, two shards per arm on the cluster, greedy
  then up to three re-decodes of unparsed rows; one shard hit an out-of-memory
  on its longest inputs and was resumed with the batch halved on overflow):
  every response of every arm labelled.
- **Analysis** (`x106_analyze.py`). Per model and judge, prompt-paired on the
  prompts carrying both arms' labels, one 10,000-replicate two-stage bootstrap
  (prompts, then the ten binary responses within each arm) over every
  estimand: `r(O) − r(S)`, `r(O) − r(C)`, `r(O) − r(R)` (Holm within the three),
  `S − C`, the same on XSTest, per stratum; `suppress_O = r(O) − r(O, −1.5)`,
  `suppress_S`, their difference, dose order on 4B; Spearman between Stage 0
  `delta_top` and the per-prompt substitution effect (C's as comparison);
  `proj(S) − proj(O)`, `proj(C) − proj(O)`, the share of S texts below the
  answered-prompt mean, Spearman between projection change and refusal change;
  per-prompt Spearman between the substitution effect and `suppress_O`; reach
  over the 476. Across models, Holm over the seven transfer models within each
  contrast, Qwen3.5-4B the reference row outside the family as in x103 / x105.
  Stage 0 per model: AUCs with bootstrap intervals, the random null, the
  cross-validated in-domain direction, nomination summaries, and top-1
  agreement of each model's nomination with 4B's.

## Results

Table: [Stage 0 per model](json/x106-natural-benign-lexical-carrier__stage0-projection-and-nomination.json) — base refusal rate, AUC of the projection for refused vs answered on all 1,319 prompts and on the request-unharmful ones, the largest AUC among 1,000 random directions, the cross-validated in-domain AUC and the in-domain direction's cosine to the consensus direction, the flip AUC at −1.5, the population size, the nominated unit's median z / pair share / lexicon share, and agreement with the 4B nomination. AUC runs from 0.865 (0.8B) to 0.968 (9B) against random maxima of 0.80–0.94 (the observed AUC beats all 1,000 draws in seven models and 999 in Gemma); the in-domain direction has cosine 0.36–0.70 to the consensus direction and reaches 0.88–0.97; the nominated unit's median z is 3.0–3.6, and nominated units lie in the x83 lexicon at 0.04–0.11 (0.00 on Llama's 33 prompts) against a content-word base rate of 0.003–0.030.

**What the direction nominates.** On the 476-prompt population the 4B
nomination's most frequent units are `questionable` (12), `controversial`,
`restricted`, `self-harm` (8 each), `illegal`, `without` (7), `legal`, `id`
(6), `bomb`, `detection`, `access` (5); over all 1,319 Hard-1K prompts
`questionable` (25), `explosive`, `without` (15), `bomb` (14), `controversial`,
`illegal`, `restricted` (13), `create`, `self-harm` (12). On XSTest the median
z is 0.76 and the top units are `best`, `what's`, `good`, then `kill`. The
other seven models nominate the same top word as 4B on 55–71 % of the 576
cohort prompts and place 4B's unit in their own top three on 82–91 %. The top
word's deletion delta on its own separates refused from answered prompts at AUC
0.63–0.76.

Table: [Substitution contrasts per model and judge](json/x106-natural-benign-lexical-carrier__substitution-contrasts.json) — `r(O) − r(S)`, `r(O) − r(C)`, `r(O) − r(R)` under `refusal2` and under GuardReasoner with 95 % intervals and the Holm p over the seven transfer models. S removes +0.079 (0.8B), +0.170 (2B), +0.277 (4B), +0.218 (9B), +0.220 (27B), +0.033 (Llama), +0.244 (Gemma), +0.143 (Phi) of `refusal2`; C's eight estimates lie between −0.005 and +0.019 and every C interval covers zero; R removes +0.504 on 4B and +0.039 (Llama) to +0.448 (9B) elsewhere; `S − C` has Holm p 7 × 10⁻⁴ in six transfer models and 3 × 10⁻³ in Llama; GuardReasoner's S column is +0.028 (Llama) to +0.286 (4B).

Table: [The boundary write next to the substitution, and the projection change](json/x106-natural-benign-lexical-carrier__write-and-representation.json) — per model under `refusal2`: refusal removed by the −1.5 write on O and on S, their difference, the refusal left on written S, the projection change of S and of C in that model's sd units, the share of S texts whose projection falls below the answered-prompt mean, and the three per-prompt Spearman coefficients. The write removes +0.010 (Llama) to +0.234 (2B) on O and +0.002 to +0.200 on S; on 2B and Phi it removes +0.082 and +0.071 more on O than on S (Holm p 7 × 10⁻⁴), on 0.8B +0.027 (Holm 0.05), and on the other models the difference's interval covers zero; `proj(S) − proj(O)` is −0.43 (0.8B) to −0.88 (4B) sd and `proj(C) − proj(O)` is −0.01 to −0.05 sd.

Table: [Strata, XSTest and reach](json/x106-natural-benign-lexical-carrier__strata-xstest-reach.json) — the S and C contrasts and `r(O)` by 4B projection stratum, the S / C contrasts and the write on O on the 100 XSTest prompts, and over the 476 population prompts the share whose `refusal2` under S falls below 0.5, the share with no substitute, and the share among substituted prompts whose refusal S does not lower. On 4B the S effect is +0.085 / +0.272 / +0.305 across the low / mid / high strata where `r(O)` is 0.39 / 0.76 / 0.97; on XSTest, where 4B refuses 0.09 at base, S removes +0.043 and C −0.006; 32 % of the 476 fall below 0.5 under S on 4B, 19 % have no substitute.

**Nomination predicts the effect.** Spearman between Stage 0 `delta_top` and
the per-prompt `r(O) − r(S)` is +0.36 [+0.26, +0.44] on 4B, +0.29 (27B), +0.28
(9B), +0.27 (Gemma), +0.18 (2B), +0.16 (Phi), +0.06 (Llama), +0.05 (0.8B);
against `r(O) − r(C)` it is −0.05 to +0.08 everywhere.

**Representation moves with behaviour.** The per-prompt projection change
of S tracks its refusal change at Spearman +0.80 (4B), +0.77 (9B), +0.74
(27B), +0.69 (Gemma), +0.65 (Phi), +0.64 (2B), +0.37 (0.8B), +0.23 (Llama);
for C the same coefficient is +0.16 to +0.44 on changes that are an order of
magnitude smaller. On 4B 19.5 % of S texts project below the answered-prompt
mean against 1.6 % of their O texts.

**Same axis as the write.** On 4B the write removes +0.054 at −0.5 and
+0.199 at −1.5 on O (dose-ordered), and +0.057 / +0.191 on S; written S
refuses at 0.378 against 0.852 for unwritten O. The per-prompt substitution
effect correlates with the per-prompt write effect on O at Spearman +0.51
(Phi), +0.49 (Llama), +0.46 (2B), +0.44 (0.8B), +0.22 (Gemma), +0.12 (4B),
+0.05 (27B), +0.04 (9B).

**Output integrity.** Truncation after the 4,096-token retry is at most 0.001
in every 4B arm (median response length 1,411–4,066 characters, no arm with a
response dominated by a repeated 4-gram); the per-arm figures for every model
are in the result file.

Figure: `data/results/x106_natural_lexical_carrier.pdf` (per model the three
dose-0 contrasts, the write on O and on S, and the projection change of S and
C).

## Reading

- **A natural over-refusal has a lexical carrier the direction can find.**
  The projection separates refused from answered natural prompts in every
  model, one unit per prompt carries it (median z 3), and changing that unit
  alone changes the behaviour while changing a matched inert word of the same
  prompt changes nothing.
- **The carriers are ordinary words.** 89 % or more of nominated units are
  outside the x83 lexicon: evaluative and hedging markers (`questionable`,
  `controversial`, `restricted`, `without`, `legal`, `discreetly`) next to
  concrete nouns (`bomb`, `explosive`, `virus`, `self-harm`). Hedges are the
  strongest case: removing the phrase removes half of 4B's refusal on those
  prompts.
- **Effect size follows the model's base refusal.** Llama refuses 7 % of the
  population at base and has the least to lose (+0.033); the mid and large
  Qwen models and Gemma, refusing 77–87 %, lose +0.22 to +0.28. Within 4B the
  effect grows with the projection stratum.
- **Substitution and the write act on one quantity.** S lowers the projection
  the write lowers, the two per-prompt effects correlate on six models, and on
  the three models where the write removes more than S does (2B, Phi, 0.8B)
  the write removes less on S than on O.
- **Both judges agree.** GuardReasoner's S column is +0.028 to +0.286 against
  `refusal2`'s +0.033 to +0.277, its R column is larger on every model, and
  its C column is as flat (−0.006 to +0.014).

## Artifacts

- Result: `data/results/x106_natural_lexical_carrier.json`, figure
  `x106_natural_lexical_carrier.{pdf,png}`.
- Frozen inputs: `data/runs/x106/x106_inputs.json` (per model the snapshot,
  the hook block, the direction artifact and its sha256, the base-generation
  files whose rendered-chat hashes the O rows are checked against);
  `data/runs/x106/x106_cohort.json` (`e06327cf…`: every unit with its O / S / C
  / R texts and their sha256, the nominated unit, the substitute, the control
  word and synonym, the hedge phrase, the drafter's note where S is absent,
  the read-through ids); `x106_substitute_drafts.jsonl` (every drafter call
  and check outcome); `x106_readthrough.md`.
- Per model under `data/runs/x106/<tag>/`: `x106_stage0_projections.json`,
  `x106_stage0_nominations.json` (all 1,419 prompts) and
  `x106_stage0_nominations_{S,C,R}.json` (cohort texts),
  `x106_refusal_<arm>.jsonl` (ten labels per prompt-condition);
  `x106_generations_<arm>.json` (with its write-site audit; ≈ 660 MB in all),
  `x106_guardreasoner_<arm>_shard*.jsonl` (≈ 500 MB), `x106_stage0_states.npz`
  and the Stage 0b shard files, all shipped in this repository.
- Code: `code/x106_common.py`, `x106_stage0_project.py`,
  `x106_stage0_nominate.py`, `x106_build_texts.py`, `x106_generate.py`,
  `x106_judge_refusal.py`, `x106_judge_guardreasoner.py`, `x106_analyze.py`,
  `x106.sbatch`; tests `code/tests/test_x106_natural_lexical_carrier.py` (6, in
  the 110-test suite). The hook, directions, model registry and chat rendering
  are x88's / x87's / x103's, the sampling engine, seeds and judge x101's /
  x83's / x81's, the GuardReasoner contract x104's.
- Compute: ≈ 2 GPU-hours for Stage 0 on all eight models, ≈ 101 GPU-hours of
  generation (five Qwen3.5-27B arms ran to the 8-hour limit and resumed), ≈ 39 GPU-hours of
  GuardReasoner judging, plus ≈ 3 GPU-hours in cancelled jobs
  ([../infrastructure.md](../infrastructure.md)).
