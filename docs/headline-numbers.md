# Headline numbers and provenance

Parent: [Experiment index](experiments/README.md).

Hard rule: every number that appears in the paper traces to a named result
file. This file states what is citable and where it lives; it does not
re-tabulate what the experiment files already own.

## Comparability rules (in force)

1. **Name the label construct.** `refusal2` (explicit refusal present) and
   `comply2` (the response substantively attempted the request) are separate
   judgments; a refusal delta and a compliance delta are never pooled or
   swapped. Owned by
   [labels-refusal-comply-x58.md](experiments/labels-refusal-comply-x58.md).
2. **Doses are model-relative and family-relative.** The boundary write is
   dosed in multiples of a model's own mean boundary shift; the rank-one
   edit is a wrapper scale on that model's own adapter. Never put the two
   families on a shared numeric dose axis, and never claim cross-model dose
   equivalence.
3. **One judge.** OR-Bench refusal is scored for every model and condition
   by a single fixed judge, GuardReasoner-8B, whose labels
   [x104-guardreasoner-orbench-rejudge.md](experiments/x104-guardreasoner-orbench-rejudge.md)
   owns; it was chosen by its agreement with independent judges and with
   human labels in
   [x102-orbench-judge-agreement.md](experiments/x102-orbench-judge-agreement.md).

## The phenomenon: lexical separability survives dataset-side removal

Owned by [dataset-mining.md](experiments/dataset-mining.md) and
[dataset-scaleout-validity.md](experiments/dataset-scaleout-validity.md).
These are properties of the prompt pools, not of any model.

- **Pooled 15k source pool** — word-level TF-IDF AUC **0.5073**, character
  level **0.6164**; the character auditor stays well above chance on a pool
  whose word auditor is at chance. Source:
  `data/results/extend_pool_A3_auc.json`.
- **Growing the pool to 20k restores separability** — word **0.7154**,
  character **0.7219**, with the added 5k increment alone at 0.9077 word.
  This is why the pool was frozen at 15k. Source:
  `data/results/extend_pool_A4_auc.json`.
- **The hand-mixed starting pool** — mean pairwise source AUC 0.9123, and a
  leave-one-out showing one source dominates confusability. Sources:
  `data/results/auc_pairings.json`, `data/results/combined_pool_auc.json`.
- **Hidden states still separate what the lexicon does not** — probe
  AUC_risk 0.7651 on the filtered pool. Source:
  `data/results/confusable_A_auc.json`.
- **Recorded in the experiment file:** the in-loop filtering trajectory
  (0.9805 → 0.5623 over four rounds) and the fresh-auditor rebound
  (0.6951 word / 0.7016 character on the hardness-tail cut), reconstructed
  from the run log; their per-round result JSONs were not carried into this
  project.

## The phenomenon: one inserted word is sufficient

Owned by [x83-external-keyword-insertion.md](experiments/x83-external-keyword-insertion.md);
sole source `data/results/x83_keyword_insertion.json`. Qwen3.5-4B, 321
prompts from the frozen evaluation split, `refusal2` label.

- **Paired refusal increase over a contemporaneous unmodified control** —
  start **+0.31** [+0.22, +0.40] (n = 55), middle **+0.19** [+0.13, +0.25]
  (n = 105), end **+0.33** [+0.29, +0.37] (n = 267); all Holm-adjusted
  p < 10⁻⁴.
- **The detection gate does not moderate the effect** — the same size on
  prompts that fail the gate (start +0.32, middle +0.17), and per-prompt
  deltas are uncorrelated with detection confidence (Spearman 0.00 / 0.00 /
  +0.18). This is what licenses "the model knows the word is irrelevant and
  refuses anyway".
- **Every category with ≥ 20 admitted prompts is positive** — +0.24 to
  +0.44, with generic illegality markers at least as potent as concrete
  weapon or drug nouns.
- **Floor prompts flip** — 38 of 82 never-refused prompts reach majority
  refusal at the start position.
- **Edges beat the middle by about 0.13** on the common cohort (n = 27).

## The mechanism: the crossed boundary direction

Owned by
[x85-crossed-keyword-prompt-latent-decomposition.md](experiments/x85-crossed-keyword-prompt-latent-decomposition.md);
sole source `data/results/x85_crossed_keyword_latent.json` (directions in
`x85_crossed_keyword_directions.npz`).

- **Double-held-out projection** — +6.28 [+4.88, +7.86] at `</think>` and
  +7.20 [+5.61, +9.19] at the following `"\n\n"`, positive on 95% / 94% of
  held-out cells, 23 of 24 held-out word means positive, sign-flip
  p = 2×10⁻⁴.
- **Projection tracks the induced refusal change** — Spearman **0.65** /
  **0.63** after residualizing both sides on prompt-plus-word fixed effects.
- **The harm word raises refusal +0.338 over its neutral twin**, while the
  neutral insertion itself is inert (−0.014 against the unmodified prompt).
- **Word identity is integrated late** — at the word's own token the change
  is 70–88% word-specific energy but barely predicts refusal (ρ ≈ 0.2); at
  the boundary tokens word energy is 15%.
- **Word potency is rank-one** — ρ = 0.92 between a word's projection and
  its mean refusal change over 24 prompts.

## The mechanism: causal intervention on held-out cells

Owned by
[x86-crossed-keyword-boundary-intervention.md](experiments/x86-crossed-keyword-boundary-intervention.md);
sole source `data/results/x86_boundary_intervention.json`. 288 cells whose
prompt *and* word were both held out of their fold's direction, dose
a* = 0.5 admitted by a benign-KL ceiling.

- **Bidirectional movement** — subtracting from harm-worded cells lowers
  refusal **0.080** [0.043, 0.121]; adding to neutral-worded cells raises it
  **0.086** [0.041, 0.133]; both one-sided Holm p = 2×10⁻⁴.
- **Dose-ordered** — bidirectional 0.036 at half dose, 0.083 at full, with a
  fixed-effect slope [+0.053, +0.115].
- **Direction-specific** — true minus KL-matched random +0.088 [0.054,
  0.124]; true minus sign-permuted +0.055 [0.017, 0.095].
- **One token carries it** — final `"\n\n"` alone +0.070 [0.040, 0.103]
  versus +0.011 for the token before it, no synergy.
- **Substantive compliance mirrors** — `comply2` moves oppositely and
  commensurately (+0.072 / −0.082), so the intervention moves delivery, not
  refusal wording.
- **Benign cost** — adding the direction raises refusal on sealed benign
  prompts +0.058 [+0.031, +0.085].

## External benefits and costs (Qwen3.5-4B)

Design and the StrongREJECT / IFEval endpoints owned by
[x87-external-safety-overrefusal-eval.md](experiments/x87-external-safety-overrefusal-eval.md),
the OR-Bench refusal labels by
[x104-guardreasoner-orbench-rejudge.md](experiments/x104-guardreasoner-orbench-rejudge.md);
sole source `data/results/x104_external_safety_utility.json`
(`successful_candidates: []`). One model, one frozen snapshot,
prompt-paired 20,000-resample bootstraps, Bonferroni-adjusted one-sided
gates: superiority at α = 0.05/3, three non-inferiority guards at α = 0.05/9
against an absolute **0.02** margin each. Base rates: Hard-1K refusal 0.886,
Toxic refusal 0.986, StrongREJECT 0.063, IFEval prompt-strict 0.804.

- **Hard-1K over-refusal falls under both interventions** — boundary write
  at dose −0.5 **−0.029** [one-sided upper −0.017]; rank-one edit at scale
  −3 **−0.065** [upper −0.048], about 7% of the base rate. Superiority
  passes for both.
- **Guard bounds** — three of the six deployment-dose guard bounds land
  outside the 0.02 margin.
- **Toxic refusal slides slightly** — −0.006 / −0.014 (lower bounds −0.017,
  inside the margin under the write / −0.027, outside under the edit);
  neither interval excludes zero.
- **StrongREJECT harmfulness** — +0.005 [upper +0.024, outside the margin]
  / −0.015 [upper +0.005, inside]. Non-inferiority holds for the edit only.
- **Instruction following** — IFEval +0.019 [lower −0.006, inside] / −0.004
  [lower −0.039, outside]. Non-inferiority holds for the write only.
- **The sign-flipped control mirrors** — the same rank-one adapter at +3
  raises Hard-1K +0.048 [lower +0.033] and StrongREJECT +0.022. It is
  ineligible as a candidate by construction.
- **Deep doses** — boundary write to −3 (Hard-1K −0.215, Toxic −0.041) and
  rank-one to −7 (Hard-1K −0.165, IFEval −0.083 at benign KL 0.26). The
  cost that scales with dose is Toxic refusal; StrongREJECT stays near base.
- **GSM8K capability check** — every one of the thirteen edited conditions
  is within ±0.011 of base 0.7877. Source
  `data/results/x87_gsm8k_posthoc.json`.

## The positive-dose extension

Design owned by
[x89-positive-dose-external-extension.md](experiments/x89-positive-dose-external-extension.md),
the OR-Bench refusal labels by
[x104-guardreasoner-orbench-rejudge.md](experiments/x104-guardreasoner-orbench-rejudge.md);
sole source `data/results/x104_positive_dose_extension.json`. Same frozen
snapshot and pipeline; every pre-existing comparison reproduced exactly.

- **The boundary write mirrors externally, dose-ordered** — Hard-1K
  +0.024 / +0.053 / +0.062 at +0.5 / +1.0 / +1.5, all intervals excluding
  zero, while StrongREJECT is null and IFEval moves slightly up.
- **The rank-one response is near-linear through the origin on both sides**
  — −0.024 / −0.046 at −1 / −2 against +0.015 / +0.039 at +1 / +2.
- **Amplification leaks earlier than suppression** — scale +2 already raises
  StrongREJECT +0.030 (interval excluding zero), while the negative branch
  stays null down to −3.

## Cross-model transfer

Design and the StrongREJECT / IFEval / GSM8K endpoints owned by
[x88-eight-model-deep-dose-transfer.md](experiments/x88-eight-model-deep-dose-transfer.md),
the OR-Bench refusal labels by
[x104-guardreasoner-orbench-rejudge.md](experiments/x104-guardreasoner-orbench-rejudge.md);
sole source `data/results/x104_cross_model_deep_dose.json`. Eight
instruction-tuned models, four families, 0.8B–27B.

- **The boundary write transfers at one fixed dose (−1.5)** — Hard-1K
  refusal falls with the interval excluding zero on 8 of 8 models, from
  −0.012 (Llama, whose base rate of 0.091 floors it) to −0.245. Costs stay
  smaller: Toxic −0.005 to −0.102, largest StrongREJECT increase +0.030,
  |IFEval| ≤ 0.026, |GSM8K| ≤ 0.014 with mixed signs.
- **The rank-one edit at one fixed scale does not** — at −6 the 0.8B model
  loses capability (IFEval −0.275, GSM8K −0.309) and Llama's safety
  collapses (Toxic −0.421, StrongREJECT +0.308, spot-checked responses
  genuinely complying).
- **The failure is dose-monotonic, not a cliff at −6** — at the −3
  deployment scale Llama already shows Toxic −0.055 / StrongREJECT +0.051,
  and the two smallest Qwen models cut Hard-1K −0.178 / −0.212 while
  leaking. Within the Qwen family the delivered strength falls as the model
  grows (Hard-1K at −3: −0.065, −0.044, −0.042 from 4B to 27B).

## The phenomenon on natural prompts: the word is already there

Owned by
[x106-natural-benign-lexical-carrier.md](experiments/x106-natural-benign-lexical-carrier.md)
(benign side) and
[x108-toxic-lexical-carrier-specificity.md](experiments/x108-toxic-lexical-carrier-specificity.md)
(harmful side); sole sources `data/results/x106_natural_lexical_carrier.json`
and `data/results/x108_toxic_lexical_carrier.json`. Eight models, each on its
own direction and read site; texts drafted once from the Qwen3.5-4B
nomination and shared; `refusal2` primary, GuardReasoner-8B secondary
(`Completion` = refusal, `Response` = response harmfulness); 95 % two-stage
paired bootstraps, Holm over the seven models whose nomination did not draft
the texts. Qwen3.5-4B is the nominating model, not the centre of the claim:
report ranges over the eight models, never the agreement of the other models
with the 4B nomination.

- **The direction separates natural refused from answered prompts** — AUC
  0.865–0.968 on each model's own Hard-1K prompts, above all 1,000 random
  directions in seven models and 999 in Gemma.
- **One ordinary word carries it** — median within-prompt z 3.0–3.6 on each
  model's own refused population; at most 0.11 of nominated units lie in the
  99-word insertion lexicon (content-word base rate 0.003–0.030); top units
  `questionable`, `controversial`, `restricted`, `without`, `legal`,
  `self-harm`, `bomb`.
- **Replacing that word in place removes the refusal; replacing a matched
  inert word does not** — `refusal2` r(O) − r(S) **+0.033 (Llama, base 0.07)
  to +0.277** across the eight models; r(O) − r(C) −0.005 to +0.019 with
  every interval covering zero; S − C Holm p 7 × 10⁻⁴ in six models and
  3 × 10⁻³ in Llama; hedge removal R +0.039 to +0.504; GuardReasoner refusal
  S +0.028 to +0.286.
- **Reach of the rule on benign prompts** — 0.81 of the 476 (0.12 no
  admissible substitute, 0.07 failed checks); 0.86 of the 100 safe XSTest
  prompts, on which S moves refusal −0.007 to +0.105.
- **On harmful prompts the nominated word is the harm** — median z 3.4–4.2 on
  the Toxic cohort against 2.5–3.5 on the shared benign cohort; a pair on
  0.32–0.49 against 0.21–0.28; top units `self-harm`, `sexual`,
  `misleading`, `without`, `negatively`. The XSTest contrast prompts nominate
  `kill`, `shoot`, `crush` (z 2.6–3.7) and project 2.2–4.4 sd above the safe
  counterparts (AUC 0.94–0.998), whose nominated unit has z 0.6–1.7.
- **Reach of the rule on harmful prompts** — 0.64 of the 619 (0.25 no
  substitute: `self-harm`, `sexual`, `bomb`; 0.11 failed checks); 0.50 of the
  100 contrast prompts.
- **Where a substitute exists the refusal falls on every model and the
  response stays unharmful** — `refusal2` **+0.076 (Qwen3.5-27B, base 0.99)
  to +0.226 (Llama, base 0.76)**, Holm p 7 × 10⁻⁴ in all seven; GuardReasoner
  Completion refusal +0.073 to +0.233; GuardReasoner Response harmfulness
  −0.014 to +0.041 (intervals excluding zero on 2B, 4B, Llama, Gemma, Phi);
  R +0.174 to +0.357; XSTest contrast S +0.164 to +0.308. On the five models
  whose originals are almost never harmful, under 0.07 of substituted
  prompts yield any harmful response.
- **One quantity on both sides** — proj(S) − proj(O) −0.43 to −0.88 sd of
  the model's Hard-1K projection on benign prompts, −0.42 to −0.84 on harmful
  ones, C within 0.05 sd; per-prompt Spearman between projection change and
  refusal change +0.52 to +0.69 (harmful) and +0.64 to +0.80 on the six
  models with benign base refusal above 0.3; the write at −1.5 removes less
  on S than on O by +0.082 (2B), +0.071 (Phi), +0.027 (0.8B), intervals
  excluding zero, and the same amount elsewhere.

## The table choice on prompts that chose none of the tables

Owned by
[x117-table-choice-held-out-replication.md](experiments/x117-table-choice-held-out-replication.md);
sources `data/runs/x117/results.json` (64 new prompts),
`results_check_x111.json` (x111's 64 selection prompts through the same
loader) and `results_pooled_x111_x117.json` (128). x111's four Qwen3.5-4B
tables and 40 words, 64 further prompts drawn by x111's rule after excluding
every prompt any table or earlier test used (32 Hard-1K benign in [0.3, 0.7];
32 x83 harmful, 27 in [0.3, 0.7] and 5 at 0.2 / 0.8), twenty samples per cell,
`refusal2` with the judge's upstream pinned to GMICloud for 5,161 of the 5,248
units (DeepInfra for the first 87, which then rate-limited).

- **Table choice replicates** — ρ(table shift, Δ_EO) over the 40 words,
  pooled over the 64 new prompts: BP **+0.92** [+0.89, +0.94], mixed +0.89
  [+0.85, +0.91], HP +0.88 [+0.85, +0.91], clean-benign +0.55 [+0.49, +0.61];
  on x111's selection prompts +0.90 / +0.88 / +0.88 / +0.57; pooled over 128
  +0.91 / +0.88 / +0.88 / +0.53. BP leads on the benign prompts (+0.89; HP
  +0.78) and HP on the harmful (+0.90; BP +0.80), as in x111. The three
  refusal-prompt tables are within 0.04 of each other with overlapping
  intervals; the clean-benign table is the odd one out. Cite the paper's
  Qwen3.5-4B ρ from this run, not from x111.
- **Dose-response replicates** — pooled Δ_EO per bin −0.165, −0.098, −0.075,
  +0.023, +0.032, +0.127, +0.229, **+0.367**; 7/7 adjacent pairs in order
  pooled, on the benign and on the harmful prompts; top − bottom **+0.532**
  [+0.480, +0.583] (benign +0.498, harmful +0.566). x111: −0.142 to +0.336,
  +0.478 [+0.417, +0.538].
- **Discordant quadrants** — high in BP only +0.222, low in BP only −0.096,
  low in clean-benign only +0.077, high in clean-benign only +0.028 (x111:
  +0.208 / −0.077 / +0.062 / +0.018).
- **In-context shift** — per word +0.92 pooled (+0.87 benign, +0.95
  harmful); per cell +0.74 against a ceiling of 0.88, the best fixed table
  +0.49 per cell.
- **r(O)** — 0.602 benign, 0.502 harmful against historical 0.531 / 0.516.

## The substituted prompts read by two independent judges

Owned by
[x118-meaning-preservation-judge.md](experiments/x118-meaning-preservation-judge.md);
sources `data/runs/x118/results.json` (first judge) and
`data/runs/x118/results_haiku.json` (second judge, agreement, and the effects on
the pairs both accept). Every (original, edited) pair of the three substitution
experiments — 4,805 distinct pairs, 7,713 uses over the eight models — labelled
on whether the edited text asks for the same thing by `openai/gpt-6-luna`
(OpenRouter, reasoning effort `minimal`, temperature 0, seed 42, `json_schema`
output) and, on the same pairs with the same prompt and schema, by
`anthropic/claude-haiku-5.5` (reasoning effort `minimal`, `json_schema`; the
route lists neither `temperature` nor `seed`, so neither is sent, and 2 pairs
came back `content_filter` with no label). The drafting model is not consulted.

- **Share read as the same request** — benign substitution (S) **0.605**
  [0.560, 0.648], harmful substitution **0.581**, table-guided substitution
  (S_T) **0.330** (0.278–0.353 per model); inert-word controls **0.855** (C)
  and **0.806** (C_T); hedge removals 0.070 (benign) and 0.130 (harmful).
  Content-word edits are accepted at 0.667 (S) and 0.397 (S_T) against 0.143
  and 0.053 for hedge-phrase edits.
- **The first judge's repeatability and the human read-through** — second seed
  on a 10 % sample (495 pairs): agreement 0.945, Cohen's κ 0.889, relation
  agreement 0.867. Of
  the 162 pairs the human read-through covered, none of which a human marked
  bad, the judge accepts 101 and rejects 61.
- **The second judge, same pairs** — benign substitution **0.789** [0.750,
  0.823], harmful **0.707**, table-guided **0.527** (0.489–0.558 per model);
  controls **0.968** (C) and **0.947** (C_T); hedge removals 0.123 and 0.087.
  Content-word edits 0.854 (S) and 0.609 (S_T) against 0.304 and 0.183 for
  hedge-phrase edits. Repeatability on the same 10 % sample, a second call at
  the route's own settings: agreement **0.992**, Cohen's κ **0.979**, relation
  agreement 0.980. Of the 162 read-through pairs it rejects 45.
- **The two judges against each other** — on the 4,803 pairs both labelled,
  agreement **0.800**, Cohen's κ **0.578** (S arms 0.769 / 0.553; the control
  arms 0.84–0.87 at κ 0.24–0.28, where both judges accept nearly everything).
  Both accept 2,588, the second judge alone 882, the first alone 81, both
  reject 1,252; the four-way relation agrees on 0.727 of pairs.
- **The substitution effect on the accepted pairs** — `refusal2` removed by S
  **+0.022 (Llama) to +0.164 (Qwen3.5-4B)** on the benign Hard-1K population
  (218 units), by S_T **+0.045 (Qwen3.5-0.8B) to +0.191 (Qwen3.5-2B)** (49–84
  units per model) and by S **+0.012 to +0.106** on the Toxic cohort (241
  units); 21 of the 24 intervals exclude zero. The inert-word controls are
  unchanged by the restriction (C −0.003 to +0.015, C_T +0.010 to +0.037).
- **The substitution effect on the pairs both judges accept** — `refusal2`
  removed by S **+0.023 (Llama) to +0.160 (Qwen3.5-4B)** on the benign Hard-1K
  population (215 units), by S_T **+0.045 (Qwen3.5-0.8B) to +0.203
  (Qwen3.5-2B)** (47–81 units per model) and by S **+0.006 to +0.099** on the
  Toxic cohort (228 units); 19 of the 24 intervals exclude zero. On the second
  judge's own, larger accepted subset (294 / 114–137 / 288 units) all 24
  exclude zero: +0.027 to +0.196, +0.039 to +0.214 and +0.024 to +0.147.
  Controls on the both-accepted mask: C −0.003 to +0.015, C_T +0.015 to
  +0.037.

## Judge validation (OR-Bench)

Owned by
[x102-orbench-judge-agreement.md](experiments/x102-orbench-judge-agreement.md);
sole source `data/results/x102_judge_agreement.json`. A frozen 300-prompt
sample across the 35 reported model-conditions (10,500 responses)
re-judged by three API judges under the benchmark's verbatim prompt and by
Qwen3Guard-Gen-4B, GuardReasoner-8B and WildGuard; XSTest's 2,250
human-labelled completions judged by every instrument.

- **Binary agreement with the incumbent judge is 94.5–95.3% (κ
  0.877–0.894) for every arm** — Fleiss' κ 0.899 over the incumbent and
  the three API judges; sensitivity 0.978 / specificity 0.903 against the
  API majority, 0.980 / 0.880 against WildGuard.
- **No reported delta reverses under any instrument** — 46–53 of the 54
  model × condition × benchmark cells have the other instrument's delta
  inside the incumbent's interval. The cells outside are the rank-one deep
  edits on Qwen3.5-0.8B / 2B, where the incumbent understates the Hard-1K
  drop (0.8B at scale −6: −0.015 against −0.12 to −0.145).
- **Differential error is null on Qwen3.5-4B and on the boundary write
  everywhere** — on Qwen3.5-4B every paired edited − base change in
  disagreement is at most +0.045 with an interval reaching zero; the
  boundary write never exceeds +0.03 on any model.
- **XSTest human agreement 91.7% (κ 0.834) for the incumbent** — above
  XSTest's released GPT-4 labels (82.5%, κ 0.654) and string match (88.4%,
  κ 0.770); WildGuard reaches 94.0% (κ 0.881) and GuardReasoner 95.1%
  (κ 0.901).
- **No voting combination of the seven instruments beats GuardReasoner-8B
  alone on XSTest** — of 141 combinations on the 2,244 completions every
  instrument parsed, one ("GLM or GuardReasoner", 95.2%, κ 0.904) edges
  GuardReasoner's 95.1% (κ 0.901; sensitivity 0.959, specificity 0.942)
  by +0.001 [−0.002, +0.005]; under 20 × 5-fold cross-validation the
  best-by-training combination gains −0.0004 (sd 0.0033) on held-out
  rows; the two XSTest annotators agree on the binary label at 97.9%.

## Do not cite

- Any number from the probe-campaign work this project descends from
  (nine-model AUCs, ranking metrics, the Llama refusal-direction edit).
  Those results were scored under a different question and a different
  supervision policy and are not comparable to anything here. If a quantity
  matters, re-measure it in this project and give it a file above.
- Anything from `data/results/x80_*`: that line's primary gate failed and
  its edit stages never ran
  ([x80](experiments/x80-qwen-refusal-harm-contrast-edit.md)).
