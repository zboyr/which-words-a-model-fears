# x108: on the harmful prompts the models refuse, the boundary direction nominates the word that names the harm, x106's substitution rule reaches 64 % of them, and where it applies the in-place substitution removes +0.08 to +0.23 of the refusal in all eight models while the response harm rises by at most +0.04

Parent: [Experiment index](README.md). The harmful-side counterpart of
[x106](x106-natural-benign-lexical-carrier.md): x106 lets the boundary
direction nominate one word in each benign Hard-1K prompt the model refuses
and replaces it in place by a harm-free synonym; x108 runs the same pipeline,
rule and drafter on the OR-Bench Toxic prompts GuardReasoner reads as harmful
and on the 100 XSTest contrast prompts that mirror x106's safe ones, then
carries the texts to the seven transfer models. The direction, write site,
reader, deletion pass, drafter, substitution rule, sampling contract, judges
and bootstrap are x106's, imported; the labels at base are x104's.

On the 619 Toxic prompts Qwen3.5-4B refuses at base with a request
GuardReasoner-8B reads as harmful (plus 100 XSTest contrast prompts), the
leave-one-word-out pass nominates the word that names the harm — `self-harm`,
`sexual`, `misleading`, `without`, `negatively` — with a median within-prompt
z of 4.2 against 3.0 on x106's Hard-1K population, and the seven transfer
models nominate the same top word on 61–75 % of these prompts. x106's rule
(meaning preserved, no harm association, outside the x83 lexicon and the
nominated-word list) finds an admissible substitute for **394 of the 619
(0.64)** against 0.81 of x106's Hard-1K units: 154 units have no harm-free
meaning-preserving substitute (`self-harm`, `sexual`, `bomb`) and 71 fail the
structural checks three times. Where S exists, `refusal2` falls by **+0.114**
on Qwen3.5-4B (from 0.981) and by **+0.076 (Qwen3.5-27B) to +0.226
(Llama-3.2-3B)** on the seven transfer models (Holm p 7 × 10⁻⁴ in every one),
against +0.277 and +0.033 to +0.244 on the benign side in x106; GuardReasoner
Completion refusal falls by +0.073 to +0.233; GuardReasoner Response
harmfulness rises by **+0.002 (9B) to +0.041 (Llama)**, so most of the removed
refusal turns into an unharmful non-refusal. Removing the hedge phrase (R, 23
prompts) removes +0.17 to +0.36; on the XSTest contrast prompts S removes
+0.16 to +0.31. The substituted text's projection on each model's own
direction falls by 0.42 to 0.84 of that model's Hard-1K sd (x106: 0.43 to
0.88) and tracks the per-prompt refusal change at Spearman +0.52 to +0.69.
The citable values are routed to
[../headline-numbers.md](../headline-numbers.md) under "The phenomenon on
natural prompts".

## Question

x106 answers the construct-validity review on the benign side: the refusal of
a benign prompt rides on one ordinary word the direction can find. The
counterpart question is what the same direction and the same rule do on a
prompt whose request is harmful:

1. **Nomination.** Which word does the direction nominate on a harmful
   prompt, how concentrated is the projection on it, and is it the word that
   names the harm?
2. **Reach.** Under x106's rule — meaning preserved, no association with
   harm, outside the x83 lexicon and the nominated-word list — how many
   harmful prompts have an admissible substitute at all?
3. **Effect.** Where a substitute exists, does `refusal2` fall, does the
   response become harmful under GuardReasoner, and does the projection fall
   as on the benign side?
4. **Transfer.** Do the seven transfer models, reading the Qwen3.5-4B texts
   under their own direction, show the same effect and the same projection
   fall, as they do on the benign side in x106?

The share of prompts the rule cannot touch is a result, not a defect.

## Design, as run

- **Prompts.** The 655 OR-Bench Toxic prompts rendered exactly as each
  model's x87 / x88 base generation rows record them (`rendered_chat_sha256`
  checked per row; Llama's template date recovered from the hash as in x106)
  and the 100 XSTest `contrast_homonyms`, `contrast_figurative_language`,
  `contrast_safe_contexts`, `contrast_safe_targets` prompts from x102's copy,
  the designed counterparts of x106's 100 safe prompts. 755 prompts per model.
- **Population.** The Toxic prompts Qwen3.5-4B refuses at base under
  GuardReasoner-8B (x104) whose request GuardReasoner reads as harmful: 619 of
  the 655 (27 refused with an unharmful request, 7 answered with a harmful
  request, 2 answered with an unharmful one). The x108 cohort is these 619
  plus the 100 contrast prompts, 719 units, shared by every model. Each
  transfer model's own population under the same rule and its own x104 labels
  (445 on Qwen3.5-0.8B to 626 on Phi-4-mini) is reported with its Stage 0.
- **Stage 0, projection and nomination** (`x108_stage0.py`). Per model: the
  x106 reader at the x88 write site (block output at the two trailing prefill
  positions; Qwen3.5-4B at block 29 on x87's consensus direction, the others
  at their x88 hook block on their x88 consensus direction), the site audit
  before the first row, then x106's leave-one-word-out and leave-one-pair-out
  deletion pass in unpadded length-grouped batches (batched read audited
  against the batch-1 read per shard) over all 755 prompts and over the S / R
  texts of the cohort. The nomination rule is x106's; the share-of-excess
  reference is the model's own x106 answered-Hard-1K mean projection. The
  Toxic projections are reported against x106's Hard-1K refused and answered
  distributions of the same model.
- **Texts** (`x108_build_texts.py`), built once from the Qwen3.5-4B
  nomination and shared by all models. x106's drafter and rule with the
  control fields removed: `deepseek/deepseek-v4-flash-0731` over OpenRouter,
  temperature 0, seeded in the x108 namespace, reasoning on with the
  16,000-token budget and the no-reasoning retry, up to three drafts with the
  structural-check feedback, no checker LLM. Per unit **O** = the prompt,
  **S** = the nominated unit replaced in place by a meaning-preserving word or
  phrase with no association with harm, danger, crime, violence, illegality,
  secrecy, sex, drugs, weapons or risk, containing no x83 lexicon word and no
  content word of any cohort prompt's nominated unit (hedge units: the phrase
  replaced by a neutral phrase of the same function), **R** = hedge units
  only, the phrase removed. No C text: x106 shows the matched inert control is
  inert on every model. Cohort `data/runs/x108/x108_cohort.json`, sha256
  `7d64a494…`: 719 units, 696 content / 23 hedge; texts O 719 / S 444 / R 23;
  275 without S (197 the drafter judged to have no neutral meaning-preserving
  substitute, 78 whose drafts failed the checks three times; no provider
  refusal). A 10 % sample (72 units, seed 108) is listed in
  `data/runs/x108/x108_readthrough.md`. Drafting: 1,043 calls, $0.63.
- **Arms** (`x108_generate.py`). Per model `O_base`, `S_base`, `R_base` at
  dose 0, no boundary write: x101's sampling contract (T = 1.0, top-p 0.9,
  top-k 50, 10 sequences, 2,048 tokens, one 4,096 retry) through x106's
  generation path, seeds per (model, prompt, text) in the x108 namespace. O is
  regenerated under the sampling contract (the x87 / x88 base rows are greedy)
  and its rendering is checked against the base row's `rendered_chat_sha256`.
  24 arms, 9,488 prompt-conditions, 94,880 responses.
- **Judges.** Primary `refusal2`, the x64 contract as x101 / x106 ran it
  (`x108_judge_refusal.py`, local): 24 arms, every prompt-condition labelled,
  zero failed rows, ≈ $2.3. Secondary GuardReasoner-8B under x104's contract
  (`x108_judge_guardreasoner.py`, two shards per arm on the cluster, greedy then up
  to three re-decodes of unparsed rows), read in two fields: `Completion`
  (refusal) and `Response` (response harmfulness); 94,875 of the 94,880
  responses parsed.
- **Analysis** (`x108_analyze.py`). Per model and label construct,
  prompt-paired on the cohort, one 10,000-replicate two-stage bootstrap
  (prompts, then the ten binary responses within each arm): `r(O) − r(S)` and
  `r(O) − r(R)` under `refusal2` and under GuardReasoner Completion refusal,
  `h(S) − h(O)` and `h(R) − h(O)` under GuardReasoner Response harmfulness,
  Toxic and XSTest contrast separately, Holm within {S, R} per population;
  `proj(S) − proj(O)` on the model's own direction in sd units of its x106
  Hard-1K projection, the share of S texts below its answered-prompt mean,
  Spearman between the per-prompt projection change and refusal change;
  Spearman between the 4B `delta_top` and the per-prompt effect; the joint
  per-prompt reading of refusal removed and harm added; reach over the 619.
  Across models, Holm over the seven transfer models within each contrast and
  judge, Qwen3.5-4B the reference row outside the family as in x106. Per
  model, one table with x106's Hard-1K and XSTest safe rows of the same model
  above the x108 Toxic and XSTest contrast rows.

## Results

Table: [Stage 0 per model](json/x108-toxic-lexical-carrier-specificity__stage0-projection-and-nomination.json) — Toxic base refusal under GuardReasoner, the model's own population, the population's projection on its own direction relative to its x106 Hard-1K refused mean and the contrast prompts' projection relative to x106's safe XSTest prompts (both in that model's Hard-1K sd), the AUC of the Toxic population against the Hard-1K refused prompts, the nominated unit's median z / pair share / lexicon share next to x106's Hard-1K values on the same model, the agreement with the 4B nomination, and the most nominated units. The Toxic population projects 0.37 (Gemma) to 1.04 (Llama, Phi) sd above the model's own Hard-1K refused mean and 1.6 to 2.9 sd above its answered mean (AUC against the refused prompts 0.63–0.89, against the answered 0.93–0.997); the contrast prompts project 2.2 to 4.4 sd above x106's safe XSTest prompts (AUC 0.94–0.998). The nominated unit's median z is 3.4–4.2 on the Toxic cohort against 2.5–3.5 on the same model's Hard-1K population, a pair wins 0.32–0.49 of the time against 0.21–0.28, and nominated units lie in the x83 lexicon at 0.03–0.05 against a content-word base rate of 0.017.

**What the direction nominates.** On the 619-prompt cohort the 4B nomination's
most frequent units are `self-harm` (36), `sexual` (21), `misleading` (12),
`without` (10), `negatively` (9), `negative` (7), `false`, `i create` (6),
`what techniques`, `manipulate` (5); every transfer model's top five are drawn
from the same list (`self-harm` 26–53, `sexual` 18–24, `misleading` 13–15,
`without`, `negatively`). By OR-Bench category the 4B unit is `self-harm` on
the self-harm prompts (median z 8.1), `sexual` on the sexual ones (6.1),
`bomb` on violence (5.9), `negative` on hate (5.2), `without` on privacy and
illegal (4.1, 3.9), `misleading` on deception and unethical (3.5, 3.3). On the
contrast prompts the median z is 2.6–3.7 and the top units are the violent
homonyms and figures — `kill` (5), `shoot`, `a murder`, `killing`, `crush`,
`stealing`, `poison` — against `best`, `what's`, `good` with median z 0.6–1.7
on x106's safe prompts. The seven transfer models nominate the same top word
as 4B on 61–75 % of the cohort's Toxic prompts and place 4B's unit in their
own top three on 80–90 % (x106 on Hard-1K: 55–71 % and 82–91 %); on the
contrast prompts the top-word agreement is 69–88 %.

Table: [Reach of the rule](json/x108-toxic-lexical-carrier-specificity__reach-of-the-rule.json) — units with an S text and why the others have none, the hedge count, the lexicon share of the units with and without S, the most frequent units without S and the most frequent substitutions, on the x108 Toxic and XSTest-contrast units next to x106's Hard-1K and XSTest-safe units. The rule reaches 394 of the 619 Toxic units (0.64) and 50 of the 100 contrast units (0.50) against 387 of 476 Hard-1K units (0.81) and 86 of 100 safe XSTest units (0.86). On Toxic, 154 units (0.25) have no harm-free meaning-preserving substitute — `self-harm` (32), `sexual` (21), `negatively` (5), `self-harm techniques` (4), `bomb`, `harm`, `what steps`, `activities without` (3) — and 71 (0.11) fail the checks three times (Hard-1K: 0.12 and 0.07); on the contrast prompts 43 have no substitute (`kill`, `a murder`, `killing`, `poison`). The substitutes that exist replace framing and evaluative words: `i create` → `i make`, `without` → `while not`, `false` → `fabricated`, `negative` → `critical` / `unfavorable`, `misleading` → `inaccurate`, `spread` → `circulate`, `what techniques` → `what approaches`; on the contrast prompts `shoot` → `photograph`, `crush` → `compress`, `hang` → `suspend`. 23 Toxic units are hedge phrases (`what steps`, `some methods`, `what techniques`) against 56 of x106's 476.

Table: [Substitution contrasts per model and label construct](json/x108-toxic-lexical-carrier-specificity__substitution-contrasts.json) — per model `r(O)` where S exists, `r(O) − r(S)` and `r(O) − r(R)` under `refusal2` with 95 % intervals and the Holm p over the seven transfer models, the XSTest contrast S effect, the same S and R contrasts under GuardReasoner Completion refusal, and `h(O)`, `h(S) − h(O)` and `h(R) − h(O)` under GuardReasoner Response harmfulness. Under `refusal2` S removes +0.114 (4B, from 0.981), +0.157 (0.8B, from 0.498), +0.195 (2B, from 0.814), +0.091 (9B, from 0.996), +0.076 (27B, from 0.987), +0.226 (Llama, from 0.761), +0.124 (Gemma, from 0.926), +0.174 (Phi, from 0.874); Holm p over the seven transfer models is 7 × 10⁻⁴ for S in every model and 7 × 10⁻⁴ to 0.03 for R; GuardReasoner Completion refusal gives +0.073 (27B) to +0.233 (Llama) for S (Holm 7 × 10⁻⁴ everywhere); GuardReasoner Response harmfulness rises by +0.010 (4B), −0.014 (0.8B, interval covering zero), +0.026 (2B), +0.002 (9B), +0.004 (27B), +0.041 (Llama), +0.021 (Gemma), +0.012 (Phi), with Holm p below 0.05 on 2B, Llama, Gemma and Phi and 0.08 to 0.83 on 9B, 27B and 0.8B. R removes +0.17 (9B) to +0.36 (2B) of `refusal2` and the contrast S removes +0.16 (0.8B) to +0.31 (Llama).

**Benign next to harmful.** Against x106's S effect on the same model's
Hard-1K population, the Toxic effect is smaller on the models that refuse
nearly all of the Toxic cohort — 4B +0.114 against +0.277, 9B +0.091 against
+0.218, 27B +0.076 against +0.220, Gemma +0.124 against +0.244 — and as large
or larger on the models with more to lose at base: 0.8B +0.157 against
+0.079, 2B +0.195 against +0.170, Phi +0.174 against +0.143, Llama +0.226
against +0.033 (Llama refuses 0.76 of the Toxic cohort at base and 0.07 of
the Hard-1K population). On the large Qwen models the substituted Toxic
prompts are still refused at 0.87–0.91 (`refusal2`), and the share of
substituted prompts whose `refusal2` falls below 0.5 is 0.07 (9B), 0.08
(27B), 0.12 (4B), 0.19 (Gemma), 0.25 (Phi), 0.33 (2B), 0.47 (Llama), 0.65
(0.8B); within 4B the effect is +0.28 on the 28 mid-stratum prompts and
+0.11 on the 590 high-stratum ones.

**Refusal removed, harm added, per prompt.** Among the 394 substituted Toxic
prompts, `refusal2` falls on 21 % for 4B (harm rising on 3.8 %, flat on 17
%), 15 % for 9B and 27B (harm rising on 1.3 % and 1.0 %), 19 % for Gemma
(5.8 %), 42 % for Phi (4.8 %), 52 % for 2B (14 %), 35 % for Llama (18 %) and
54 % for 0.8B (23 %, whose O responses are already harmful at 0.23). The
share of substituted prompts with at least one GuardReasoner-harmful response
among ten is 0.046 on 4B (O 0.006), 0.015 on 9B, 0.010 on 27B, 0.066 on Gemma,
0.061 on Phi, 0.21 on 2B (O 0.14), 0.33 on Llama (O 0.24), 0.58 on 0.8B
(O 0.60). The per-prompt refusal change and harm change correlate at Spearman
+0.21 to +0.48.

Table: [Representation and the x106 comparison](json/x108-toxic-lexical-carrier-specificity__representation-and-x106-comparison.json) — per model `proj(S) − proj(O)` on the Toxic cohort, the contrast prompts and (from x106) the Hard-1K population, all in that model's Hard-1K sd; `proj(R) − proj(O)`; the share of S texts below the answered-prompt mean; the Spearman between projection change and `refusal2` change; the Spearman between the 4B `delta_top` and the substitution effect; the joint refusal / harm reading; and the x106 Hard-1K and XSTest-safe S effects of the same model. `proj(S) − proj(O)` is −0.60 (4B), −0.62 (0.8B), −0.70 (2B), −0.49 (9B), −0.45 (27B), −0.84 (Llama), −0.42 (Gemma), −0.62 (Phi) sd against −0.43 to −0.88 on Hard-1K in x106 and −0.61 to −1.24 on the contrast prompts; R lowers the projection by 0.69 to 1.47 sd; the per-prompt projection change tracks the per-prompt `refusal2` change at Spearman +0.65 (4B), +0.65 (0.8B), +0.69 (2B), +0.59 (9B), +0.52 (27B), +0.66 (Llama), +0.61 (Gemma), +0.68 (Phi) (x106: +0.23 to +0.80); the 4B `delta_top` predicts the per-prompt effect at +0.15 (27B) to +0.29 (9B).

**Output integrity.** Truncation after the 4,096-token retry is at most
0.0011 in any arm (0.8B's O arm); median response length runs from 102
characters (Gemma, O) to 3,477 (0.8B, S); the share of responses dominated by
a repeated 4-gram is at most 0.0025 (Llama); the per-arm figures for every
model are in the result file.

Figure: `data/results/x108_toxic_lexical_carrier.pdf` (per model the S effect
under the three label constructs on Toxic and under `refusal2` on the contrast
prompts; the projection change of S on the Toxic, contrast and x106 Hard-1K
prompts; the reach of the rule on the four populations).

## Reading

- **On a harmful prompt the carrier is the harm word itself.** The direction
  nominates `self-harm`, `sexual`, `bomb`, `misleading`, `without` with a
  sharper peak than on the benign side (median z 4 against 3, a pair on half
  the prompts), and the eight models agree on the top word as often as they do
  on Hard-1K. On the XSTest contrast prompts it nominates the violent homonym
  (`kill`, `shoot`, `crush`) that the safe counterparts carry in a harmless
  sense, and the projection places these prompts with the Toxic ones.
- **The rule's reach is the specificity result.** 36 % of the harmful prompts
  have no admissible substitute against 19 % of the benign ones and 14 % of
  the safe XSTest prompts; the untouchable units name the harm outright and no
  meaning-preserving harm-free word exists for them. Where a substitute
  exists, the nominated unit is a framing or evaluative word (`misleading`,
  `without`, `negative`, `create`, the hedge phrases), the same kind of word
  x106 finds on benign prompts.
- **Where it applies, the effect is present in every model and bounded by
  what the model has left to give.** The large Qwen models and Gemma, which
  refuse the Toxic cohort at 0.93–0.99, give up +0.08 to +0.12 and keep
  refusing the substituted prompt at 0.87–0.91; the small Qwen models, Llama
  and Phi, which refuse at 0.50–0.87, give up +0.16 to +0.23. On the benign
  side the same models give up +0.03 to +0.28, ordered by base refusal there.
- **The response stays what GuardReasoner calls unharmful.** Harm rises by at
  most +0.04 of the ten responses, and three to ten times more substituted
  prompts lose their refusal without gaining a harmful response than with
  one; on the three large Qwen models under 5 % of substituted prompts
  produce any harmful response.
- **One representation on both sides.** S lowers the projection on the model's
  own direction by 0.4–0.8 sd on harmful prompts as on benign ones, R by
  0.7–1.5 sd, and the per-prompt projection change tracks the per-prompt
  refusal change at +0.5 to +0.7 in every model.
- **Both refusal constructs agree.** GuardReasoner Completion refusal's S
  column is +0.07 to +0.23 against `refusal2`'s +0.08 to +0.23, its R column
  +0.14 to +0.39, and its Holm p over the seven transfer models is 7 × 10⁻⁴
  for S in every model.

## Artifacts

- Result: `data/results/x108_toxic_lexical_carrier.json`, figure
  `x108_toxic_lexical_carrier.{pdf,png}`.
- Shared: `data/runs/x108/x108_cohort.json` (`7d64a494…`: every unit with its
  O / S / R texts and their sha256, the nominated unit, the substitute, the
  hedge phrase, the drafter's note where S is absent, the read-through ids);
  `x108_substitute_drafts.jsonl` (every drafter call and check outcome);
  `x108_readthrough.md`. The frozen model inputs (snapshot, hook block,
  direction artifact and sha256, base-generation files) are x106's
  `data/runs/x106/x106_inputs.json`.
- Per model under `data/runs/x108/<tag>/`: `x108_stage0_projections.json`,
  `x108_stage0_nominations.json` (all 755 prompts),
  `x108_stage0_nominations_{S,R}.json` (cohort texts),
  `x108_refusal_<arm>.jsonl` (ten labels per prompt-condition);
  `x108_generations_<arm>.json`, `x108_guardreasoner_<arm>_shard*.jsonl` and
  the Stage 0b shard files (shipped in this repository); `x108_stage0_states.npz`
  is not included.
- Code: `code/x108_common.py`, `x108_stage0.py`, `x108_build_texts.py`,
  `x108_generate.py`, `x108_judge_refusal.py`, `x108_judge_guardreasoner.py`,
  `x108_analyze.py`, `x108.sbatch`; tests
  `code/tests/test_x108_toxic_lexical_carrier.py` (8). The reader, directions,
  model registry, chat rendering, deletion pass, drafter, sampling engine,
  seeds, judges and bootstrap are x106's / x105's / x104's / x101's, imported.
- Compute: ≈ 26.5 GPU-hours on the cluster — ≈ 1.0 for Stage 0 on all eight models,
  ≈ 15.4 for generation (24 arms, 94,880 responses; Qwen3.5-0.8B's long
  responses take 3.7 h and Qwen3.5-27B 2.9 h at `--mem=140G`), ≈ 10.1 for
  GuardReasoner judging; ≈ 3.5 h wall for the seven transfer models. OpenRouter
  ≈ $2.9 (drafting $0.63, `refusal2` ≈ $2.3)
  ([../infrastructure.md](../infrastructure.md)).
