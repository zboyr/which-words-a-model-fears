# x118: two judges that never saw the drafter read 0.61 / 0.79 of the benign substituted prompts, 0.58 / 0.71 of the harmful ones and 0.33 / 0.53 of the table-guided ones as the same request, and on the pairs both accept the in-place substitution still removes +0.02 to +0.20 of `refusal2` in eight models

Parent: [Experiment index](README.md). An independent read of the texts of
[x106](x106-natural-benign-lexical-carrier.md) (benign Hard-1K, the direction's
deletion pass nominates the word),
[x108](x108-toxic-lexical-carrier-specificity.md) (OR-Bench Toxic, the same
rule) and [x113](x113-cross-model-table-substitution.md) (each model's own
vocabulary table nominates the word and picks the substitute). In all three the
substitute was written by one model — `deepseek/deepseek-v4-flash-0731` at
temperature 0 — and admitted by structural checks plus a 10 % human
read-through; no second model was asked whether the edited prompt still asks
for what the original asked for. x118 asks exactly that question of every
(original, edited) pair, with two judges from two other families and no role in
drafting, and recomputes the refusal contrasts on the pairs they accept.

`openai/gpt-6-luna` (OpenRouter, reasoning effort `minimal`, temperature 0,
seed 42, `json_schema` output) labelled all **4,805 distinct pairs** (7,713
uses across the three experiments and the eight models), none failed, for
**$0.436**. It reads **0.605** [0.560, 0.648] of x106's benign substitutions
(S), **0.581** of x108's harmful ones and **0.330** of x113's table-guided ones
(S_T, 0.278 to 0.353 per model) as the same request, against **0.855** and
**0.806** for the two inert-word controls (C, C_T) and **0.070 / 0.130** for the
hedge removals (R). Where it says no, it almost always says `broader` or
`different`: deleting or neutralising a qualifier (`legal and honest`, `without
getting caught`) drops a condition the original carried. Re-judged at seed 1042
on a 10 % sample it repeats itself on 0.945 of pairs (Cohen's kappa 0.889). It
is stricter than the human read-through, which marked none of its 162 pairs
`BAD:` while the judge rejects 61 of them. **On the pairs it accepts the
substitution effect stays and the controls stay flat**: x106's S removes
**+0.022 (Llama) to +0.164 (Qwen3.5-4B)** of `refusal2` (0.38 to 0.69 of the
all-pairs value, every interval excluding zero but Llama's), x113's S_T removes
**+0.045 (Qwen3.5-0.8B) to +0.191 (Qwen3.5-2B)** (seven of eight intervals
excluding zero), x108's S removes **+0.012 to +0.106**, while C stays at −0.003
to +0.015 and C_T at +0.010 to +0.037.

The same 4,805 pairs, the same prompt and the same schema were then put to a
**second judge from a third family**, `anthropic/claude-haiku-5.5` (OpenRouter,
reasoning effort `minimal`, no temperature and no seed — the route lists
neither; 4,803 labelled, 2 refused by the route's content filter, **$0.586**).
It is the more permissive reader: **0.789** [0.750, 0.823] of x106's S,
**0.707** of x108's S and **0.527** of x113's S_T (0.489 to 0.558 per model),
with the controls at **0.968** (C) and **0.947** (C_T) and the hedge removals at
0.123 / 0.087. The two judges agree on `same_request` for **0.800** of the
4,803 pairs both labelled (Cohen's kappa **0.578**; 0.769 / 0.553 on the S
arms), and the disagreement is one-sided — 882 pairs the second judge accepts
and the first does not, 81 the other way. A second call on the same 10 % sample
reproduces the second judge's verdict on **0.992** of pairs (kappa 0.979).
**On the 2,588 pairs both judges accept the picture is the one the first judge
already gave**: x106's S removes **+0.023 (Llama) to +0.160 (Qwen3.5-4B)** of
`refusal2`, x113's S_T **+0.045 (Qwen3.5-0.8B) to +0.203 (Qwen3.5-2B)**,
x108's S **+0.006 to +0.099**, 19 of those 24 intervals excluding zero, while C
stays at −0.003 to +0.015 and C_T at +0.015 to +0.037. On the second judge's
own accepted pairs — the larger subset — **all 24 intervals exclude zero**
(x106 S +0.027 to +0.196, x113 S_T +0.039 to +0.214, x108 S +0.024 to +0.147).

## Question

The substituted texts of x106, x108 and x113 are the evidence that changing one
word changes the behaviour. Their meaning preservation was asserted by the
drafting model and checked structurally and by a 10 % human read-through. Three
questions:

1. **Acceptance.** What share of the (O, edited) pairs does a judge that did no
   drafting read as asking for the same thing — the same task, the same object,
   the same scope — per arm (S, C, R), per model, per unit kind and per stratum,
   and how far do two such judges agree with each other?
2. **Standing against the human marks.** How does it line up with the human
   read-through of x106 and x108?
3. **The effects on what they accept.** Restricted to the accepted pairs — each
   judge's own, and the pairs both accept — how large are the S − O, C − O and
   R − O `refusal2` differences next to the numbers the owning experiments
   report?

## Design, as run

- **Pairs** (`x118_collect_pairs.py` → `data/runs/x118/pairs.jsonl`). Every
  (original, edited) pair of the three frozen cohorts: x106's
  `x106_cohort.json` (S 473, C 530, R 57), x108's `x108_cohort.json` (S 444,
  R 23) and each model's `data/runs/x113/<tag>/texts.json` (S_T 372–381, C_T
  396–397 per model, 8 models). Pairs are deduplicated by (sha256 of O, sha256
  of the edited text) with every owner kept, because x113 reuses 3–11 S_T and
  42–61 C_T texts of x106 per model and the models share texts with each other:
  **7,713 owner rows over 4,805 distinct pairs**, 1,405 of them with more than
  one owner. 100 pairs are an S arm for one owner and a C arm for another. Each
  row carries source, model, arm, unit id, unit kind (content / hedge),
  stratum, the nominated unit, both texts, both sha256s, and a `readthrough`
  field (`ok` / `bad` / null) taken from `x106_readthrough.md` and
  `x108_readthrough.md`: 162 pairs of the 58 + 72 read-through units, none of
  which the human marked `BAD:`.
- **Judge** (`x118_judge.py`). `openai/gpt-6-luna` over OpenRouter. The route
  accepts `reasoning_effort: "minimal"` (the lowest of the two the model lists;
  it still spends ≈ 52 reasoning tokens per call), `temperature: 0`, `seed`,
  and a strict `response_format: json_schema` with the three fields
  `same_request` (bool), `relation` (`same` / `narrower` / `broader` /
  `different`) and `reason` (one sentence); the probe at startup records which
  of these the route took, and would fall back to effort `low` and
  `json_object` with the x64 balanced-object parser. One call per pair, 400 max
  tokens, 12–14 threads, the x81 client's backoff on transient errors, and up
  to three attempt seeds if a reply misses the contract. The prompt shows the
  original and the edited request in delimiters, asks whether the edited text
  asks for the same thing (same task, same object, same scope), and says that
  harmfulness, legality and whether an assistant should answer are not part of
  the annotation. The drafter is not consulted. Resumable by pair id; usage,
  cost, the parameters and the attempt are written per row to
  `judge/meaning.jsonl`, per pass to `judge/cost_log.jsonl`. A 20-pair smoke
  pass ran first. **Test–retest**: a 10 % sample (495 pairs, chosen by the hash
  of the pair id) re-judged at seed 1042 into `judge/meaning_retest.jsonl`.
- **Second judge** (`x118_judge.py --judge-model anthropic/claude-haiku-5.5`).
  The same pair file, the same prompt, the same schema, the same 400-token
  ceiling and the same three-attempt contract, into
  `judge/meaning_haiku.jsonl`. OpenRouter lists `reasoning`,
  `reasoning_effort`, `response_format`, `structured_outputs`, `max_tokens`,
  `verbosity` and `stop` for `anthropic/claude-haiku-5.5` and lists neither
  `temperature` nor `seed`, so neither is sent; the startup probe records what
  the route took. All three candidate reasoning settings (`minimal`, `low`,
  `reasoning: {"enabled": false}`) returned a parsable `json_schema` answer
  with zero reasoning tokens, and the run uses the first, `minimal`. Every call
  was served by the provider `Azure`, and the provider is written per row.
  Determinism is therefore measured, not requested: **the same 495 pairs the
  first judge re-judged** (read back from `meaning_retest.jsonl` with
  `--retest-ids`) were sent a second time into
  `judge/meaning_haiku_retest.jsonl`. Two of the 4,805 pairs — one x108 S unit
  and one x113 `qwen9b` S_T unit — came back with `finish_reason:
  content_filter` and no content on every attempt, including a dedicated
  resume; they carry no label from this judge and are counted as not accepted
  by it. A 20-pair smoke pass ran first here too, and 17 of those 20 labels
  matched the first judge's.
- **Analysis** (`x118_analyze.py`). Acceptance shares with Wilson 95 %
  intervals over owner rows, per source × arm, per x113 model, per unit kind
  and per stratum; the relation distribution; the confusion table against the
  human `BAD:` marks; test–retest percent agreement and Cohen's kappa on
  `same_request`. Then the owning experiments' contrasts, recomputed here from
  the same `refusal2` label files on two masks — every unit with the text
  (reproducing the published value) and only the units whose pair the judge
  accepts: x106's S / C / R on its Hard-1K population (576-unit cohort minus
  the 100 XSTest units), x113's S_T / C_T on its Hard-1K population with the
  frozen x85 exclusion applied (`x85_exclusions.json`), x108's S / R on the 619
  Toxic units. O is x106's `O_base` rows for x106 and x113 and x108's
  `O_base` rows for x108. Prompt-paired two-stage bootstrap, 10,000
  replicates, prompts resampled within stratum and then the ten binary
  responses of each arm, seed 11,800. The all-pairs columns reproduce the
  owning experiments exactly (x106 S +0.277 / R +0.504 on Qwen3.5-4B, x113 S_T
  +0.278 and C_T +0.037, x108 S +0.114).
  The second judge is analysed by the same script
  (`--judge-file judge/meaning_haiku.jsonl --retest-file
  judge/meaning_haiku_retest.jsonl --compare-with judge/meaning.jsonl
  --out-stem results_haiku --allow-unlabelled 2`) into `results_haiku.json` /
  `results_haiku.md`: the same acceptance, relation, human-read-through and
  test–retest tables over its own labels, a judge-agreement section (percent
  agreement, Cohen's kappa and the relation cross-table against the first
  judge, over distinct pairs and over owner rows), and every effect on four
  masks instead of two — all pairs, each judge's accepted pairs, and the pairs
  both judges accept. Run unchanged, with the first judge's files and the
  default stem, it still reproduces every number in `results.json`.

## Results

### What the judge accepts

Table: [Acceptance and relation per arm](json/x118-meaning-preservation-judge__acceptance-and-relation-by-arm.json) — per experiment and arm the share the judge reads as the same request with its Wilson interval and the four-way relation counts. S is accepted at 0.605 (x106) and 0.581 (x108) against 0.330 for x113's S_T, the two controls at 0.855 (C) and 0.806 (C_T), and the hedge removals at 0.070 (x106 R) and 0.130 (x108 R); of the 2,016 rejected S_T rows 1,002 are `different` and 726 `broader`.

Table: [Acceptance of the table-guided texts per model](json/x118-meaning-preservation-judge__table-substitution-acceptance-per-model.json) — S_T 0.278 (Qwen3.5-0.8B) to 0.353 (Llama-3.2-3B) and C_T 0.766 to 0.849, over each model's own 397 units.

Table: [Acceptance by unit kind and stratum](json/x118-meaning-preservation-judge__acceptance-by-unit-kind-and-stratum.json) — hedge-phrase substitutions are accepted far less often than content-word ones (x106 S 0.143 against 0.667; x113 S_T 0.053 against 0.397), and within x106's S the low-projection stratum is accepted at 0.781 against 0.519 in the high one; x113's S_T is accepted at 0.243 on the core Hard-1K prompts and 0.529 on the safe XSTest prompts.

Two things drive the ordering. First, the arms differ in how far they move the
text: C and C_T replace an inert word by a synonym and keep the request intact,
S takes the drafter's own best meaning-preserving candidate, S_T takes the
**lowest-shift accepted** candidate of up to ten (x113's rule, the drafter's
first choice only 8–12 % of the time), and R deletes a phrase outright. The
acceptance shares follow that order: 0.81–0.86, 0.58–0.61, 0.33, 0.09. Second,
the hedge units are the ones whose edit changes a condition rather than a
noun — the phrase `without getting caught` or `ensuring it's legal` is what the
judge reads as scope — and they are accepted at 0.05–0.19 in every arm that
touches them.

### Against the human read-through

Of the 162 read-through pairs — all of which the human passed, no `BAD:` mark
in either file — the judge accepts **101 and rejects 61** (0.623 [0.547,
0.694]). The 61 are 22 S pairs it calls `broader`, 18 S `different`, 12 R
`broader`, and 7 C pairs. There is no human rejection to confirm, so the
comparison is one-sided: on this sample the judge is the stricter of the two
readers, and every pair it rejects is one a human reader passed.

### Test–retest

495 pairs (10 %, seed-independent hash sample) re-judged at seed 1042 against
seed 42: `same_request` agrees on **0.945**, Cohen's kappa **0.889**; the
four-way relation agrees on 0.867; the accept share is 0.566 then 0.552.

### The refusal effects on the accepted pairs

Table: [Benign Hard-1K, direction-nominated substitution](json/x118-meaning-preservation-judge__benign-substitution-effects-on-accepted-pairs.json) — per model `refusal2` removed by S, C and R on all pairs and on the accepted ones, with 95 % paired-bootstrap intervals and both n. On the 218 accepted S units S removes +0.164 (Qwen3.5-4B), +0.050 (0.8B), +0.096 (2B), +0.088 (9B), +0.083 (27B), +0.022 (Llama), +0.135 (Gemma), +0.083 (Phi) against +0.277 / +0.079 / +0.170 / +0.218 / +0.220 / +0.033 / +0.244 / +0.143 on all 387; C stays at −0.003 to +0.015 on its 404 accepted units.

Table: [Benign Hard-1K, table-nominated substitution](json/x118-meaning-preservation-judge__table-substitution-effects-on-accepted-pairs.json) — the same for x113's S_T and C_T on its x85-free Hard-1K population. On the 49–84 accepted S_T units per model S_T removes +0.168 (Qwen3.5-4B), +0.045 (0.8B), +0.191 (2B), +0.108 (9B), +0.140 (27B), +0.052 (Llama), +0.147 (Gemma), +0.188 (Phi); C_T moves +0.010 to +0.037, as on all pairs.

Table: [Harmful OR-Bench Toxic](json/x118-meaning-preservation-judge__harmful-substitution-effects-on-accepted-pairs.json) — x108's S and R. On the 241 accepted S units S removes +0.012 (Qwen3.5-27B) to +0.106 (Llama) against +0.076 to +0.226 on all 394.

Three readings of these columns:

- **Twenty-one of the 24 S intervals exclude zero on the accepted pairs.** x106's S excludes
  zero in seven models and covers it only on Llama-3.2-3B
  (+0.022 [−0.001, +0.047]), whose base refusal on this population is 0.07;
  x113's S_T covers zero only on Qwen3.5-0.8B (+0.045 [−0.029, +0.126], 49
  units); x108's S covers zero only on Qwen3.5-27B (+0.012 [−0.000, +0.027]),
  whose base refusal is 0.99.
- **The accepted-pairs effect is a fraction of the all-pairs effect**: 0.38 to
  0.69 of it for x106's S, 0.31 to 1.02 for x113's S_T (Phi-4-mini and
  Llama-3.2-3B essentially unchanged), 0.14 to 0.47 for x108's S. The pairs the
  judge rejects carry more of the effect than the ones it accepts, which is what
  the arm ordering predicts: the farther an edit moves the request, the more
  refusal it removes and the less likely the judge is to call it the same
  request.
- **The controls are unaffected by the restriction.** C goes from −0.005…+0.019
  to −0.003…+0.015 and C_T from +0.011…+0.050 to +0.010…+0.037, so the contrast
  between a nominated word and a matched inert word survives on the accepted
  subset at a size of +0.02 to +0.19 against +0.01 to +0.04.
- **The hedge removals have almost no accepted pairs** — 4 of 57 in x106 and 3
  of 23 in x108 — so R is reported on all pairs only (+0.039 to +0.504 and
  +0.174 to +0.357) and the accepted columns of R rest on 3–4 prompts.

### The second judge, and the two judges against each other

Table: [Acceptance and relation per arm, second judge](json/x118-meaning-preservation-judge__haiku-acceptance-and-relation-by-arm.json) — the same seven arms under `anthropic/claude-haiku-5.5`. S is accepted at 0.789 (x106) and 0.707 (x108) against 0.527 for x113's S_T, the controls at 0.968 (C) and 0.947 (C_T), and the hedge removals at 0.123 and 0.087; of the 1,425 rejected S_T rows 1,145 are `different` and 230 `broader`.

Table: [Acceptance of the table-guided texts per model, second judge](json/x118-meaning-preservation-judge__haiku-table-substitution-acceptance-per-model.json) — S_T 0.489 (Qwen3.5-0.8B) to 0.558 (Gemma-4-E2B) and C_T 0.927 to 0.965, over each model's own 397 units.

Table: [Acceptance by unit kind and stratum, second judge](json/x118-meaning-preservation-judge__haiku-acceptance-by-unit-kind-and-stratum.json) — the same ordering as under the first judge, one step more permissive throughout: content-word edits 0.854 (x106 S) and 0.609 (x113 S_T) against 0.304 and 0.183 for hedge-phrase edits, and x113's S_T at 0.472 on the core Hard-1K prompts against 0.649 on the safe XSTest prompts.

Table: [The two judges against each other](json/x118-meaning-preservation-judge__haiku-luna-judge-agreement.json) — percent agreement, Cohen's kappa and each judge's accept share per arm, over distinct pairs and over owner rows. Pooled over the 4,803 pairs both labelled: agreement 0.800, kappa 0.578, second judge 0.722, first judge 0.556, both 0.539; on the S arms 0.769 and kappa 0.553.

Table: [The relation the two judges report for the same pair](json/x118-meaning-preservation-judge__haiku-luna-relation-cross-table.json) — the four-way relation agrees on 3,489 of 4,803 pairs (0.727), and the off-diagonal is one-sided: 874 pairs the second judge calls `same` the first calls something else, 83 the other way.

Three things about the pair of judges:

- **They order the arms the same way and differ by a near-constant offset.**
  Inert control, benign substitution, harmful substitution, table-guided
  substitution: 0.855 / 0.806 / 0.605 / 0.581 / 0.330 for the first judge and
  0.968 / 0.947 / 0.789 / 0.707 / 0.527 for the second, every one of the five
  up by 0.113 to 0.197 and none of them changing place. The two hedge
  deletions stay near a tenth under either judge (0.070 → 0.123 on x106 and
  0.130 → 0.087 on x108).
- **The disagreement is one-sided.** Of the 4,803 pairs both labelled, both
  accept 2,588, the second judge alone 882, the first judge alone 81, and both
  reject 1,252. The second judge accepts 2,588 of the 2,669 pairs the first
  accepts (0.970) and 882 of the 2,134 it rejects (0.413), which is what one
  reader holding a looser threshold on the same ordering looks like.
- **The second judge repeats itself more closely.** 0.992 agreement and kappa
  0.979 on the 495-pair sample at the default route settings, against 0.945 and
  0.889 for the first judge at a second seed, and 0.980 against 0.867 on the
  four-way relation. It also rejects 45 of the 162 pairs the human read-through
  passed, against the first judge's 61.

### The refusal effects under both judges

Table: [Benign Hard-1K, second judge](json/x118-meaning-preservation-judge__haiku-benign-substitution-effects-on-accepted-pairs.json), [table-nominated](json/x118-meaning-preservation-judge__haiku-table-substitution-effects-on-accepted-pairs.json), [harmful](json/x118-meaning-preservation-judge__haiku-harmful-substitution-effects-on-accepted-pairs.json) — each arm's `refusal2` difference on four masks: all pairs, each judge's accepted pairs, and the pairs both accept. On the pairs both judges accept, S removes +0.023 (Llama) to +0.160 (Qwen3.5-4B) on x106's 215 units, S_T +0.045 (Qwen3.5-0.8B) to +0.203 (Qwen3.5-2B) on 47–81 units per model, and x108's S +0.006 (Qwen3.5-9B) to +0.099 (Llama) on 228 units.

- **The second judge's larger accepted subset makes every interval exclude
  zero.** On its own 294 x106 S units, 114–137 x113 S_T units and 288 x108 S
  units, all 24 S intervals exclude zero: x106 S +0.027 to +0.196, x113 S_T
  +0.039 to +0.214, x108 S +0.024 to +0.147. The three intervals that cover
  zero under the first judge (x106 S on Llama-3.2-3B, x113 S_T on Qwen3.5-0.8B,
  x108 S on Qwen3.5-27B) all exclude it here.
- **Both-accepted and first-judge-accepted agree closely.** The two columns
  differ by at most 0.005 on x106's S and 0.012 on x113's S_T, because the
  both-accepted mask drops only the 2 to 7 units per arm that the second judge
  rejects and the first accepts. On x108's S it drops 13 units, the
  both-accepted column is the smaller of the two (+0.006 to +0.099 against
  +0.012 to +0.106), and five of its eight intervals exclude zero.
- **The controls stay flat under either judge.** C is −0.003 to +0.015 and C_T
  +0.015 to +0.037 on the both-accepted mask, against −0.007 to +0.018 and
  +0.016 to +0.049 on the second judge's own accepted pairs — the same +0.01 to
  +0.04 band the owning experiments report, next to +0.02 to +0.20 for the
  nominated-word arms.

## Reading

- **An independent judge confirms the majority of the benign and harmful
  substitutions and a third of the table-guided ones.** 0.605, 0.581 and 0.330
  of the pairs are read as the same request by a model from another family that
  did no drafting, with the inert-word controls at 0.81–0.86 and a test–retest
  kappa of 0.889.
- **A second judge from a third family reads the same ordering more
  permissively.** 0.789, 0.707 and 0.527 for the three substitution arms, 0.95
  for the controls and 0.11 for the hedge deletions; the two judges agree on
  0.800 of the 4,803 pairs both labelled (kappa 0.578) and disagree almost
  entirely in one direction. The level is reader-dependent and the ordering is
  not: the two put the five substitution and control arms in the same order
  while differing by 0.113 to 0.197 on how many pairs each accepts.
- **The in-place substitution effect does not depend on the pairs the judge
  rejects.** Restricted to accepted pairs, replacing the nominated word removes
  +0.02 to +0.19 of `refusal2` across the eight models in all three
  experiments, with 21 of the 24 intervals excluding zero, while the matched
  inert control stays at +0.01 to +0.04. The same holds on the pairs **both**
  judges accept (+0.02 to +0.20, 19 of 24 intervals excluding zero) and on the
  second judge's own larger subset, where all 24 exclude zero (+0.02 to
  +0.21).
- **The scale of the edit orders both the acceptance and the effect.** Inert
  synonym, drafter-best synonym, table-lowest synonym, phrase deletion: the
  acceptance share falls (0.86, 0.61, 0.33, 0.09) and the refusal removed rises
  along the same order. x113's table rule, which deliberately takes the
  lowest-shift admissible candidate, buys its extra effect in part by moving the
  request further — consistent with [x116](x116-random-synonym-control.md),
  where a random admissible synonym already removes 0.67–0.89 of S_T's effect.
- **Hedge phrases are scope, not vocabulary.** Every arm that edits a hedge is
  accepted at 0.05–0.19, and the judge's reason is the dropped condition. The
  hedge contribution to the published R and S numbers is a change in what the
  request asks for, and the content-word substitutions are where the lexical
  claim rests (x106 S content 0.667 accepted, x113 S_T content 0.397).

## Deviations from the plan

- The plan estimated about 5,000 calls; after deduplication across models the
  set is 4,805 distinct pairs (7,713 owner uses), and the retest adds 495.
  Total spend $0.436.
- x108's contrasts are recomputed as well as x106's and x113's (the plan named
  only the latter two); the x108 cohort has no C arm, so its table carries S
  and R.
- Acceptance shares are reported over owner rows, so a text two models share is
  counted once per model. The distinct-pair shares are in `results.json`
  (`by_arm_distinct_pairs`: S 0.397, C 0.814, R 0.088).
- x113's `same_as_x106` flag is kept per owner rather than used to drop pairs:
  a shared text is judged once and its label used for both owners.
- The second judge was added after the first pass was complete, on the same
  pairs and protocol. Its distinct-pair shares are in `results_haiku.json`
  (`by_arm_distinct_pairs`: S 0.586, C 0.954, R 0.113). Its route carries no
  `temperature` and no `seed`, so its repeatability is a second call rather
  than a second seed, and two pairs carry no label from it. Second-judge spend
  $0.586, bringing x118 to **$1.022** over 10,600 judge calls.

## Artifacts

- Pairs: `data/runs/x118/pairs.jsonl` (4,805 rows; sha256
  `5c723728…`) and
  `pairs_summary.json` (counts per source and arm, and the three input cohorts
  with their frozen sha256: x106 `e06327cf…`, x108 `7d64a494…`, the eight x113
  `texts_sha256`).
- Judge: `data/runs/x118/judge/meaning.jsonl` (4,805 labelled rows with the
  model, the parameters, the usage and the cost per row),
  `judge/meaning_retest.jsonl` (495 rows at seed 1042),
  `judge/cost_log.jsonl` (six passes: first judge smoke 20 / $0.0015, full
  4,785 / $0.3938, retest 495 / $0.0401; second judge smoke 20 / $0.0022, full
  4,785 / $0.5291 with 2 content-filter refusals, retest 495 / $0.0547).
- Second judge: `judge/meaning_haiku.jsonl` (4,803 labelled rows plus the 2
  refused, each row carrying the model, the accepted parameters, the serving
  provider, the usage and the cost) and `judge/meaning_haiku_retest.jsonl`
  (the same 495 pairs, judged again).
- Second-judge results: `data/runs/x118/results_haiku.json` (the same blocks
  plus `other_judge`, `judge_agreement` and
  `unlabelled_in_effect_masks`, with every effect on four masks),
  `results_haiku.md`.
- Results: `data/runs/x118/results.json` (`judge`, `acceptance`,
  `human_agreement` with every human-passed pair the judge rejected,
  `test_retest`, `effects.x106` / `.x113` / `.x108` per model and arm with both
  masks), `results.md`.
- Code: `code/x118_collect_pairs.py`, `code/x118_judge.py` (`--limit`,
  `--retest`, `--seed`, `--threads`, `--cap`, `--effort`, `--judge-model`,
  `--out`, `--retest-ids`, `--probe`),
  `code/x118_analyze.py` (`--n-boot`, `--judge-file`, `--retest-file`,
  `--compare-with`, `--out-stem`, `--allow-unlabelled`). The OpenRouter client, the balanced-JSON
  parser, the key loader and the cost meter are x81's; the `refusal2` labels
  are x106's, x108's and x113's, read, not recomputed.
- Spend: $0.436 over 5,300 first-judge calls (1.57 M input, 0.48 M output
  tokens, 0.25 M of them reasoning) and $0.586 over 5,300 second-judge calls
  (3.57 M input, 0.46 M output tokens, 2.5 k of them reasoning); $1.022 in all.
  Both figures are the provider costs OpenRouter reported per call.
  No GPU time.
