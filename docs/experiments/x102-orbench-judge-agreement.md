# x102: independent judges reproduce the Qwen3.5-4B OR-Bench verdicts — binary κ 0.88–0.89 against three API judges, Qwen3Guard, GuardReasoner and WildGuard, every reported delta keeps its sign, and the incumbent agrees with XSTest's human labels at 91.7% (κ 0.83)

Parent: [Experiment index](README.md). Measurement validation of the
OR-Bench judge stack used by [x87](x87-external-safety-overrefusal-eval.md),
[x88](x88-eight-model-deep-dose-transfer.md) and
[x89](x89-positive-dose-external-extension.md): a fresh unedited
`Qwen/Qwen3.5-4B` under the x87 rubric, `refusal = not direct_answer`
(the "incumbent"). The judge survey that chose the comparison judges is
[refusal-judges.md](../refusal-judges.md). The `refusal2` stack of the
keyword experiments is a different instrument, validated in
[x58](labels-refusal-comply-x58.md), and is not touched here.

On a frozen sample of 300 OR-Bench prompts (200 Hard-1K, 100 Toxic) across
the 35 model-conditions the draft reports (10,500 responses), three API
judges under the benchmark's own verbatim prompt (DeepSeek-V4-Flash,
GLM-5.3-Flash, GPT-5.6 Luna, all via OpenRouter) and three open-weight
refusal judges under their native formats (Qwen3Guard-Gen-4B,
GuardReasoner-8B, WildGuard) re-judged every response; every judge and the
incumbent also judged XSTest's 2,250 human-labelled completions.

**Findings.**

- **Binary agreement with the incumbent is 94.5–95.3% (Cohen's κ
  0.877–0.894) for every arm**, the same range the arms reach among
  themselves (95.4–97.2%, κ 0.897–0.937); Fleiss' κ over the incumbent and
  the three API judges is 0.899 [0.892, 0.905].
- **The three-way label is where the instruments part, not the binary
  verdict.** Three-way κ between the incumbent and the API judges is
  0.49–0.54: the incumbent's `direct_refusal` rows go to
  `indirect_refusal` under the benchmark prompt (DeepSeek: 2,832 of 4,565),
  and that move does not cross the refusal boundary.
- **Against the API majority (Reference A), the incumbent's sensitivity for
  refusal is 0.978 [0.975, 0.982] and its specificity 0.903 [0.894, 0.913]**, and against WildGuard alone
  (Reference B) 0.980 [0.976, 0.983] and 0.880 [0.870, 0.891]: when it errs, it calls refusal on a response the other judges
  read as an answer.
- **Differential error is null on Qwen3.5-4B and on the boundary write
  everywhere, and present on the rank-one edit in the two smallest Qwen
  models.** For every main-table condition of Qwen3.5-4B the paired
  edited − base change in incumbent-vs-reference disagreement is within
  ±0.015 with an interval covering zero; the boundary write (`x86_m150`)
  never exceeds +0.03 on any model (+0.04 against Reference B). The rank-one edit on Qwen3.5-0.8B and
  2B raises the disagreement rate by +0.09 to +0.145 (Hard-1K) and +0.08
  to +0.10 (Toxic), intervals excluding zero: the incumbent reads those
  models' degraded deep-edit outputs — rambling text that does attempt the
  question — as `indirect_refusal`, where the other judges read
  `direct_answer`. The incumbent therefore **understates** those drops
  (Qwen3.5-0.8B `x75_m6` Hard-1K: incumbent −0.015; the three API
  judges, GuardReasoner, WildGuard and Reference A −0.12 to −0.145;
  Qwen3Guard +0.005). Reference B gives the same partition: Qwen3.5-4B
  within ±0.02 everywhere, and the rank-one edit on Qwen3.5-0.8B / 2B at
  +0.07 to +0.21 with intervals excluding zero.
- **No instrument reverses an edited-minus-base delta.** In every one of
  the 54 model × condition × benchmark cells the instruments' deltas share
  the incumbent's sign or one side is within ±0.01 of zero. The other
  instruments' deltas fall inside the incumbent's own sample interval in
  53 (GuardReasoner), 52 (DeepSeek), 51 (GLM), 51 (Qwen3Guard), 51
  (Reference A), 49 (Luna) and 46 (WildGuard) cells; the 25
  instrument-cells outside are eighteen on the Qwen3.5-0.8B / 2B rank-one
  cells above, where the other judges see a larger drop (Qwen3Guard a
  smaller one on the 0.8B `x75_m6` cells), Llama-3.2-3B `x75_m6` Toxic
  under WildGuard (−0.47 against −0.37), Qwen3.5-27B `x75_m6` Hard-1K
  under GLM and Reference A (−0.10 against −0.05), Phi-4-mini `x75_m3`
  Toxic under GLM (0.00 against −0.05), and three Toxic cells with a
  ±0.01 delta against an incumbent delta of exactly zero.
- **Absolute `base` rates agree within ±0.02 of Reference A and within
  ±0.03 of WildGuard** on every model and benchmark except Qwen3.5-0.8B
  Hard-1K (incumbent 0.445 against Reference A 0.400, +0.045 [+0.015,
  +0.080], and against WildGuard 0.375, +0.070 [+0.035, +0.105]).
- **On XSTest the incumbent agrees with the human label on 91.7%
  [90.6, 92.9] of 2,250 completions, κ 0.834 [0.812, 0.857]** (partial
  refusal counted as refusal; sensitivity 0.923, specificity 0.911; it
  calls 82% of the 176 partial refusals refusal). The API judges land at
  90.8–92.6% (κ 0.816–0.853), Qwen3Guard at 92.0% (κ 0.839), WildGuard at 94.0%
  (κ 0.881; sensitivity 0.939, specificity 0.942) and GuardReasoner at 95.1% (κ 0.901; sensitivity 0.959, specificity 0.942,
  6 unparsed); XSTest's
  own released GPT-4 labels reach 82.5% (κ 0.654) and its string match
  88.4% (κ 0.770). With partial refusals excluded the incumbent is at
  92.6% (κ 0.851), WildGuard at 95.3% (κ 0.906) and GuardReasoner at 96.2%
  (κ 0.925).
- **No voting combination of the seven instruments beats GuardReasoner
  alone on XSTest.** On the 2,244 completions every instrument parsed,
  141 combinations place GuardReasoner-8B alone second at 95.1% (κ 0.901); the one
  combination above it, "GLM or GuardReasoner", reaches 95.2% (κ
  0.904; paired difference +0.0013 [−0.0022, +0.0053]), and the
  best combination without GuardReasoner, "GLM or WildGuard", 94.3%
  (κ 0.887). Under 20 repeats of 5-fold cross-validation the
  combination with the highest training agreement gains −0.0004 (sd
  0.0033) over GuardReasoner alone on the held-out fold. XSTest's two
  annotators agree on the binary label on 97.9% of completions (κ 0.957).
- **Within-arm repeatability** on 300 re-sent requests: binary verdict
  98.3% (DeepSeek), 97.7% (GLM), 97.3% (Luna); three-way 95.3 / 97.0 /
  88.3%.

Result file: `data/results/x102_judge_agreement.json` (every number above;
section names in the analyzer's docstring); figure
`data/results/x102_judge_agreement_deltas.{png,pdf}`.

## Agreement tables

Table: [Binary verdict agreement between instrument pairs on the 10,500 OR-Bench rows](json/x102-orbench-judge-agreement__binary-agreement.json) — agreement and Cohen's κ for the incumbent against each arm and for the arms among themselves; incumbent pairs span 0.945–0.953 (κ 0.877–0.894), arm pairs reach 0.972 (WildGuard – Qwen3Guard).

Three-way confusion of the incumbent (rows) against DeepSeek (columns), in
the order `direct_answer` / `direct_refusal` / `indirect_refusal`:
[3156, 5, 199] / [139, 1594, 2832] / [201, 88, 2286]. GLM and Luna show the
same shape (GLM moves 3,243 of the incumbent's `direct_refusal` rows to
`indirect_refusal`, Luna 2,439). On the incumbent's own `indirect_refusal`
stratum (2,575 rows) the arms call refusal on 88–92% of rows.

Table: [Incumbent sensitivity / specificity for refusal against Reference A and Reference B, by model](json/x102-orbench-judge-agreement__sens-spec-by-model.json) — Hard-1K with Toxic in parentheses; specificity is the weaker side, lowest on Qwen3.5-0.8B Hard-1K (0.850 against A, 0.812 against B).

Rows where the incumbent stands alone against a unanimous API verdict:
275 of 10,500 (196 incumbent-refusal, 79 incumbent-answer); the largest
cells are Qwen3.5-0.8B `x75_m6` Hard-1K (29), Qwen3.5-0.8B `x75_m3`
Hard-1K (20) and Qwen3.5-2B `x75_m3` Hard-1K (14). The 30 most split rows
are written out in full in the result file (`disagreement_anatomy`).

## Differential error (Reference A and Reference B)

Table: [Differential error against Reference A](json/x102-orbench-judge-agreement__differential-error-ref-a.json) — paired edited − base change in incumbent-vs-Reference-A disagreement on the same prompts, per model and condition; the largest starred rises are the Qwen3.5-0.8B / 2B rank-one edits (up to +0.145, 0.8B `x75_m6` Hard-1K), and `x86_m150` never exceeds +0.030.

Base-condition disagreement rates against Reference A are 0.015–0.06 on
Hard-1K and 0.00–0.04 on Toxic.

Base-condition disagreement rates against Reference B (WildGuard) are
0.015–0.070 on Hard-1K and 0.000–0.030 on Toxic.

Table: [Differential error against Reference B (WildGuard)](json/x102-orbench-judge-agreement__differential-error-ref-b.json) — the same change against WildGuard; the largest starred rises are again the Qwen3.5-0.8B / 2B rank-one edits (up to +0.210, 0.8B `x75_m6` Hard-1K), and `x86_m150` reaches at most +0.040.

## Voting combinations on XSTest

Every voting combination of the seven instruments (incumbent, DeepSeek,
GLM, Luna, Qwen3Guard, GuardReasoner, WildGuard), scored against the
human binary label (partial refusal = refusal) on the 2,244
completions every instrument parsed (the 6 GuardReasoner
repetition loops dropped): each instrument alone (7), "both" (refusal
only if both say refusal) and "either" (refusal if either says refusal)
of every pair (42), the majority of every subset of size 3 (35), 5 (21)
and 7 (1), and the majority of every subset of size 4 with a tied vote
abstaining and the row dropped from that combination's statistics (35);
141 combinations, ranked by agreement in
`xstest_combinations.combinations` of the result file.

Table: [Selected voting combinations on XSTest, by rank](json/x102-orbench-judge-agreement__xstest-combinations.json) — 13 of the 141 combinations with n, agreement, κ, sensitivity and specificity; GuardReasoner alone ranks second at 0.951, and Reference A ranks 116th at 0.919.

17 combinations lie within ±0.005 of GuardReasoner's 0.951, and every one of
them contains GuardReasoner. The 4-way majorities abstain on 35–90 rows each.

Cross-validation (20 repeats of 5-fold, seed 10201, the 35 abstaining
4-way majorities excluded, 106 candidates): in each of the 100 folds the
combination with the highest training-fold agreement, evaluated on the
held-out fold, gains −0.0004 (sd 0.0033; range
−0.0089 to +0.0045) over GuardReasoner alone. Selected: GLM or GuardReasoner in 80
folds, GuardReasoner alone in 11, WildGuard or GuardReasoner in 8,
DeepSeek or GuardReasoner in 1.

Table: [Agreement with the human label by XSTest stratum](json/x102-orbench-judge-agreement__xstest-by-stratum.json) — per-instrument agreement on the 2,244 rows by full compliance, full refusal and partial refusal; GuardReasoner is highest on full refusal (0.985) and the incumbent on partial refusal (0.818).

XSTest's two annotators agree on the three-way label on 96.4% of the
2,250 completions.

## Design, as run

- **Sample.** 200 Hard-1K and 100 Toxic OR-Bench ids drawn uniformly
  within benchmark from the 1,974 shared ids with `rng(102)`; the same 300
  ids for all 35 model-conditions — Qwen3.5-4B `base`, `x86_full`,
  `x75_m3`, `x75_p3`, `x86_p050`, `x75_m6`, `x86_m150` and the seven
  transfer models × `base`, `x75_m3`, `x75_m6`, `x86_m150`. The builder
  re-verified every judgment file's recorded generation sha256 against
  the generation file next to it and the row-id coincidence of all 49
  source files; the sample's sha256 over its sorted row keys
  (`befa0263…`) is registered in the manifest and every judge stage
  refuses an unregistered sample. Repeat subsample: 300 rows, `rng(1102)`;
  pilot: 100 rows, `rng(2102)`.
- **Incumbent.** The existing x87/x88/x89 verdicts, taken as they are
  (10,500 rows: 4,565 `direct_refusal`, 2,575 `indirect_refusal`, 3,360
  `direct_answer`). Its one new pass is XSTest: the x87 scorer loop
  (rubric, greedy, 512 new tokens, same-judge retry at 32 tokens) on the
  same `851bf6e8…` snapshot, batch 8, on one B200; 2,174 bracketed and
  76 bare-final-line verdicts, none unparsed, 5 minutes.
- **API arms.** `response_checking_prompt` of
  `response_checker/check_response.py` at or-bench commit `4fad1f9e…`,
  byte for byte (sha256 `58b298dc…`, asserted by the tests), as one user
  message with no system prompt; `seed 42`, `max_tokens 512`,
  `temperature 0` for DeepSeek and GLM (the Luna slug does not accept
  `temperature`). Reasoning: `effort: "none"` for `deepseek/deepseek-v4-
  flash-0731` and `openai/gpt-5.6-luna` (0 reasoning tokens on every
  call); `z-ai/glm-5.3-flash` rejects `none` ("Reasoning is mandatory for
  this endpoint") and ran at `minimal`, billing a median of 26 reasoning
  tokens on 74% of calls. Every reply stored with two readings: the
  benchmark's parser (last `[[…]]`, `direct_refusal` by default) and the
  x87 parser with x87's retry conversation on failure; the two readings
  agree on every parsed row. Unparsed after retry: DeepSeek 0, Luna 0,
  GLM 12 of 10,500 (0.11%: replies that spent the whole budget reasoning)
  and 0 of 2,250 on XSTest. 24 threads per arm; main pass 9–21 minutes
  per arm. Spend: USD 3.50 for 39,420 calls (DeepSeek 0.61, GLM 0.92,
  Luna 1.97), well inside the per-arm caps (15 / 10 / 15).
- **Open-weight arms** (the cluster, one B200, bf16, greedy, each model's own
  prompt and refusal field, resumed by row key). Qwen3Guard-Gen-4B
  (revision `6ec42827…`): the model card's response-moderation chat
  template, `Refusal: Yes/No`, 64 new tokens (mean 13.7), batch 16,
  parse rate 100%, 15 + 3 minutes. GuardReasoner-8B (revision
  `f3f57a9c…`): the model card's instruction as raw text with BOS,
  `Completion: refusal/compliance` after the reasoning trace, 1,024 new
  tokens (mean 295 on the sample, 289 on XSTest), batch 48, `generate`
  stopping on both the tokenizer's `<|eot_id|>` and the generation
  config's `<|end_of_text|>`; 21 of 10,500 and 6 of 2,250 rows are
  repetition loops that reach the cap before an `Answers:` block and are
  unparsed; 67 + 15 minutes. WildGuard (Reference B, revision
  `cbba4823…`): the official `WILDGUARD_INPUT_FORMAT` with BOS, `Response
  refusal: yes/no`, 64 new tokens (mean 20.0), batch 16, parse rate 100%
  on both passes, 17 + 3 minutes; its `tokenizer.model` is served by
  transformers 5.15 through `sentencepiece`, installed into the venv for
  this arm.
- **XSTest.** All 2,250 completions of the five models at commit
  `d7bb5bd7…` (sha256 of every raw file in `x102_xstest.json`), human
  `final_label` → full refusal = refusal, full compliance = compliance,
  partial refusal = refusal (primary) or excluded (sensitivity); the
  repository's released GPT-4 labels (three-way; eleven free-text verdicts
  on `mistralinstruct` kept as `malformed` and excluded) and string-match
  labels scored as published baselines on the same axis.
- **Analysis.** Rows an arm left unparsed are dropped from that arm's
  statistics and counted (no arm above 0.2%). Reference A: majority of
  the three API binaries, a row with one missing vote decided by the two
  remaining only if they agree (1 row excluded). Reference B: WildGuard
  alone. Deltas paired by OR-Bench id within a model; all intervals
  percentile bootstraps over rows, 10,000 resamples, seed 10201.

## Artifacts

- Code: `code/x102_common.py` (sample roster, arm registry, the verbatim
  benchmark prompt with both parsers, open-judge formats and parsers,
  label mappings, κ / majority / bootstrap helpers),
  `code/x102_build_sample.py`, `code/x102_fetch_xstest.py`,
  `code/x102_judge_api.py`, `code/x102_judge_open.py`,
  `code/x102_judge_incumbent_xstest.py`, `code/x102_analyze.py`,
  `code/x102.sbatch`; tests `code/tests/test_x102_judge_agreement.py`
  (23).
- Data: `data/runs/x102/x102_sample.json`, `x102_xstest.json`,
  `x102_manifest.json` (settings as used, pilot results, snapshot
  revisions, per-pass spend), `x102_judgments_<arm>.jsonl`,
  `x102_pilot_<arm>.jsonl`, `x102_repeat_<arm>.jsonl`,
  `x102_xstest_<arm>.jsonl`, `x102_xstest_incumbent.jsonl`.
- Result: `data/results/x102_judge_agreement.json`,
  `data/results/x102_judge_agreement_deltas.{png,pdf}`.
