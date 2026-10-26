# x87: both refusal-reducing edits cut OR-Bench Hard-1K refusal on external benchmarks, with a mixed verdict on the safety and instruction-following guards

Parent: [Experiment index](README.md).

> **Margin correction, 2026-09-09.** The non-inferiority margin is **0.02**
> on each guard endpoint, not the 0.05 this file and the result
> JSON carried until now; by author decision the 0.05 was a transcription
> error. Only verdicts moved. Every delta, interval, one-sided bound, base
> rate, count and alpha below is the original seeded bootstrap output,
> unchanged — the bootstrap seed is fixed, so re-deciding the gates against
> 0.02 is arithmetic on numbers that were already recorded. The gate
> booleans, `claim_pass` and `successful_candidates` in
> `data/results/x87_external_safety_utility.json` were recomputed
> accordingly (that file now carries a `corrections` entry), and the same
> correction was applied to the x89 dose extension.

The interventions under test are owned by
[x86-crossed-keyword-boundary-intervention.md](x86-crossed-keyword-boundary-intervention.md)
and [x75-boundary-weighted-refusal-edit.md](x75-boundary-weighted-refusal-edit.md);
x87 only evaluates them on external benchmarks.

On one frozen `Qwen/Qwen3.5-4B` snapshot (`851bf6e8…`, the x86 snapshot),
x87 compares four conditions — `base`, the x86 negative-full-dose consensus
boundary write (`x86_full`, dose −0.5), and a Qwen-native port of x75's
`refusal_weighted` rank-one edit at wrapper scales −3/+3 (`x75_m3`,
`x75_p3`) — on OR-Bench Hard-1K (1,319 seemingly-toxic benign prompts),
OR-Bench Toxic (655), StrongREJECT full (313), and IFEval (541), all
prompt-paired against `base` with 20,000-resample bootstraps.

**Both candidates clear the Hard-1K superiority endpoint; neither clears all
three guards** (Hard-1K superiority at one-sided α = 0.05/3; three
non-inferiority endpoints at one-sided α = 0.05/9, absolute margins 0.02):

Table: [Primary gates: edited − base at deployment dose, with gate-relevant bounds and verdicts](json/x87-external-safety-overrefusal-eval__primary-gates.json) — `x86_full` (−0.032) and `x75_m3` (−0.078) pass Hard-1K superiority; `x86_full` fails the Toxic and StrongREJECT guards, `x75_m3` the Toxic and IFEval guards, and the signed control `x75_p3` raises Hard-1K +0.058.

Base rates: Hard-1K refusal 0.880, Toxic refusal 0.989, StrongREJECT score
0.063, IFEval prompt-strict 0.804. Result file:
`data/results/x87_external_safety_utility.json` (`successful_candidates:
[]`).

**Exactly which bound lands outside the margin.** Four of the six
deployment-dose guard bounds (two candidates × three guards) fall outside
0.02: Toxic refusal under *both* candidates (lower bounds −0.023 for
`x86_full`, −0.031 for `x75_m3`), StrongREJECT under the boundary write
(upper +0.024), and IFEval under the rank-one edit (lower −0.039). Formal
non-inferiority holds only for the write on IFEval and the edit on
StrongREJECT. The signed control `x75_p3` additionally misses on
StrongREJECT (+0.041) and IFEval (−0.042), and by construction fails
superiority. Every deployment-dose point estimate sits inside the margin
(the largest adverse one is Toxic's −0.017); the Toxic drops −0.011 and
−0.017 have two-sided intervals excluding zero.

The two edits move Hard-1K with different profiles: `x75_m3` moves it more
than twice as far (−0.078 vs −0.032) at similar safety cost; `x86_full` is
the lighter-touch activation write. The signed control behaves as a mirror:
`x75_p3` raises Hard-1K refusal +0.058 and StrongREJECT +0.022, so
direction, not merely perturbation, drives the effect.

## Design, as run

- **Frozen inputs.** `x87_manifest.json` pins the model snapshot manifest,
  OR-Bench revision `e36d8b80…` (hard-1k + toxic), StrongREJECT data commit
  `f7cad6c1…`, lm-eval 0.4.12, and every prompt hash (2,287 generation rows).
  Generation: chat template with thinking disabled, greedy, ≤1,024 new
  tokens, batch 8, left padding; the two-token `</think>`/`"\n\n"` boundary
  (ids 248069, 271) audited on every rendered prompt.
- **`x86_full`.** Mean of the four frozen x86 true fold directions at
  state 30, written at the two boundary positions
  by a prefill-only hook at dose −(admitted a*) = −0.5, directions verified
  by sha256 against the x86 KL calibration artifact.
- **Qwen-native x75 port** (stage 0, all under `data/runs/x87/`): the x75
  `refusal_weighted` chain rerun end-to-end on the `qwen_A15k_v4_full`
  cache with Qwen v4 refusal labels and the x66 seed-42 split
  (config `x87_x75_qwen_config.json`; 13,527 harmful + 10,500 benign rows).
  Probe and adapter-selection values are owned by
  [x75](x75-boundary-weighted-refusal-edit.md#the-qwen-native-port-the-paper-reports).
  Rank-one adapter (`x75_qwen_refusal_weighted_true.pt`): state 31. The checked-in Llama
  x75 adapters were never touched; `x87_common.validate_x75_adapter` fails
  closed on model, probe kind, and arm, and `x87_generate` rejects a
  snapshot-name mismatch. `m3`/`p3` are wrapper scales ±3 on this adapter.
- **Scoring.** OR-Bench: the benchmark's three-way rubric
  (`direct_answer` / `direct_refusal` / `indirect_refusal`), judged by a
  fresh unedited copy of the same Qwen snapshot, greedy;
  refusal = not-`direct_answer`. StrongREJECT: official
  `strongreject_finetuned` evaluator (gemma-2b + LoRA
  `qylu4156/strongreject-15k-v1`), local inference, all 313 rows in [0,1].
  IFEval: lm-eval-harness `ifeval` with chat template, prompt-level strict
  accuracy primary. No external LLM API anywhere in x87.
- **Analysis.** Prompt-paired deltas, `rng(8701)` bootstrap (20,000), gates
  as defined above; `x75_p3` is the signed control, fully reported.

## Dose extension (added 2026-08-24)

Three further dose points through the identical pipeline — `x86_m075`
(dose −0.75 = 1.5·a*), `x86_m100` (−1.0 = 2·a*) and `x75_m4` (wrapper
scale −4). In the result file they carry `role: post_hoc_dose_extension`;
the primary comparisons, seeds and alphas are byte-identical to the primary
analysis (adding them left `successful_candidates` untouched; that field
changed only later, with the 2026-09-09 margin correction).

Table: [Dose extension: edited − base at `x86_m075`, `x86_m100` and `x75_m4`](json/x87-external-safety-overrefusal-eval__dose-extension.json) — Hard-1K refusal −0.052 / −0.059 / −0.086; `x86_m075` carries a StrongREJECT upper bound of +0.059.

Reads:

- **x86 dose-response continues but saturates.** Hard-1K refusal falls
  −0.032 → −0.052 → −0.059 across doses 0.5/0.75/1.0 with shrinking
  marginal gain; IFEval stays flat-to-positive throughout. *(Superseded
  by the second wave below: the flattening at −0.75/−1.0 was local, the
  response resumes falling through −3.)*
- **Safety cost appears beyond the admitted dose.** `x86_m075` carries the
  largest StrongREJECT upper bound anywhere in the x86 family (+0.059) —
  outside the 0.02 margin by a wide mark, and outside even the 0.05 figure
  this file used before the correction. `x86_m100`'s StrongREJECT
  (+0.009) is lower than `x86_m075`'s point estimate and the two intervals
  overlap. *(Superseded by the second wave: no deeper x86 dose reproduces
  the crossing, so the −0.75 excursion reads as endpoint noise; the
  monotone safety channel is Toxic refusal.)*
- **`x75_m4` buys little over −3 and starts to bite.** Hard-1K −0.086 vs
  −0.078 at −3, while Toxic slips to −0.023 and StrongREJECT flips from
  −0.015 to +0.017. −3 is close to this edit's useful edge.
- **Benign-KL context** (`data/runs/x87/x87_x75_qwen_dose_kl.json`, x71 KL
  definition, budget 0.02): the Qwen adapter crosses the budget between
  |2| and |3| (−2: 0.018, −3: 0.039, −4: 0.068; +2/+3/+4: 0.026/0.059/0.105,
  asymmetric) — earlier than the Llama x71 adapter's |3|–|4| crossing. So
  the passing `x75_m3` already runs at benign KL 0.039, the same order as
  `x86_full`'s 0.031.

### Second wave (later on 2026-08-24): x86 −1.25/−1.5/−2/−3, x75 −5/−6/−7

Seven more suppression doses through the identical pipeline, appended to `EXTENSION_CONDITIONS` so every earlier
bootstrap seed index is unchanged — verified by re-analysis: the six
pre-existing comparisons and the first GSM8K block are byte-identical, and
`successful_candidates` was still `["x86_full", "x75_m3"]` under the 0.05
figure then in force (it is empty under the corrected 0.02 margin).

Table: [Second wave: edited − base at x86 doses −1.25/−1.5/−2/−3 and x75 scales −5/−6/−7](json/x87-external-safety-overrefusal-eval__second-wave.json) — Hard-1K refusal falls to −0.211 at x86 −3 and −0.178 at x75 −7; Toxic refusal reaches −0.052 (x86 −3) and IFEval −0.083 (x75 −7).

Reads:

- **The x86 response does not saturate.** After the local flattening at
  −0.75/−1.0, Hard-1K refusal resumes falling: −0.089 → −0.100 → −0.145 →
  −0.211 across −1.25/−1.5/−2/−3. The wave-1 "saturates" read is
  superseded.
- **The x86 write's capability profile stays flat far past any admitted
  dose.** IFEval is flat-to-positive and GSM8K null even at −3 — six
  times a*, where the calibration grid stops at −1.0 (KL 0.128) and
  extrapolates past 1.0. Spot-checked −3 responses are fully coherent
  prose, and IFEval/GSM8K would both crater on degenerate text, so the
  Hard-1K drop is not a degeneration artifact.
- **The wave-1 StrongREJECT excursion does not replicate.** Deeper x86
  uppers are +0.034/+0.032/+0.045/+0.023 — all outside the 0.02 margin,
  but none approaches `x86_m075`'s +0.059, and the point estimates stay
  near base, so the −0.75 spike reads as endpoint noise rather than a
  dose effect. The cost that does scale with dose is OR-Bench Toxic
  refusal: −0.020 → −0.052 across −1.25…−3, with the point estimate
  crossing the 0.02 margin from −2 onward (−0.031, then −0.052). The same
  holds for x75, which crosses at every deep scale tested (−0.028 →
  −0.047 across −5…−7). Deep suppression buys
  over-refusal relief by also releasing refusal on genuinely toxic
  prompts, while StrongREJECT-style harmfulness stays near base.
- **The x75 wrapper hits a capability wall at −5 that the x86 write
  never shows.** IFEval −0.022 / −0.039 / −0.083 (lower bound −0.131 at
  −7) while GSM8K stays null throughout; benign KL climbs 0.108 / 0.168 /
  0.260 (5–13× budget; full curve now −7…+4 in
  `data/runs/x87/x87_x75_qwen_dose_kl.json`). Strict instruction
  following is the first casualty of the rank-one edit, well before math.
- **Judge health is unchanged at extreme doses:** ~3% bare-line label
  fallback per condition (same as the primary run), four single
  `bracketed_retry` calls total, all resolved by the same-judge retry.

## GSM8K capability check (added 2026-08-24)

Capability context for all edited conditions
(`data/results/x87_gsm8k_posthoc.json`, `role:
post_hoc_capability_extension`). Same frozen
snapshot, manifest, and condition setup as the primary run;
lm-eval-harness `gsm8k` (5-shot as multi-turn chat, thinking off,
greedy, seeds 8700), all 1,319 test questions, question-paired
bootstrap deltas with 95% intervals.

Base accuracy 0.7877 (strict-match) / 0.8044 (flexible-extract).

Table: [GSM8K exact-match deltas (edited − base, 95% intervals) for the 13 edited conditions](json/x87-external-safety-overrefusal-eval__gsm8k.json) — every strict-match and flexible-extract delta lies within ±0.011; only `x86_m150` (strict +0.0076 [+0.0015, +0.0136]) excludes zero.

Every delta is within ±0.011 (13 edited conditions, second wave
included). All intervals include zero except two positive
blips (`x86_m150` strict +0.0076 [+0.0015, +0.0136] and its
flexible-extract twin). The
mechanism differs by family: the x86 boundary write flips only 12–28 of
1,319 strict-match answers even at −3 (balanced signs), while the x75
wrapper perturbs more — 104–154 flips (~8–12%) — that cancel to a null
net effect.

## Deviations (recorded 2026-08-24; none touch generation or the rubric)

- **OR-Bench judge label parsing.** The judge prompt asks for a
  `[[label]]` ending; the Qwen judge emits the label bare (no brackets) in
  ~4% of calls. The parser accepts, in order: exactly one bracketed label;
  else the final non-empty line equal to exactly one label
  (`label_parse: "bare_final_line"`, per-row auditable). Anything else
  still fails closed — no response-keyword heuristic exists anywhere.
  Judge budget raised 160→512 new tokens; scoring is batch-checkpointed
  and resumes by generation sha256. Parse mix per condition:
  base 1902/72, x86_full 1900/74, x75_m3 1904/70, x75_p3 1880/93
  (bracketed/bare).
- **One ambiguous verdict, resolved by the same judge.** Exactly 1 of
  7,896 judge calls (an `x75_p3` row) produced an equivocating explanation
  with no label at all. The scorer replays the rubric conversation and
  asks the judge to end its verdict with only the bracketed label
  (`label_parse: "bracketed_retry"`, both replies stored); a second
  failure would still stop the run.
- **StrongREJECT evaluator loaded by local snapshot path.** transformers
  5.15's offline mode cannot represent "config.json does not exist on the
  hub", so loading the adapter repo by id fails on air-gapped compute
  nodes. `x87_score.preload_strongreject_evaluator` resolves the pinned
  snapshot from the local HF cache (`snapshot_download` offline) and
  injects the identically-loaded model/tokenizer into
  `strong_reject.evaluate.cached_models`; all scoring stays in upstream
  strong_reject code, and the artifact records `evaluator_snapshot`.

## Artifacts

- Manifest, per-condition generations, IFEval, OR-Bench and StrongREJECT
  scores: `data/runs/x87/x87_{manifest,generations_*,ifeval_*,orbench_*,strongreject_*}.json`.
- x75 Qwen port: `data/runs/x87/x75_qwen_{weight_audit.json,probes/,refusal_weighted_true.pt,refusal_weighted_true.json,refusal_weighted_true.jsonl}`
  and `x87_x75_qwen_config.json`. The 2.2 GB prepared tensor
  (`x75_qwen_prepared/`) stays on the cluster scratch, regenerable by
  `code/x71_prepare_dataset.py`.
- Analysis: `data/results/x87_external_safety_utility.json` (sole gate
  authority).
- GSM8K: `data/runs/x87/x87_gsm8k_<condition>.json` (logged
  samples slimmed of `doc`/`arguments`; full copies on the cluster scratch as
  `x87_gsm8k_full_*.json`) and `data/results/x87_gsm8k_posthoc.json`;
  code `code/x87_gsm8k.py`, `code/x87_gsm8k_analyze.py`.
- Code: `code/x87_{common,build_manifest,generate,ifeval,score,analyze}.py`;
  tests `code/tests/test_x87_common.py` (15). The Qwen-port enablers
  (`code/lib/edit_common.py` decoder-resolution fallback and multimodal
  wrapper load; `code/x71_train_edit.py --model-source`) are shared-chain
  changes committed with this experiment.
