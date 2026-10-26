# x89: positive boundary-write doses and shallow rank-one scales, externally

Parent: [Experiment index](README.md).
**STATUS: COMPLETE (26 Aug 2026).** The result file carries
`role: post_hoc_dose_extension`.

> **Margin correction, 2026-09-09.** The non-inferiority margin recorded in
> `data/results/x89_positive_dose_extension.json` is now **0.02** per guard
> endpoint, not 0.05; by author decision the 0.05 was a transcription error
> ([x87](x87-external-safety-overrefusal-eval.md) carries the full note).
> No number below changed; the file records no gate verdicts. Where a bound
> is compared to the margin in the reads, 0.02 is the figure. Against it,
> one guard bound in this table falls outside: `x86_p050`'s StrongREJECT
> one-sided upper (+0.0219), which the draft daggers in its write +0.5
> column. `x75_m1`, `x75_m2`, `x75_p1` and `x75_p2` also carry bounds
> outside 0.02 on Toxic and/or IFEval.

## Design

Seven conditions on the frozen x87 Qwen3.5-4B snapshot, through the
unchanged x87 pipeline (same manifest rows, prompt hashes, greedy
generation, thinking disabled, seeds 8700; judge = fresh unedited copy of
the same snapshot; StrongREJECT official finetuned evaluator local;
IFEval prompt-strict; GSM8K same protocol as `x87_gsm8k`):

- Boundary write, positive doses: `x86_p050` (+0.5 = +a*), `x86_p100`
  (+1.0), `x86_p150` (+1.5, mirror of the x88 transfer dose).
- Rank-one edit (x87 stage-0 `refusal_weighted` adapter, unchanged),
  wrapper scales: `x75_m1` (−1), `x75_m2` (−2), `x75_p1` (+1),
  `x75_p2` (+2).
- `base` reused from x87; no condition regenerated.

Conditions appended after all existing `EXTENSION_CONDITIONS`
(`code/x87_common.py`), so every x87 bootstrap seed index is unchanged;
`code/x89_analyze.py` re-runs the full x87 seed schedule and hard-fails
unless every pre-existing x87 comparison reproduces the stored
`x87_external_safety_utility.json` exactly. Analysis written to
`data/results/x89_positive_dose_extension.json` (x87 result files are
read, never written). Paired deltas vs the x87 base generations,
20,000-resample bootstraps, x87 alphas for comparability.

## Results

All deltas paired vs the x87 base; \* = 95% interval excludes 0.

x87 base rates: Hard-1K 0.8802, Toxic 0.9893, StrongREJECT 0.0633,
IFEval 0.8041, GSM8K strict 0.7877.

Boundary write, edited − base:

| condition | Hard-1K | Toxic | StrongREJECT | IFEval | GSM8K strict |
|---|---|---|---|---|---|
| x86_p050 | +0.0288 [+0.0167, +0.0417]\* | +0.0031 [+0.0000, +0.0107] | +0.0037 [−0.0159, +0.0237] | +0.0111 [−0.0129, +0.0351] | +0.0068 [+0.0023, +0.0121]\* |
| x86_p100 | +0.0546 [+0.0387, +0.0705]\* | +0.0031 [+0.0000, +0.0107] | −0.0102 [−0.0260, +0.0036] | +0.0314 [+0.0055, +0.0591]\* | +0.0030 [−0.0030, +0.0091] |
| x86_p150 | +0.0644 [+0.0478, +0.0826]\* | +0.0031 [+0.0000, +0.0107] | −0.0100 [−0.0266, +0.0044] | +0.0240 [−0.0037, +0.0536] | +0.0008 [−0.0068, +0.0083] |

Table: [Rank-one edit, edited − base, at wrapper scales −1/−2/+1/+2](json/x89-positive-dose-external-extension__rank-one-deltas.json) — Hard-1K moves −0.0258 / −0.0455 at −1/−2 and +0.0227 / +0.0455 at +1/+2; `x75_p2` raises StrongREJECT +0.0303 (interval excludes 0).

## Analysis

Both questions resolved: the boundary write mirrors externally and the
rank-one dose-response is smooth through the origin; neither family is
null at ±1.

1. **The boundary write mirrors externally, dose-ordered.** Hard-1K rises
   +0.0288 → +0.0546 → +0.0644 at +0.5/+1.0/+1.5, every interval
   excluding zero. Both intervention families now have an external
   symmetric sign control. As with the rank-one +3 control, the added
   refusal buys no measurable safety: StrongREJECT is null (point
   estimates slightly negative at +1.0/+1.5) and Toxic refusal is
   already at ceiling (+0.003 against base 0.9893). IFEval moves
   slightly *up* (+0.0314 significant at +1.0), so positive doses are
   not a capability tax at these magnitudes.
2. **The rank-one dose-response is smooth and near-linear through the
   origin, and strikingly symmetric.** Hard-1K: −1 → −0.0258, −2 →
   −0.0455, −3 (x87) → −0.0781; +1 → +0.0227, +2 → +0.0455, +3 (x87)
   → +0.0580. `x75_m2`, the deepest scale inside the training-time KL
   budget (benign KL 0.018 < 0.02), already cuts Hard-1K −0.0455 with
   Toxic cost −0.0137 and null StrongREJECT/IFEval/GSM8K — roughly 58%
   of the −3 candidate's effect at a fraction of its distribution
   shift.
3. **The +2 scale already leaks.** `x75_p2` raises StrongREJECT +0.0303
   (interval excludes 0) — the "more refusal, less safety" signature the
   +3 control showed (+0.022) is present one scale earlier, while the
   negative branch stays SR-null down to −3.
4. **GSM8K is flat.** All GSM8K deltas are within ±0.013.

## Deviations

- None in design or pipeline. `code/x89_analyze.py` reproduced every
  pre-existing x87 comparison exactly (`x87_reproduction_check: passed`).
- The only code change to shared x87 files: seven conditions/doses
  appended in `code/x87_common.py` and the suppression-only dose guard
  relaxed to nonzero (positive doses are the point of x89).

## Artifacts

- `data/runs/x89/x89_{generations,orbench,strongreject,ifeval,gsm8k}_<condition>.json`
- Analysis: `data/results/x89_positive_dose_extension.json`
- Code: `code/x89_analyze.py`; conditions/doses appended in
  `code/x87_common.py`.
