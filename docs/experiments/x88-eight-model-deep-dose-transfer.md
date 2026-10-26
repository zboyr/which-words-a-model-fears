# x88: eight-model transfer of the deep-dose refusal edits (x86 −1.5, x75 −6)

Parent: [Experiment index](README.md). **STATUS: COMPLETE (25 Aug 2026,
incl. the x75_m3 safety-endpoint extension).**

## Question

x87 (Qwen3.5-4B only) found that at deep suppression doses the x86-style
consensus boundary write keeps cutting OR-Bench Hard-1K refusal with a
flat capability profile, while the x75-style rank-one edit at scale −6 cuts
Hard-1K more but degrades IFEval at a benign KL far over budget (the −1.5
and −6 rows of
[x87's second-wave table](json/x87-external-safety-overrefusal-eval__second-wave.json)
and its [GSM8K table](json/x87-external-safety-overrefusal-eval__gsm8k.json)).
Does this pattern — boundary write capability-flat, rank-one edit paying an
instruction-following cost — transfer across the campaign models?

## Design

- **Models.** The x66 nine-model campaign minus smollm3 (near-zero refusal
  makes suppression doses meaningless there): qwen (Qwen3.5-4B, imported
  from x87), qwen08b, qwen2b, qwen9b, qwen27b, llama (Llama-3.2-3B),
  gemma (gemma-4-e2b-it), phi (Phi-4-mini). Frozen snapshots from the
  the cluster HF cache, recorded per model in the x88 manifest.
- **Conditions per model: 3.** `base` (unedited), `x86_m150` (boundary
  write, dose −1.5), `x75_m6` (rank-one refusal_weighted edit, wrapper
  scale −6). Doses are the x87 wave-2 points of interest. Dose units are
  model-relative by construction (x86: multiples of the model's own mean
  boundary shift; x75: wrapper scale on the model's own trained adapter).
- **Per-model x86-style direction.** The frozen x85 crossed
  keyword-prompt cohort texts (model-independent) are re-rendered with
  each model's chat template; hidden states extracted at the two boundary
  positions (last two prefill tokens); fold directions
  `mu[f,t] = mean(h_H − h_N)` rebuilt exactly as in x86 stage 0; the
  x87-style consensus (mean of 4 fold directions) is written by a
  prefill-only hook at dose −1.5. Hook block per model by the frozen
  relative-depth rule `round(29/33 · n_blocks)` (recovers block 29 on
  Qwen3.5-4B); no per-model layer optimization.
- **Per-model x75-style adapter.** The x87 stage-0 chain rerun per model
  on its own x66 `_A15k_v4_full` cache, v4 refusal labels, and seed-42
  split: `x71_prepare_dataset` → `x75_prepare_weighted_probe` →
  `x75_train_weighted_probe` (state selected per model by the frozen
  minimum-weighted-validation-NLL criterion) → `x75_train_edit`
  (rank-one `down_proj.right`, KL budget 0.02 at selection). Wrapper
  scale −6 at eval.
- **Benchmarks.** Identical to x87: OR-Bench Hard-1K + Toxic refusal
  (three-way rubric), StrongREJECT score (official finetuned evaluator,
  local), IFEval prompt-level strict, GSM8K exact-match (strict-match
  primary). Same frozen prompt sets, greedy generation, seeds 8700.
- **Judge.** OR-Bench rubric judged by a single fixed judge — the
  unedited Qwen3.5-4B snapshot — for every model and condition, so
  cross-model deltas share one measurement instrument. (x87 judged qwen
  with qwen; the qwen row imported below is re-used unchanged, judged by
  the same model.)
- **Analysis.** Per model: prompt-paired edited-minus-base deltas,
  20,000-resample bootstrap, 95% intervals.
  The qwen (4B) row is imported from
  `data/results/x87_external_safety_utility.json` /
  `x87_gsm8k_posthoc.json`, not rerun.

## Results

Table: [Base rates per model](json/x88-eight-model-deep-dose-transfer__base-rates.json) — Hard-1K, Toxic, StrongREJECT, IFEval and GSM8K base values for the eight models; llama's Hard-1K refusal is only 0.103, against 0.47–0.90 elsewhere.

Table: [`x86_m150` − base per model: paired deltas with 95% bootstrap CIs](json/x88-eight-model-deep-dose-transfer__x86-m150-deltas.json) — Hard-1K refusal falls on all eight models, from −0.014 (llama) to −0.228 (qwen2b), with StrongREJECT at most +0.030 and |IFEval|, |GSM8K| at most 0.026 and 0.014.

Table: [`x75_m6` − base per model: paired deltas with 95% bootstrap CIs](json/x88-eight-model-deep-dose-transfer__x75-m6-deltas.json) — qwen08b loses IFEval −0.275 and GSM8K −0.309 while llama's Toxic refusal drops −0.362 and StrongREJECT rises +0.308.

`x75_m3` − base is an extension (25 Aug design decision: safety endpoints only —
same per-model adapters at scale −3, the x87 candidate scale;
no IFEval/GSM8K runs).

Table: [`x75_m3` − base per model (safety endpoints only): paired deltas with 95% bootstrap CIs](json/x88-eight-model-deep-dose-transfer__x75-m3-deltas.json) — Hard-1K refusal falls on all eight models; qwen08b, qwen2b and llama leak (StrongREJECT +0.029 / +0.076 / +0.051).

Table: [Per-model derivation records](json/x88-eight-model-deep-dose-transfer__derivations.json) — per model the probe-selected x75 layer, adapter KL at selection, selected step, x86 hook block and min cross-fold direction cosine; all adapters stayed inside the 0.02 benign-KL budget and all direction sets have min cross-fold cosine >= 0.971.

Judge health: all 21 OR-Bench judgment files (7 models x 3 conditions) have
n=1974 with zero unparsed labels; all StrongREJECT files have exactly 313
rows. The `x75_m3` extension is equally clean (7 x 1974 OR-Bench rows, zero
bad labels; StrongREJECT 313 rows everywhere).

## Analysis

**The activation-space boundary write transfers cleanly at a fixed dose;
the weight-space rank-one edit does not.**

1. **`x86_m150` moves over-refusal in the same direction on all eight
   models.** Hard-1K refusal drops with the 95% interval excluding zero on
   8/8 models, from −0.014 (llama, a floor effect: its base Hard rate is
   only 0.103) to −0.228 (qwen2b). The cost side is an order smaller
   everywhere: Toxic refusal moves 0 to −0.084, StrongREJECT is near-null
   (largest increase +0.030, on the 0.8B model), and utility is intact
   (largest |IFEval delta| 0.026, largest |GSM8K delta| 0.014, mixed
   signs). The same model-independent cohort texts, the same frozen
   relative-depth rule, and the same dose −1.5 produce a proportionate
   effect from 0.8B to 27B across four model families.

2. **`x75_m6` at a fixed scale −6 lands at wildly different effective
   doses.** The mid/large qwen models reproduce the x87 qwen picture
   (Hard −0.064 to −0.154, small toxic cost, utility roughly flat). At the
   edges it breaks in two different ways:
   - **qwen08b — capability damage:** IFEval −0.275, GSM8K −0.309, while
     Hard barely moves (−0.020, null). Responses stay fluent but become
     long and meandering (mean response length 2708 → 4271 chars); the
     StrongREJECT *decrease* (−0.122) reads as degraded answer quality,
     not improved safety.
   - **llama — safety collapse:** Toxic refusal −0.362 and StrongREJECT
     +0.308. Spot-checked toxic responses genuinely comply ("here are
     some ways technology can be used..."), so this is real behavior, not
     a judge artifact.
   - qwen2b and phi sit between: large Hard drops (−0.298 / −0.208) but
     significant StrongREJECT increases (+0.090 / +0.044) and, for
     qwen2b, IFEval −0.113.

3. **`x75_m3` (the x87 candidate scale) shows the same
   non-portability already at half the depth.** The mid/large qwen models
   stay clean (Hard −0.041/−0.049, StrongREJECT flat or down). Everywhere
   else the fixed scale still overshoots, just more mildly than −6:
   qwen08b and qwen2b cut Hard −0.162/−0.190 but leak (StrongREJECT
   +0.029/+0.076, Toxic −0.128/−0.082); llama repeats its signature in
   miniature (Toxic −0.053, StrongREJECT +0.051, vs −0.362/+0.308 at −6);
   phi and gemma show small significant StrongREJECT increases (+0.012 /
   +0.014). The failure is dose-monotonic, not a −6 cliff.

4. **Reading.** The boundary write is expressed in each model's own
   residual units (per-model direction norms ~0.9, extracted from the
   model's own activations), so a fixed dose stays proportionate to the
   model. The rank-one weight edit's effective strength depends on the
   model's scale and weight geometry, so a fixed scalar multiplier
   overshoots small or off-family models — into either incoherence-adjacent
   verbosity (qwen08b) or genuine harmful compliance (llama). Weight-space
   edits need per-model dose calibration; the activation-space
   intervention is dose-portable as-is.

## Deviations

- `code/x88_analyze.py` had a read-side bug (treated lm_eval's `metrics`
  list-of-names field as a dict container); patched. Analysis
  code only — no generation, scoring, or lm_eval artifact was affected.
- Wave 1 of the cluster submission was lost to an sbatch authoring bug (see
  [../infrastructure.md](../infrastructure.md)); its completed small-model stage-0 artifacts (written before the
  cancellation) were reused by wave 2 after per-step artifact checks.

## Artifacts (expected)

- Manifest and per-model/per-condition artifacts:
  `data/runs/x88/x88_manifest.json`,
  `data/runs/x88/<tag>/x88_{generations,orbench,strongreject,ifeval,gsm8k}_<condition>.json`.
- Per-model derivations: `data/runs/x88/<tag>/x88_directions.npz`,
  `x75_<tag>_refusal_weighted_true.pt` (+ probe/audit/config records).
  Prepared tensors stay on the cluster scratch (regenerable).
- Analysis: `data/results/x88_cross_model_deep_dose.json`. `x75_m3` is in
  `x88_common.CONDITIONS`/`DOSES`; `x88_analyze.py`'s `SAFETY_ONLY` set makes
  the utility endpoints optional for it.
- Code: `code/x88_*.py`.
