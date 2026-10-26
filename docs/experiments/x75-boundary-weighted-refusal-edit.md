# x75: the refusal-weighted probe and the rank-one compiler (path A)

Parent: [Experiment index](README.md). The plan document is not kept in
this project.

This file owns **the method** the paper's path A uses: supervise a linear
probe on a prompt's sampled refusal rate, undo the feature standardization
to get a residual-space covector, and compile that covector into one MLP
`down_proj` as a rank-one weight update with a signed deployment scale,
selected under a benign-KL budget. The paper reports this method's
**Qwen-native port**; the numbers below are from the Llama experiment that
produced and stress-tested the recipe. The compiler is the rank-one editing
lineage; what path A contributes is the supervision — a behaviorally
supervised refusal-rate probe — as an independent route to the same place
the label-free crossed differencing lands.

## The method, as it is used

- **Supervision.** Each training prompt is answered ten times and judged;
  the sampled refusal rate `r` is a soft target for a logistic fit, which
  keeps the fractional-label loss exactly equal to the average Bernoulli
  loss of the ten observed outcomes. The `refusal_weighted` arm additionally
  weights each example by `r`, so behaviorally mixed and refusing prompts
  dominate the fit. (The draft narrates the probe as predicting `1 − r`;
  that flips the sign of the weight vector and the intercept and leaves the
  axis unchanged.)
- **Probe to covector.** Fit on one frozen cache state with train-only
  standardization; undoing the standardization gives a residual-space
  covector `q`, normalized to `q̂`.
- **Compiler.** `W ← W + s·q̂ vᵀ` on the chosen block's MLP `down_proj`,
  with only the right factor `v` trained. Every prompt-dependent update is
  therefore confined to the probe axis and `v` supplies a scalar gate. The
  objective pushes the probe score up on harmful prompts under a benign-KL
  penalty; the checkpoint is selected as the best objective value inside a
  0.02 benign-KL budget. Training at `s = +1` makes the gate raise the
  refusal score, so a **negative deployment scale reverses the same
  one-dimensional write toward the comply side**.
- **Dosing.** The deployment scale is a wrapper multiplier on the trained
  adapter. It is model-specific by construction: nothing makes a given
  scalar mean the same thing on two models, which is exactly what the
  cross-model transfer result exploits
  ([x88](x88-eight-model-deep-dose-transfer.md)).

Implementation: `code/x71_prepare_dataset.py` (builds the probe dataset from
a feature cache), `code/x75_prepare_weighted_probe.py`,
`code/x75_train_weighted_probe.py`, `code/x75_train_edit.py` (a shim over
`code/x71_train_edit.py`), `code/x75_dose_kl.py`, with the edit mechanism in
`code/lib/edit_rank1.py` and model/config helpers in
`code/lib/edit_common.py`. Unit tests: `code/tests/test_x75_weighting.py`,
`code/tests/test_edit_rank1.py`.

## The Qwen-native port the paper reports

Rerun end to end on Qwen3.5-4B's own cache, refusal labels and seed-42
split, inside the external-evaluation experiment
([x87](x87-external-safety-overrefusal-eval.md), stage 0). Weighted probe:
test unweighted Spearman 0.766, AUC against majority refusal 0.991. Adapter:
trainable `model.layers.30.mlp.down_proj.right` only, selected at step 200
at benign-prompt KL 0.0059 inside the 0.02 budget. Deployment scale −3; its
benign-KL dose curve is in
[x87](x87-external-safety-overrefusal-eval.md#dose-extension-added-2026-08-24).

Artifacts (all under `data/runs/x87/`):
`x75_qwen_refusal_weighted_true.pt` (the adapter),
`x75_qwen_refusal_weighted_true.json`/`.jsonl`, `x75_qwen_probes/`,
`x75_qwen_weight_audit.json`, `x87_x75_qwen_config.json`, and the dose–KL
curve `x87_x75_qwen_dose_kl.json`. The guard against loading the Llama-era
adapters is described in [x87](x87-external-safety-overrefusal-eval.md#design-as-run).

## Why the method exists in this form: the originating negative result

The originating experiment asked whether re-weighting the refusal-rate probe
toward behaviorally *mixed* prompts (entropy weightings `H(r)` and
`(1−r)·H(r)`) produces a larger refusal edit at matched benign KL. **It does not.** Both
entropy arms failed before any
behavior was read: zeroing the weight at `r = 0` and `r = 1` left about
2,000 usable training prompts, on which the inherited regularization grid
under-fit a 3,000-dimensional logistic probe — every candidate state scored
worse than an intercept-only baseline, selection fell to nearly
refusal-free early layers, and the resulting directions were geometrically
indistinguishable from noise. Their adapters trained but could not be
KL-calibrated to the reference doses, so their primary comparison was
impossible.

The `refusal_weighted` arm — weighting by `r`, the control arm — is the one
that behaved like a real refusal probe (validation AUC 0.936, test Spearman
0.754) and reproduced the reference dose-response almost exactly (slope
0.01594 versus 0.01585 on the same coordinate) while sharing only cosine
0.30 with the direction it was compared against. It was **not better** at
matched KL (Holm-adjusted p = 1.0), and on the frozen capability battery it
missed 7 of 16 margins that the earlier direction had passed — same benign
KL does not imply the same capability cost.

Two things follow:

1. `refusal_weighted` is the arm the paper uses because it is the one that
   calibrates and behaves.
2. Because the same benign KL did not give the same capability cost, the
   external evaluation measures instruction following and math directly
   rather than inferring them from KL.

The originating run's artifacts are not kept in this project.
