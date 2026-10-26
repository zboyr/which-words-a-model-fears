# Refusal and compliance surface labels, and the targets built from them

Parent: [Experiment index](README.md).

Owns where the `refusal2` and `comply2` judgments come from and why refusal
presence is the label this paper supervises and scores on. Coverage, the
judge settings and the frozen label files are in
[../dataset-freeze.md](../dataset-freeze.md).

## Which label scheme is repeatable and portable?

On the 25 most-adversarial `disagree100` rows, regenerated at full length
(Qwen3.5-4B, 2048 tokens) and judged by GPT-5.6/DeepSeek-v4-flash/GLM-5.2, two
passes each: `bh2` (benign/harmful) had the best mean native test-retest kappa
(0.882) but degenerated for GLM (labeled all 250 responses benign, kappa=1.000
trivially); `sev02` (severity 0/1/2) was the only scheme non-degenerate for
all three judges and the cross-judge binary-collapsed kappa winner (mean
0.597; pairs 0.657/0.488/0.644); `class3` (hc/dh/ar) was worst on both axes.
This motivated
looking past holistic/class3-style schemes toward the simpler surface labels
of the next section.

## Refusal presence is the stable cross-family surface label

Same 250 full-length responses and judge grid, comparing `comply2`
(substantively attempted the request) vs `refusal2` (contains an explicit
refusal statement):

- `refusal2`: 49% base rate, within-judge kappa 0.992/0.968/1.000
  (GPT/DS/GLM), cross-LLM kappa 0.976-0.988, 98.0% six-pass unanimity.
- `comply2`: within-judge kappa 0.842/0.750/0.816, cross-LLM kappa only
  0.464-0.687, 61.2% unanimity — much fuzzier on partial-delivery and redirect
  responses.

These are exactly the `comply2`/`refusal2` prompts every experiment in this
paper reuses unchanged.

## Harm is strongly refusal-conditional at scale

300 prompts (a risk-decile sample), 10 Qwen3.5-4B responses each, 3 judge
families × 2 passes. P(harm | refusal) = 0.8% (13/1707) vs P(harm | no
refusal) = 14.6% (187/1285); within the 175 prompts producing both refusal and
non-refusal responses (controlling for prompt identity): 0.3% (3/997) vs 20.9%
(156/746) — the contrast widens, not disappears, within-prompt. Refusal
reliability replicates at scale: within-judge kappa 0.985-0.990, cross-judge
kappa 0.975-0.978. The refusal/harm fork is what motivated rebuilding
supervision on refusal/compliance surface labels instead of criteria-based
severity.

## The supervision targets built from these labels

The production pass runs the `refusal2` and `comply2` prompts verbatim on
the frozen harmful set — judge, settings, coverage and artifact paths:
[../dataset-freeze.md](../dataset-freeze.md).

Two blends are built from the same per-response refusal/comply booleans
(10 responses per prompt). **This paper uses the refusal rate `r` itself and
the v4 line built from it; v5 is described because the checked-in label
files carry both.**

- **v4** (refusal-risk blend): r = refusal_rate over the 10 responses; T_v4 =
  0.7\*(1-r)+0.3\*H2(r), H2 = normalized binary entropy of r. Uses `refusal2`
  only — `comply2` is stored per row but not part of T_v4. The
  refusal-weighted probe of the paper's path A supervises on `r` directly
  (the paper writes it as 1 − r; the two differ only in sign).
- **v5** (3-class rebuild, "mapping B"): per response, comply=1 & refusal=0 ->
  HC; comply=1 & refusal=1 -> DH; comply=0 -> AR. safety_score =
  p(HC)+0.5\*p(DH), safety_entropy = normalized 3-class entropy; T_v5 =
  0.7\*safety_score+0.3\*safety_entropy — a weighted blend of a safety
  score and its entropy, rebuilt from these surface labels instead of a
  criteria classification.

Both reduce to the shared T = 0.7\*a+0.3\*b form; v4 is refusal-rate-only, v5
folds in compliance to recover a 3-class-shaped construct. "Zero-T" means
"always refused" under both, but by each blend's own criterion (refusal_rate=1
for v4; comply=0 on all 10 responses for v5) — the two zero-sets are close but
not identical per model.

`comply2` also appears in this paper as a *secondary outcome*: the
intervention experiment judges it alongside refusal to check that the
intervention moves substantive delivery, not refusal wording
([x86](x86-crossed-keyword-boundary-intervention.md)). It is a separate
construct from `refusal2` and the two are never pooled.
