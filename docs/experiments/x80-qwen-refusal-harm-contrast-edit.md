# x80: Qwen3.5-4B refusal-versus-compliance paired probes and contrast

Parent: [Experiment index](README.md). The plan document is not kept in
this project.

## Status

Run 2026-08-21/22. Stage 0 local CPU; Stages 1–2 on the cluster (six
single-GPU jobs: three per L2 grid, one per outer seed). **Stages 3–5 (localization, edit,
generation) did not run: gate G1 fails** (section 1), and the plan stops
editing when the two targets are practically complementary.

## Label contract as actually run

The plan's H target, a full-coverage realized-harm label ("v8"), does not
exist: x64 judged only `refusal2` and `comply2`; the only harm judgements in
the lineage are on other responses (v2 `llm_clustering`, 0/2000 response
lists identical to x63) or on 300 prompts (x59, fields not in the repo). By
explicit author decision on 2026-08-21 the H target is the x64 **`comply2`**
label — substantive compliance, whose rubric states harmfulness is
irrelevant. **Every H-side number below is compliance, not judged harm.**
Lineage, hashes and the rubric-source hashes are in
`data/runs/x80/x80_label_manifest.json` (G0 passes: 13,527/13,527 IDs on
both methods, ten booleans each, same x63 responses in index order, 11
failed refusal2 attempts all re-run to success).

## 1. Stage 0 — refusal and compliance are nearly complementary (G1 fails)

Response-level 2×2 over all 135,270 responses:

| | H=0 (no attempt) | H=1 (complied) |
|---|---:|---:|
| R=1 (explicit refusal) | 109,310 | 4,176 |
| R=0 | 1,554 | 20,230 |

Non-complementary cells `(1,1)+(0,0)` = **4.24%** (outer-train 4.17 / 4.27 /
4.33% on seeds 42/123/7; gate needs ≥ 5%). Prompt-level `r` and `h`
correlate −0.907 Pearson / −0.857 Spearman. 9,608 of 13,527 prompts are
deterministic on both (9,494 at 10/10 refusals); 2,724 vary on both, 367 on
R only, 828 on H only. Label-sampling floors (exact binomial) are 0.022 (R)
and 0.025 (H) in MAE of the rate.

The 4,176 refuse-then-comply responses are the construct's only genuinely
two-dimensional content, and they are 3.1% of responses. Under the plan's
rule the two heads carry one binary axis plus a small residual, so
localization and editing stop.

## 2. Stages 1–2 — paired probes and the contrast direction

Three outer seeds, 32 non-embedding states, 9,739 inner-train / 2,435
inner-val / 1,353 outer-test harmful prompts per seed. Heads: L2 logistic on
the train-standardized final-prompt-token state, fractional-label BCE over
the ten draws, L2 chosen per head on inner-validation NLL, five
label-permutation fits per head and state. Common state = argmax over states
of min(skill_R, skill_H) on validation; outer test read only at that state.

**Both heads are predictive everywhere.** All 32 states are eligible on all
three seeds (permutation-null 95th percentiles are −0.24 to −0.29 skill).
Deviance skill rises from ~0.12 (R) / ~0.08 (H) at state 1 to a plateau from
state ~19 on. At the selected state (23 / 19 / 21 for seeds 42 / 123 / 7):

| head | val skill | test skill | test Brier | test Spearman vs 10-draw rate | test AUROC(count ≥ 5) |
|---|---:|---:|---:|---:|---:|
| R (refusal) | 0.60–0.63 | 0.574–0.598 | 0.015–0.016 | 0.735–0.771 | 0.985–0.986 |
| H (compliance) | 0.50–0.51 | 0.462–0.481 | 0.026–0.030 | 0.712–0.715 | 0.955–0.964 |

The H head is uniformly the weaker of the two (skill lower by 0.10–0.12),
and it is the H skill that fixes the common state.

**The two directions are distinguishable, and that is robust.** At the
selected state `cos(q_R, −q_H)` = 0.577 / 0.599 / 0.588 — the refusal and
compliance-suppression gradients sit 54–55° apart, not antiparallel. Across
states, `cos(q_R, q_H)` runs from −0.90 at state 1 to −0.46 – −0.49 at the
deepest states on every seed: the two heads *separate* with depth even as
both get more predictive. The contrast `unit(q_R − q_H)` lies 27° from each
single-head direction (cos 0.888–0.894 to both, by construction symmetric);
`refusal_residual` is orthogonal to `−q_H` and keeps cos 0.80–0.82 with
`q_R`. Permutation and random directions are at |cos| ≤ 0.08 to everything.

**G2 passes on the x71 grid, narrowly, and the narrowness is a grid
artefact.** Prompt-bootstrap (50 resamples) stability of the contrast:
median cos 0.765–0.771, 90th-percentile angle 40.7–41.5° against the 45°
gate; the single-head directions are no more stable (41.7–42.3°). Every head
at every state selected L2 = 1e-3, the **top of the inherited x71 grid**, and
the validation calibration slopes of 0.79–0.84 say the same thing:
under-regularized. A sensitivity rerun with the grid extended to 1.0
(`data/runs/x80/probes_l2ext/`) picks interior optima (R 0.01–0.03,
H 0.03–0.1), raises test skill to 0.625–0.642 (R) / 0.534–0.550 (H), brings
calibration slopes to 1.00–1.09, moves the selected state to 30–31, and
tightens the contrast's p90 angle to 30.2–30.7° — while `cos(q_R, −q_H)`
stays at 0.563–0.596. The x71 grid is reported as primary; the extended
grid is the one a future edit should inherit.

## Artifacts

- Audit: `code/x80_label_audit.py`, `data/results/x80_label_audit.json`,
  `data/runs/x80/x80_label_manifest.json`, `data/runs/x80/x80_labels.npz`.
- Probes: `code/x80_fit_paired_probes.py`, `code/x80.sbatch`;
  `data/runs/x80/probes/x80_probes_s{42,123,7}.json`,
  `x80_paired_probe_s{seed}.pt`; extended-grid sensitivity under
  `data/runs/x80/probes_l2ext/`; aggregate `data/results/x80_probe_geometry.json`.
- the cluster: `<workdir>/x80/` (job ids in `x80_jobids.txt`).
