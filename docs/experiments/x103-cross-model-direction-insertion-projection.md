# x103: in all eight models, the shift an inserted harm word makes along that model's own boundary direction predicts the refusal it induces, on 136 prompt–word units the direction was never fit on (ρ = 0.32–0.77)

Parent: [Experiment index](README.md). Joins the behavioral result of
[x101](x101-cross-model-keyword-insertion.md) to the transferred direction of
[x88](x88-eight-model-deep-dose-transfer.md): x101 showed the lexical reaction
recurs in all eight models, x88 showed that writing each model's own crossed
boundary direction lowers OR-Bench Hard-1K refusal there, and this file
supplies the link between them off the home model — the association that
[x85](x85-crossed-keyword-prompt-latent-decomposition.md)/[x86](x86-crossed-keyword-boundary-intervention.md)
established on Qwen3.5-4B alone. The cohort is x101's plus
[x114](x114-x85-free-insertion-topup.md)'s top-up.

No new generation and no new judging beyond each cohort's own: one forward
pass per text, reading the residual stream at the site the x88 boundary write
edits. For each of the 136 units whose prompt and inserted word are both
outside the x85 cohort the directions were fit on — the 64 such x101 units and
the 72 x114 units, which are outside it by construction — and each insertion
position, the insertion-induced shift at the two boundary tokens is projected
onto that model's own consensus boundary direction and compared with that
prompt's `refusal2` rise. **In every one of the seven transfer models the
projection predicts the rise after position fixed effects** (Spearman ρ = +0.32
on Qwen3.5-0.8B to +0.77 on Qwen3.5-9B, permutation p = 10⁻⁴, Holm p = 7 × 10⁻⁴
over the seven), and in every model it survives both specificity controls: the
partial ρ controlling for the size of the shift stays positive (+0.22 to +0.53,
every bootstrap interval excluding zero, lower bounds +0.11 to +0.40), and the
real ρ sits above all 1,000 random directions on seven models and above 997 on
Phi. Qwen3.5-4B as the reference row gives ρ = +0.64, reproducing x85's
held-out ρ = 0.65 by a different route. Nothing here is routed to
[../headline-numbers.md](../headline-numbers.md).

## Design, as run

- **Texts and behavior.** Two frozen insertion cohorts with the same O/S/M/E
  construction: the 130-prompt x101 cohort (`data/runs/x101/x101_cohort.json`)
  with the x101 per-prompt `refusal2` rates per condition (for Qwen3.5-4B, the
  x83 rows x101 used), and the 72-prompt x114 cohort
  (`data/runs/x114/x114_cohort.json`) with its own `refusal2` rates, judged
  under the same contract. `Δr_{i,pos} = r_{i,pos} − r_{i,O}`. Prompts without a
  successful judgment in some condition are dropped from that model in all
  four, as x101 did (one Qwen3.5-27B prompt of x101, outside the retained 64;
  none in x114).
- **Cohort modes.** The directions were fit on the x85 crossed cohort (96
  prompts × 24 harm words, `data/runs/x85/x85_cohort.json`, cohort sha256
  `93784a93…`). Of the 130 x101 units, 47 use an x85 prompt and 32 carry one of
  17 x85 harm words (`anthrax`, `bomb`, `botnet`, `burglary`, `cyanide`,
  `forgery`, `fraud`, `grenade`, `harmful`, `illegal`, `malware`, `porn`,
  `racist`, `rape`, `rifle`, `torture`, `toxin`); the 66 units with either are
  excluded and 64 retained (`data/runs/x105/x105_x85_exclusions.json`,
  `2305ef73…`, built by `code/x105_exclusions.py` from
  `x103_common.leakage_sets`). x114's prompts are drawn outside x83's (and so
  x85's) and its words outside x85's 48, and the same rule applied to it as a
  check excludes none (`data/runs/x114/x114_x85_exclusions.json`). The
  analysis runs in three modes (`code/x114_cohorts.py`, `--cohort`): `extended`
  — the 64 x101 units pooled with the 72 x114 units, **the primary cohort and
  the canonical result files** (136 prompts, 408 cells per model);
  `x101_leak_free` — the 64 x101 units alone, the `_x101only` siblings; `full`
  — all 130 x101 units, the `_full` siblings. Every exclusion artifact is
  re-derived and checked at load, and its sha256 is recorded in the result's
  `cohort` block.
- **Rendering.** x101's renderer (`x87_common.format_user_chat`: user turn
  only, `add_generation_prompt`, `enable_thinking=False`). Every rendering is
  checked against the `rendered_chat_sha256` the generation row recorded, so
  the text entering the forward pass is byte-identical to the text that was
  generated from. The Llama template stamps the run date into its system
  header; the run date is recovered from that hash (x101: `02 Oct 2026`; x114:
  `09 Oct 2026`).
- **Read site.** A forward hook on the decoder block the x88 boundary write
  hooks (`x88_common.setup_condition` at `x86_m150`, block by the frozen
  relative-depth rule `round(29/33 · n_blocks)`), reading the block output at
  the two trailing prefill positions — the same tensor, the same block and the
  same positions the write edits. Blocks: 21/24 (0.8B, 2B), 28/32 (9B, Phi),
  56/64 (27B), 25/28 (Llama), 31/35 (gemma), 29/32 (Qwen3.5-4B, x86's site).
- **Site audit, in every job.** On four texts per model: the hooked read is
  bitwise equal to `output_hidden_states[block + 1]` at those positions; with
  the x88 write hook registered ahead of the reader, the read tensor moves by
  exactly `dose · μ` to bf16 rounding (minimum cosine to μ over all models and
  both cohorts 0.99985, norm ratio within 0.3 %; the bound covers two bf16
  roundings, because the x88 hook casts μ to bf16 before adding, and the job
  also checks cosine ≥ 0.999 and norm ratio within 1 %); the write fires once on
  prefill and never on decode; the logits change; removing the hook restores
  the read exactly. A failing audit aborts the job before any cell is written.
  All sixteen jobs (eight per cohort) passed.
- **Directions.** Per transfer model the x88 consensus (mean of the four fold
  directions) in `data/runs/x88/<tag>/x88_directions.npz`; for Qwen3.5-4B the
  x87 consensus of the x86 fold directions
  (`data/runs/x86/x86_directions.npz`). Both are read with the same
  `x87_common.consensus_x86_vectors` the write uses, and each artifact's sha256
  is pinned in the frozen inputs of both cohorts and re-checked in the job.
  Direction norms 0.89 (0.8B) to 42.6 (27B).
- **Measures.** Per prompt `i` and position `pos`, with `t` over the two
  boundary positions: `proj = Σ_t ⟨h_{i,pos,t} − h_{i,O,t}, μ_t⟩ / ‖μ_t‖`;
  `shift = ‖h_{i,pos} − h_{i,O}‖` over the concatenated boundary positions.
- **Analysis** (`code/x103_analyze.py`). Primary: Spearman ρ between `proj` and
  `Δr`, both residualised on position fixed effects; p from 10,000 permutations
  of `Δr` across prompts within position; 95 % CI from a 10,000-replicate
  prompt-level bootstrap (a drawn prompt brings all three of its positions);
  Holm over the seven transfer models, with Qwen3.5-4B outside the family.
  Controls: partial Spearman given `shift`; 1,000 random Gaussian unit
  directions per boundary position at matched dimension, scored with the
  identical primary statistic on the same cells; prompt + position fixed
  effects; per-position raw Spearman. Seeds, permutation count, bootstrap and
  Holm are the same in all three cohort modes. No decision rule.

## Results

Table: [Projection–refusal association per model](json/x103-cross-model-direction-insertion-projection__projection-refusal-association.json) — per model the primary position-FE Spearman ρ with CI, permutation and Holm p, the shift-controlled partial ρ, the random-direction percentile, the prompt + position FE ρ, and the ρ of the 64-unit and 130-unit siblings; primary ρ runs from +0.324 (qwen08b) to +0.766 (qwen9b), and every real ρ sits at or above the 99.7th random-direction percentile.

Every ρ interval excludes zero (permutation p = 10⁻⁴ everywhere), and so does
every partial-ρ interval (lower bounds +0.11 on Qwen3.5-0.8B to +0.40 on
Qwen3.5-9B).

Figure: `data/results/x103_direction_insertion_projection.pdf` (per-model
primary ρ and shift-controlled partial ρ with intervals, over the
random-direction band).

**Random-direction null.** The null is wide — a random direction's ρ spans
[−0.55, +0.54] at 2.5–97.5 % on Qwen3.5-9B — because `proj` and `shift` are
strongly coupled (Spearman 0.66 on Phi to 0.93 on Qwen3.5-4B and 27B), so any
direction inherits some of the shift-size signal. The real ρ is nevertheless
above **all 1,000** draws on Qwen3.5-4B, 2B, 9B, 27B, Llama and Gemma (above
the largest draw by 0.01 on 2B to 0.12 on 9B, 0.766 vs a maximum of 0.650) and
on Qwen3.5-0.8B (0.3236 against a largest draw of 0.3236, by less than 0.001),
and above 997 on Phi (0.343 vs 0.379); the one-sided random-direction p is
0.001 everywhere except Phi (0.004).

**Secondary.**

- *Per position* (raw ρ, no fixed effects). End is the strongest position in
  four of eight models and the three positions are within 0.03 of each other in
  the other four: qwen 0.67 / 0.59 / 0.67 (S / M / E), qwen08b 0.26 / 0.19 /
  0.44, qwen2b 0.56 / 0.53 / 0.72, qwen9b 0.77 / 0.76 / 0.75, qwen27b 0.70 /
  0.72 / 0.69, llama 0.42 / 0.39 / 0.65, gemma 0.46 / 0.42 / 0.61, phi 0.37 /
  0.29 / 0.36. The 0.8B model, whose S and M behavior x101 and x114 found null,
  is also where the association is weakest at those positions.
- *Shift size alone* predicts `Δr` less well than the projection in every
  model (position-FE ρ: 0.24 on 0.8B and Phi, 0.41 gemma, 0.46 llama, 0.55–0.66
  on the Qwen models ≥ 2B, against primary ρ 0.32–0.77).
- *By boundary token.* Residualised ρ computed from one token's projection
  alone: the final token (`tbg`) is at least as predictive as `tbg_minus1` in
  seven of eight models (27B: 0.63 vs 0.68), most clearly on Phi (0.43 vs 0.18)
  and Llama (0.58 vs 0.49) — the same ordering x85/x86 found on Qwen3.5-4B,
  where the final token carried the causal effect.
- *Raw ρ without fixed effects* is 0.34 (Phi) to 0.79 (9B); position fixed
  effects move the estimate by at most 0.05, so the association is not an
  artifact of end insertions both shifting more and raising refusal more.
- *Mean projections scale with the model's residual units* (mean `proj` at E:
  1.4 on 0.8B, 7.9 on 2B, 13.7 on 4B, 62.9 on 9B, 104.6 on 27B, 10.0 llama,
  15.2 gemma, 19.0 phi) and are positive at every position in every model:
  the inserted harm word moves each model's boundary state along its own
  suppression direction, in the sign x88 writes against. On the x114 units
  alone the end-position means are 1.4 / 8.4 / 14.5 / 66.0 / 108.5 / 10.5 /
  15.6 / 19.5, within 10 % of the x101 units'.

## Reading

- **It is direction-specific by both controls.** The prediction is carried
  by *where* the word pushes the state, not only by *how far*.
- **Strength tracks the model, in the same order as the behavior.** The
  weakest associations are Qwen3.5-0.8B (+0.32) and Phi (+0.34), the models
  whose start- and middle-insertion effects x101 and x114 found null or
  smallest; the strongest are the mid and large Qwen models (+0.63 to +0.77).
  Llama and Gemma sit between (+0.50 to +0.55) despite near-floor baseline
  refusal (r_O 0.02–0.06).
- **The direction's fitting cohort is not doing the work, and neither is
  either insertion cohort on its own.** Every number above is on prompts and
  words the directions never saw, from two cohorts drawn by different rules
  (x101's from the x66 test split with x83's word assignment, x114's from the
  rest of the campaign pool with a fresh draw). **64-unit cohort** (the x101
  leak-free units alone, 192 cells per model;
  `data/results/x103_direction_insertion_projection_x101only.json`, from
  `python3 code/x103_analyze.py --cohort x101_leak_free`): ρ = +0.26 (0.8B),
  +0.66 (2B), +0.68 (4B), +0.79 (9B), +0.74 (27B), +0.56 (Llama), +0.57
  (Gemma), +0.41 (Phi), partial ρ +0.21 to +0.53, random-direction percentile
  98.5 (0.8B) to 100, Holm p = 7 × 10⁻⁴. **Full cohort** (all 130 x101
  prompts, 390 cells per model, 387 on Qwen3.5-27B;
  `data/results/x103_direction_insertion_projection_full.json`, from
  `--cohort full`): ρ = +0.31 (0.8B) to +0.75 (9B), partial ρ +0.22 to +0.45,
  every real ρ above all 1,000 random directions, Holm p = 7 × 10⁻⁴. The
  top-up moves ρ by at most 0.07 against the 64-unit values (0.8B +0.26 →
  +0.32, Gemma +0.57 → +0.50, Phi +0.41 → +0.34, 27B +0.74 → +0.69), in both
  directions, and tightens every interval; it changes no verdict.

## Artifacts

- Result: `data/results/x103_direction_insertion_projection.json` (the 136
  leak-free units; the `cohort` block records both cohorts' counts and the
  sha256 of both exclusion artifacts), figure
  `x103_direction_insertion_projection.{pdf,png}`; siblings
  `x103_direction_insertion_projection_x101only.{json,pdf,png}` (64 units)
  and `x103_direction_insertion_projection_full.{json,pdf,png}` (130 units).
- Exclusion sets: `data/runs/x105/x105_x85_exclusions.json` (`2305ef73…`),
  `data/runs/x114/x114_x85_exclusions.json`.
- Frozen inputs: `data/runs/x103/x103_inputs.json` (per model the snapshot, the
  x88 hook block and boundary ids, the direction artifact and its sha256, the
  rendered-chat sha256 of all 520 x101 texts, and the x85 prompt/word sets);
  `data/runs/x114/x114_inputs.json` (the same for the 288 x114 texts).
- Per model: `data/runs/x103/<tag>/x103_projections.json` (390 cells: proj,
  shift, per-token parts, the site audit) and `x103_boundary_deltas.npz`
  (fp16 insertion-induced boundary deltas, 27 MB in total, kept for the
  random-direction control); `data/runs/x114/<tag>/x114_projections.json` (216
  cells) and `x114_boundary_deltas.npz` (14 MB in total).
- Code: `code/x103_common.py`, `x103_extract.py`, `x103_analyze.py`
  (`--cohort {extended,x101_leak_free,full,all}`), `x103.sbatch`; the x114
  stages `code/x114_extract.py`, `x114_prepare.py`, `x114.sbatch` and the mode
  table `code/x114_cohorts.py`; the exclusion sets from
  `code/x105_exclusions.py` and `x114_build_cohort.py`; tests
  `code/tests/test_x103_direction_projection.py` (8) and
  `test_x114_cohort_topup.py` (9), in the 127-test suite. Directions, hook,
  model registry and chat rendering are imported from x88 / x87 / x101; the
  behavior loaders from x83 / x101 / x114.
