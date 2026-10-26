# x104: GuardReasoner-8B re-judges every OR-Bench response of x87/x88/x89 — every reported delta keeps its sign, Toxic non-inferiority now holds for the boundary write at its deployment dose, and the rank-one deep edits on Qwen3.5-0.8B/2B read larger

Parent: [Experiment index](README.md). Re-measurement of the OR-Bench
refusal endpoints of [x87](x87-external-safety-overrefusal-eval.md),
[x88](x88-eight-model-deep-dose-transfer.md) and
[x89](x89-positive-dose-external-extension.md) with the judge
[x102](x102-orbench-judge-agreement.md) found closest to XSTest's human
labels, GuardReasoner-8B. The three result tables below are the x87/x88/x89
analyses rerun on its labels; StrongREJECT, IFEval and GSM8K are copied
through unchanged, so every number not shown here is the one the owning
file reports.

## Question

x102 ranked GuardReasoner-8B first among the single judges on XSTest's human
labels, ahead of the unedited Qwen3.5-4B that judged x87/x88/x89 (agreement
and κ per instrument in
[x102's XSTest table](json/x102-orbench-judge-agreement__xstest-combinations.json)
and its [lead findings](x102-orbench-judge-agreement.md)), and found
that the Qwen judge understates the Hard-1K drop of the rank-one deep edits
on Qwen3.5-0.8B and 2B. x104 asks what the
three result tables look like when GuardReasoner-8B reads all 96,726
OR-Bench responses (49 model-conditions × 1,974 prompts) instead of the
300-prompt sample.

## Findings

- **96,694 of 96,726 responses carry a GuardReasoner verdict.** 163 greedy
  traces reached the 1,024-token cap without an `Answers:` block; sampling
  at temperature 0.6 recovered 131 of them (93 at seed 1, 22 at seed 2, 16
  at seed 3) and 32 stay unparsed: 12 on Qwen3.5-2B, 10 on Qwen3.5-0.8B,
  8 on Phi-4-mini, one each on Qwen3.5-9B and Qwen3.5-4B, 25 of them in
  the rank-one conditions. The smallest paired cell is Qwen3.5-0.8B
  `x75_m6` Hard-1K, n = 1,313.
- **The two judges agree on 95.8% of the parsed rows** (Hard-1K 94.9% of
  64,610, Toxic 97.6% of 32,084); GuardReasoner's pooled
  refusal rate is 0.660 against 0.667 on Hard-1K and 0.901 against
  0.908 on Toxic. On the 10,479 x102 sample rows both runs parsed,
  this run agrees with x102's GuardReasoner arm on 99.7%, and the 21 rows
  x102 left unparsed are exactly the 21 sample rows whose greedy trace
  hit the cap here.
- **Every one of the 54 reported edited − base deltas keeps its sign**, and
  the two gate verdicts that change both move in the write's favour:
  `x86_full` and `x86_m075` now pass Toxic non-inferiority (one-sided
  lower bounds −0.0168 and −0.0183, inside the 0.02 margin, against
  −0.0229 and −0.0214 under the incumbent). `x86_full` therefore clears
  Hard-1K superiority, Toxic and IFEval, and fails only StrongREJECT
  (+0.0240, unchanged); `x75_m3` is unchanged (Hard-1K ✓, StrongREJECT ✓,
  Toxic ✗ at −0.0275, IFEval ✗ at −0.0388); `successful_candidates` stays
  empty.
- **Qwen3.5-4B reads 0.003–0.017 less Toxic cost at every suppression
  dose and a smaller rank-one Hard-1K effect.** The deployment-dose Hard-1K deltas are −0.0288 (write;
  incumbent −0.0318) and −0.0652 (edit; −0.0781), 7.4% of the 0.8855 base
  rate; the deep boundary write reaches −0.2146 at dose −3 (−0.2108) with
  Toxic −0.0412 (−0.0519), and the rank-one edit −0.1653 at scale −7
  (−0.1782) with Toxic −0.0305 (−0.0473). The positive side is
  dose-ordered as before (+0.0243 / +0.0531 / +0.0622 at write +0.5 / +1.0
  / +1.5; +0.0152 / +0.0394 / +0.0478 at edit +1 / +2 / +3).
- **The rank-one deep edits on the two smallest Qwen models read larger,
  as x102 predicted.** Qwen3.5-0.8B `x75_m6` Hard-1K −0.1333 (incumbent
  −0.0197) and Toxic −0.2217 (−0.1115); Qwen3.5-2B `x75_m6` Hard-1K −0.3645
  (−0.2980) and Toxic −0.2435 (−0.2061); Llama-3.2-3B `x75_m6` Toxic
  −0.4214 (−0.3618). The boundary write at −1.5 moves by at most 0.019 between
  the two judges on any model and endpoint (Hard-1K −0.0121 on Llama to −0.2449 on
  Qwen3.5-2B; Toxic −0.0046 to −0.1023), and the 4B → 9B → 27B ordering
  of the rank-one effect at −3 holds (−0.0652, −0.0440, −0.0417).
- **Base rates move by at most 0.056.** Qwen3.5-4B Hard-1K 0.8855 (incumbent
  0.8802), Toxic 0.9863 (0.9893); Qwen3.5-0.8B Hard-1K 0.4102 (0.4655);
  Llama-3.2-3B Hard-1K 0.0910 (0.1031).

Result files: `data/results/x104_external_safety_utility.json` (x87
schema), `x104_cross_model_deep_dose.json` (x88 schema),
`x104_positive_dose_extension.json` (x89 schema), each stamped with
`refusal_judge` and the 32 `unparsed_rows`; `x104_judge_comparison.json`
holds every incumbent and GuardReasoner delta side by side with the
agreement figures above.

## Qwen3.5-4B: the x87 table rebuilt

Edited − base, prompt-paired, 20,000-resample bootstraps (`rng(8701)`,
x87's seed schedule), two-sided 95% intervals; `[up]` and `[low]` are the
gate-relevant one-sided bounds (superiority α = 0.05/3, non-inferiority
α = 0.05/9, margin 0.02).

Table: [Qwen3.5-4B edited − base OR-Bench deltas and gates, incumbent judge vs GuardReasoner (the x87 table rebuilt)](json/x104-guardreasoner-orbench-rejudge__x87-qwen4b-gates.json) — both judges' deltas, GuardReasoner intervals and gate bounds, and gate verdicts for the 13 Qwen3.5-4B conditions; `x86_full` and `x86_m075` are the two whose Toxic non-inferiority verdict changes (lower bounds −0.0168 and −0.0183).

## Qwen3.5-4B: the x89 positive-dose table rebuilt

Table: [Qwen3.5-4B positive- and low-dose edited − base OR-Bench deltas, incumbent judge vs GuardReasoner (the x89 table rebuilt)](json/x104-guardreasoner-orbench-rejudge__x89-qwen4b-positive-dose.json) — Hard-1K and Toxic deltas under both judges for the seven x89 conditions; the positive side stays dose-ordered (write +0.0243 / +0.0531 / +0.0622, edit +0.0152 / +0.0394).

## Eight-model transfer: the x88 tables rebuilt

Table: [Eight-model OR-Bench base refusal rates, incumbent judge → GuardReasoner (the x88 base rates rebuilt)](json/x104-guardreasoner-orbench-rejudge__x88-base-rates.json) — Hard-1K and Toxic base rates per model under both judges; the largest move is Qwen3.5-0.8B Hard-1K, 0.4655 → 0.4102.

Table: [Eight-model edited − base OR-Bench deltas, incumbent judge vs GuardReasoner (the x88 transfer table rebuilt)](json/x104-guardreasoner-orbench-rejudge__x88-transfer-deltas.json) — `x86_m150`, `x75_m3` and `x75_m6` deltas per model under both judges, with GuardReasoner intervals; the rank-one `x75_m6` edits on Qwen3.5-0.8B (Hard-1K −0.1333 vs −0.0197) and 2B (−0.3645 vs −0.2980) read larger.

## Design, as run

- **Rows.** `code/x104_build_rows.py` collected every (experiment, model,
  condition, id) row of the 49 OR-Bench judgment files with its prompt and
  response into `data/runs/x104/x104_rows.jsonl` (96,726 rows), verifying
  the generation sha256 recorded in each judgment file; the manifest
  registers each source file's sha256, the row file's sha256 and the
  sha256 over the sorted row keys, and every later stage refuses an
  unregistered row file.
- **Judge.** x102's GuardReasoner arm, imported from `x102_common`:
  `yueliu1999/GuardReasoner-8B` revision `f3f57a9c…`, the model card's
  instruction as raw text with BOS, bf16, greedy, 1,024 new tokens
  (mean 295), batch 48, stopping on `<|eot_id|>` and `<|end_of_text|>`,
  `Completion: refusal/compliance` parsed from the trace. Eight shards
  (every eighth row of the key-sorted list; 12,091 or 12,090 rows each),
  append-only JSONL checkpointed after every batch and resumed by row
  key. A greedy trace that reached the cap without an `Answers:` block was
  re-decoded with sampling (temperature 0.6, top-p 1, seeds 1, 2, 3 in
  order, stopping at the first parsable trace); the re-decodes live in
  the shard's resample file with the greedy trace kept beside them.
- **Analysis.** `code/x104_analyze.py` merges the shards into
  `x104_judgments.jsonl` (full traces) and `x104_labels.jsonl` (the fields
  the analysis reads, with the incumbent label beside each row), then runs
  `x87_analyze`, `x88_analyze` and `x89_analyze` unchanged except that
  their OR-Bench reader returns the GuardReasoner refusal bit and their
  pairing drops the unparsed rows together with their partners; every
  other endpoint, seed, alpha and margin is the owning experiment's.
  Fed the incumbent labels, the same code reproduces
  `x87_external_safety_utility.json`, `x88_cross_model_deep_dose.json`
  and `x89_positive_dose_extension.json` byte for byte. The incumbent
  files stay as the old-judge record.

## Artifacts

- Code: `code/x104_common.py`, `code/x104_build_rows.py`,
  `code/x104_judge.py`, `code/x104_analyze.py`, `code/x104.sbatch`.
- Data: `data/runs/x104/x104_manifest.json`, `x104_labels.jsonl`,
  `x104_summary_shard{0..7}of8.json`; gitignored and regenerable:
  `x104_rows.jsonl` (233 MB), `x104_judgments.jsonl` (209 MB) and the
  shard and resample files.
- Results: the four `data/results/x104_*.json` files above. The draft's
  `fig_dose.py` and `fig_transfer.py` read the x104 files.
