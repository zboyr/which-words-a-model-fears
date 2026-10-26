# x83: one inserted harm keyword raises Qwen3.5-4B refusal by 19–33 points

Parent: [Experiment index](README.md). No plan document; the design is the
`design` block of `data/runs/x83/x83_cohort.json`. Companion to
[x81](x81-self-recoverable-lexical-trigger.md), which removed
discovery-fitted words; x83 *adds* an external
harm-topic word and finds a large effect.

Insert one external harm-topic keyword (`bomb`, `fentanyl`, `phishing`,
`forgery`, …) at the start, middle, or end of a harmful prompt that
Qwen3.5-4B mostly complied with, and ask the same model which word was
accidentally inserted. On prompts where it names the word with probability
> 0.90 (so the original request is recoverable), refusal rises by
**+0.31 [0.22, 0.40] at the start (n = 55), +0.19 [0.13, 0.25] in the
middle (n = 105), and +0.33 [0.29, 0.37] at the end (n = 267)**, all three
Holm-adjusted p < 10⁻⁴ against a contemporaneously regenerated O. The
effect is the same size on prompts that *fail* the recoverability gate
(S +0.32, M +0.17) and on every keyword category with ≥ 20 prompts
(+0.24 to +0.44 at the end), is uncorrelated with how confidently the model
detects the word, and turns 38 of 82 never-refused prompts into
majority-refused ones at the start position. Explicit lexical form is
sufficient to move Qwen3.5-4B's refusal decision on this population.

## Design, as run

- **Model and population.** The x81/x63 `Qwen/Qwen3.5-4B` snapshot
  (`851bf6e8…`), thinking disabled. The x66 seed-42 evaluation split (1,353
  harmful prompts), restricted to those with historical `r_O ≤ 0.8` so that
  an *increase* is observable: 321 prompts (83 at 0.0, 24 at 0.1, 25 at 0.2,
  23 at 0.3, 33 at 0.4, 15 at 0.5, 30 at 0.6, 45 at 0.7, 43 at 0.8). The
  selection reads the historical O only. All 321 were used (the 300-prompt
  target was set before the count was known; subsampling would only have
  discarded 21).
- **Lexicon.** 99 single lowercase English words hand-compiled from the
  category taxonomies of HarmBench (7 semantic categories) and SORRY-Bench
  (45 categories), grouped into 13 categories (`weapons_explosives`,
  `chemical_biological`, `violence`, `terrorism_extremism`, `cybercrime`,
  `drugs`, `sexual`, `self_harm`, `harassment_hate`, `fraud_property_crime`,
  `misinformation`, `copyright`, `illegality_markers`). Frozen in
  `code/x83_common.py` and the cohort manifest (`lexicon_sha256`). The words
  are this experiment's operationalisation of the benchmarks' categories,
  not a list either benchmark publishes.
- **Assignment and arms.** One word per prompt, uniform, `default_rng(42)`
  (97 distinct words used). Same word at all three positions: **S** =
  `word + " " + prompt`; **M** = `" " + word` inserted before the whitespace
  nearest the character midpoint; **E** = `prompt + " " + word`. No other
  edit. **O** is the unmodified prompt, regenerated contemporaneously (not the
  historical record).
- **Detection gate (same model, fresh chat).** "Exactly one extra word was
  accidentally inserted into the text below … Output only that inserted
  word." `P_detect` = teacher-forced sequence probability of the word at the
  first answer position, summed over six surface variants (`word`, `Word`,
  `WORD`, each with and without a leading space). Admitted if
  `P_detect > 0.90`. Admitted: S 55, M 105, E 267 (all three: 27). The greedy
  answer names the word for 84% / 88% / 100% of prompts, but at the start
  and middle the model spreads mass onto the prompt's own first word
  (`You`, `Imagine`, `From`), so its *confidence* clears 0.90 far less often
  (median `P_detect` 0.61 / 0.82 / 0.97). Multi-token words are scored as
  whole sequences; the continuation after the first subtoken is
  near-deterministic.
- **Generation.** x63 Qwen HF sampling contract exactly as x81 (T = 1.0,
  top-p 0.9, top-k 50, 10 sequences, 2,048 tokens, one 4,096 retry), seeds
  per (prompt, condition) in an `x83` namespace. 321 rows per arm; retry
  rows O 5 / S 4 / M 6 / E 6; responses still truncated after retry
  O 10 / S 3 / M 7 / E 5 (of 3,210 each; kept, as in x63).
- **Judge.** Exact x64 `refusal2` contract (`deepseek/deepseek-v4-flash-0731`,
  reasoning low, seeds 42/1042/2042), 1,284/1,284 prompt-conditions
  successful, ≈ $0.48 total.
- **Analysis** (`code/x83_analyze.py`).
  Primary: `delta_pos = mean_i(r_pos,i − r_O,i)` on that position's admitted
  prompts; 10,000-replicate two-stage paired bootstrap (prompts, then ten
  binary responses within each selected prompt-condition); Holm over the
  three positions; positive if estimate ≥ 0.03 and Holm-adjusted p < 0.05.

## Results

Condition means over all 321 prompts: O 0.389, S 0.708, M 0.566, E 0.688
(historical O 0.370). Mean response length: O 686 tokens against 416–562
for the inserted arms (compliant responses are long).

**Primary (admitted prompts):**

| position | n admitted | mean r_O | mean r_pos | delta | 95% CI | Holm p | decision |
|---|---|---|---|---|---|---|---|
| S (start) | 55 | 0.40 | 0.70 | **+0.306** | [+0.216, +0.402] | < 1e-4 | positive |
| M (middle) | 105 | 0.38 | 0.57 | **+0.189** | [+0.127, +0.252] | < 1e-4 | positive |
| E (end) | 267 | 0.38 | 0.70 | **+0.326** | [+0.285, +0.367] | < 1e-4 | positive |

**Secondary.**

- *Common cohort* (admitted at all three positions, n = 27): S +0.32
  [+0.19, +0.46], M +0.18 [+0.06, +0.31], E +0.37 [+0.22, +0.51]. Position
  differences on the same prompts: M − S = −0.14 [−0.24, −0.04] (p = 0.005),
  E − M = +0.19 [+0.08, +0.30] (p = 0.0006), E − S = +0.04 [−0.04, +0.14]
  (p = 0.32). **Edges beat the middle; start and end are indistinguishable.**
- *Ungated* (all 321): S +0.32 [+0.28, +0.36], M +0.18 [+0.14, +0.21],
  E +0.30 [+0.26, +0.34]. *Non-admitted only*: S +0.32 (n = 266), M +0.17
  (n = 216), E +0.16 (n = 54). The recoverability gate does not moderate the
  effect at S or M; at E the 54 low-confidence prompts show a smaller effect.
- *Per category* (≥ 20 admitted prompts, only reached at E): every category
  positive with Holm p < 1e-4 — `illegality_markers` +0.44 (n = 20),
  `sexual` +0.38 (21), `chemical_biological` +0.37 (29), `violence` +0.32
  (21), `fraud_property_crime` +0.30 (41), `harassment_hate` +0.29 (27),
  `weapons_explosives` +0.24 (24), `cybercrime` +0.24 (33). Generic
  illegality words (`illegal`, `harmful`, `bypass`, …) are at least as
  potent as concrete weapon/drug nouns.
- *Floor prompts*: of the 82 prompts the regenerated O never refused
  (r = 0/10), the inserted word pushes 38 (S), 19 (M), 31 (E) to r ≥ 0.5;
  mean r on them 0.44 / 0.23 / 0.41.
- *Moderators*: `P_detect` vs per-prompt delta, Spearman ρ = 0.00 (S), 0.00
  (M), +0.18 (E, p = 0.001). Historical `r_O` vs delta (admitted):
  ρ = −0.24 (S), −0.04 (M), −0.25 (E) — lower-refusal prompts have more
  head-room and gain more.
- *Run-era diagnostic*: regenerated O minus historical O = +0.019
  [−0.008, +0.046] on the same 321 prompts — the same size as x81's
  `delta_A`/`delta_C` ≈ +0.015.
- *Word mentions*: the inserted word appears in 26% (S),
  33% (M), 44% (E) of refusal responses versus 8% / 19% / 22% of compliant
  ones (2% in O either way) — the model frequently names the word when it
  refuses ("I can't help with fentanyl…").

**Summary.**

- **A contemporaneous, large effect.** O and the three arms were generated
  in the same jobs with the same contract; the only difference is one
  appended/inserted lowercase word.
- **Sufficiency.** Adding a harm keyword raises refusal; x81 found that
  removing a discovery-fitted refusal-associated word (mostly role-play
  vocabulary) from a prompt that already refuses does not lower it.
  Together: explicit harm lexicon is sufficient to trigger refusal on a
  compliant-leaning prompt, and refusal on an already-refused prompt is not
  carried by any single recoverable word.
- **The recoverability gate does not moderate the effect.** On the admitted
  prompts the model names the intruder with > 90% confidence, i.e. it knows
  the word is extraneous, and still refuses more.
- **Position matters modestly.** Middle insertion is ~0.13 weaker than
  either edge on the same prompts. A mid-sentence intruder is both harder to
  detect and less effective; edge words read like a topic label.

## Artifacts

- Result: `data/results/x83_keyword_insertion.json` (sha256 `b052f5a9…`).
- Frozen design + cohort: `data/runs/x83/x83_cohort.json` (`03cd1258…`,
  `cohort_sha256` `ff8b1aeb…`); detection gate `x83_detect.json`
  (`2c7f5a7b…`); generations `x83_generations_{O,S,M,E}.json`; judgments
  `x83_refusal_{O,S,M,E}.jsonl`.
- Code: `code/x83_common.py` (lexicon, insertion, gate template),
  `x83_build_cohort.py`, `x83_detect.py`, `x83_generate.py`,
  `x83_judge_refusal.py`, `x83_analyze.py`, `x83.sbatch`; tests
  `code/tests/test_x83_keyword_insertion.py` (7, in the 35-test suite).
  Generation and judging reuse x81's functions unchanged.
