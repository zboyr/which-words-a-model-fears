# Confusable prompt-pool mining and filtering

Parent: [Experiment index](README.md). This is a property of the prompt
pools and their auditors, not of any model; it is the paper's opening
observation that the lexical route could not be filtered away.

## What was done

- Goal: build a harmful/benign prompt pool that a lexical (TF-IDF) classifier
  cannot separate by source, so later hidden-state probes are tested on
  prompts a text baseline can't trivially win on.
- Source corpora pooled into `data/pools/pool.parquet`: wildjailbreak
  mirror, or-bench, FalseReject, jbb, xstest, toxicchat, phtest, salad,
  alert, coconot.
- Pairwise screen: TF-IDF (word 1-2gram) + LogisticRegression, 5-fold CV
  source-AUC per source pairing, cap 5000/side, seed 42
  (`data/results/auc_pairings.json`) — finds which harmful/benign source
  pairs are already lexically confusable.
- Hand-picked combined pool (`code/mining/s3_combined_pool.py`): quota
  drawn across the lowest-AUC pairings into one pool, plus a leave-one-out
  sensitivity pass (`data/results/combined_pool_auc.json`).
- AFLite-style iterative adversarial filtering
  (`code/mining/s{4,5,6}_*.py`) on two variants:
  `A_wjb_adv_iter` (WildJailbreak-adversarial-only, n=161430 start) and
  `B_multisource_iter` (n=102009 start, includes alert/coconot/jbb/orbench/
  salad/toxicchat/wjb/xstest/falsereject/phtest). Each round: 5-fold
  out-of-fold TF-IDF(word)-LR scoring, keep the harder half of the pool
  (SHRINK=0.5) each round until either side < 5000.
- Two final-cut methods were compared on the last-round pool: hardest
  5000/side by OOF score ("hardness-tail") vs a random balanced 5000+5000
  draw. **Hardness-tail was rejected**: ranking by the same OOF scores used
  to shrink the pool overfits to that scoring, so an honest retrain on the
  cut bounces back up (or inverts) instead of staying near the target — a
  random draw from the final round avoids that feedback loop.

## What came out

- `data/results/auc_pairings.json`: easiest pairings ~1.0 (falsereject vs
  salad attack_enhanced = 0.9999); hardest jbb harmful/benign = 0.5875,
  xstest = 0.6283; wjb adversarial = 0.9472.
- `data/results/combined_pool_auc.json`: hand-mixed pool auc_mean = 0.9123;
  LOO shows wjb_adv dominates confusability (removing it drops AUC to
  0.8464).
- Round trajectory (OOF word-TFIDF-LR AUC): A: r1 n=161430 auc=0.9805 ->
  r2 80715/0.9395 -> r3 40357/0.7202 -> r4 20178/0.5623. B: r1 102009/0.9693
  -> r2 51004/0.8998 -> r3 25501/0.5707 -> r4 12750/0.5969.
- Final-subset honest retrain (word/char TF-IDF-LR CV AUC), hardness-tail
  cut: A = 0.6951/0.7016; B = 0.2966/0.4825 (B's word AUC is below chance —
  inverted, the clearest evidence against hardness-tail cuts).
- Final-subset honest retrain, random cut: A = 0.5719/0.6034; B =
  0.6078/0.6468 — closer to (but still above) the ~0.51 target used later
  in the pipeline.
- Canonical output: `data/pools/mined_pool_A_rand.json`
  (5000 harmful + 5000 benign, source `wjb`, subtype
  `adversarial_harmful`/`adversarial_benign`).
- Downstream probe re-audit of this filtering line
  (`data/results/confusable_A_auc.json`): despite the low TF-IDF AUC,
  hidden-state MLP-concat probes still separate risk well
  (AUC_risk = 0.7651) — confusability filtering suppresses surface wording,
  not the hidden-state signal the benchmark exists to probe.

## What the frozen prompt set inherits

- Nothing directly: the freeze is decided downstream — see
  [dataset freeze](../dataset-freeze.md). This file only supplies the
  starting pool (`data/pools/mined_pool_A_rand.json`) that gets extended to
  15k in [scale-out and validity](dataset-scaleout-validity.md).

## Artifacts present here

`data/results/auc_pairings.json`, `combined_pool_auc.json`,
`confusable_A_auc.json`, and `data/pools/mined_pool_A_rand.json`. The
per-round filtering outputs (`adversarial_filter_results.json`,
`iterative_filter_results.json`, the round parquets) were not carried into
this project: the round trajectory above is the run record.
