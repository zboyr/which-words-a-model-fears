# Scale-out to the 15k pools and validity checks

Parent: [Experiment index](README.md). Continues
[dataset-mining.md](dataset-mining.md); the 15k-versus-20k contrast here is
the paper's "growing the pool restores separability" point.

## What was done

- Boundary-extension mining: extend `data/pools/mined_pool_A_rand.json`
  (see [dataset-mining.md](dataset-mining.md)) toward 10k/15k/20k per class
  by drawing further 5000 harmful+5000 benign batches from unused mining
  leftovers, disjoint id ranges (A2 20000+, A3 40000+, A4 60000+), each
  re-scored by a fresh 5-fold CV word+char TF-IDF filter alone and combined
  with all prior pools.
- A2 used a random cut (as A_rand did). A3/A4 switched to "boundary
  extension": the least-easy unused examples adjacent to the prior
  keep/drop cut, rather than a fresh random draw.
- Mining scripts: `code/mining/s1{0,1,2}_extend_pool_A{2,3,4}.py`.
  Outputs: `data/pools/mined_pool_A2_rand.json`,
  `data/pools/mined_pool_A3_bnd.json`, `data/pools/mined_pool_A4_bnd.json`
  and `data/results/extend_pool_A{2,3,4}_auc.json`. The staging scripts
  (`code/mining/t1{2,3,4,5}_*.py`) combine the kept pools into the 15k
  clustered + benign files; the 10k and 20k staging pools were
  intermediate and are not kept.
- Validity checks on the mined pools (vs the TF-IDF AUC ~0.57-0.60
  baseline that motivated them), all on pool A /
  `data/pools/mined_pool_A_rand.json`: bge-base-en-v1.5 embedding
  separability, a blind LLM judge, and the probed model's own zero-shot
  judgment.

## What came out

- A2 alone: word=0.5695/char=0.6004. A+A2 (10k): word=0.5585/char=0.6136
  (near target; random cut still worked at this scale).
- A3 alone: word=0.7131/char=0.6929 (much easier than A/A2). A+A2+A3
  (15k): word=0.5073/char=0.6164 — pooled AUC stayed low; **this is the
  canonical final 15k set**.
- A4 alone: word=0.9077/char=0.887 (near-trivial). A..A4 (20k):
  word=0.7154/char=0.7219 — combined AUC jumped well above target.
  Boundary-extension mining runs out of genuinely confusable material past
  15k (each 5k increment gets lexically easier), so the dataset was frozen
  at 15k+15k and the 20k/A4 extension was rejected.
- Embedding check (s7, bge-base-en-v1.5 CLS, 5-fold LR CV): pool A
  embedding_auc = 0.6763 (std 0.0079), cross-side NN cosine median = 0.7779
  — well above the TF-IDF floor, i.e. lexically confusable but not
  semantically collapsed.
- Blind LLM judge (s8, local Qwen3.5-9B-Uncensored, 100+100 sampled,
  seed 42, unaware of source label): mined pool A (n=200) — harmful side
  judged {harmful:63, benign:37}, benign side judged {harmful:7,
  benign:93}. An unfiltered WildJailbreak sample (n=200) scored {harmful:73,
  benign:27} / {harmful:4, benign:96} — mining costs some judge-separability
  but the pool stays well above chance separability.
- Own-model zero-shot check (s9, single-forward-pass H/B letter-logit
  softmax by the probed Llama-3.2-3B-Instruct itself): pilot n=500
  AUC=0.8168/acc=0.674; full n=5743 AUC=0.7952/acc=0.5048 — strong ranking
  AUC persists at scale; the accuracy drop is threshold miscalibration, not
  a ranking failure.
- `cp_15k_clustered_v2.json` (the harmful source pool
  [dataset-freeze.md](../dataset-freeze.md) builds from) is the same
  15,000 prompt identities as this 15k pool's original clustering file
  (`cp_15k_clustered.json`, superseded and not kept), re-judged — not a new
  mining pass.

## Artifacts present here

`data/results/extend_pool_A{2,3,4}_auc.json` (the 15k and 20k pooled word
and character AUCs the paper cites) and
`data/pools/mined_pool_A{2_rand,3_bnd,4_bnd}.json`. The validity-check
outputs (embedding separability, the blind judge, the own-model zero-shot
pass) are recorded here but their result files were not carried into this
project.

## What the frozen prompt set inherits

- The frozen harmful/benign source pools (`cp_15k_clustered_v2.json`
  `source=="toxic"`, `cp_15k_benign.json` `source=="hard"`), the retained-row
  count, the x61 mask, and the benign seed-42 prefix are all
  owned by [dataset-freeze.md](../dataset-freeze.md).
