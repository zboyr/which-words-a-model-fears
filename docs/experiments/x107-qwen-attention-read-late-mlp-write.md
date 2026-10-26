# x107: on Qwen3.5-4B the appended harm word is not attended more than its neutral control; attention carries a harm feature that separates unseen words at 0.92–0.98, and the late MLPs write 73 % of the boundary shift

Parent: [Experiment index](README.md). Asks where, inside the model, the
lexical reaction of [x85](x85-crossed-keyword-prompt-latent-decomposition.md)
comes from: does the generation boundary attend to the inserted harm word
more than to a tokenizer-matched neutral word in the same slot (an
attention-allocation account), or does it attend equally and react to what
the word is (a content account)? The question is answered on the x85 crossed
grid itself, so every quantity here sits next to x85's behaviour and its
boundary projection. The first job measures the softmax-attention reads of the
word and the per-layer writes along the x86 consensus direction; the second
asks whether what attention reads is linearly separable between harm and
neutral words on words it was never fitted on.

**Finding.** The harm word receives the same attention as its neutral control
at `<|im_end|>` (summed over all 128 heads 7.17 vs 7.36, Δ −0.19 [−0.61,
+0.28]) and a little more at the two tokens after it; given the same query,
the harm word's key is *less* attractive than the neutral word's (key side
−0.39 logits, query side +0.48). The direct reads of the word by the softmax
heads write 1 % of the H − N shift along the boundary direction; the 30 MLPs
write 73 %, the 23 Gated DeltaNet mixers 24 %, the 7 softmax-attention mixers
3 %, nearly all of it from layer 18 on. No attention statistic predicts the
`refusal2` change after prompt and word fixed effects (|ρ| ≤ 0.13) while the
boundary projection does (ρ = +0.65, the x85 value). What the heads read is
nonetheless a harm feature: the value vectors at the word token separate harm
from neutral words at 0.92–0.98 held-out accuracy in every attention layer
(the token embedding alone 0.90; label-flip control 0.45), and the margin of
that separation only begins to rank the words' refusal effects at the
boundary tokens in the late layers (ρ 0.5), where the MLPs write along the
direction. The reaction is a content pathway: the token mixers deliver the
word's identity to the tail, and the late MLPs convert it into the boundary
direction. Nothing here is routed to
[../headline-numbers.md](../headline-numbers.md).

## Design, as run

- **Model and texts.** `Qwen/Qwen3.5-4B` at the x85 snapshot (`851bf6e8…`),
  bf16, loaded through `AutoModelForImageTextToText` with eager attention so
  the attention weights exist. The model is a hybrid: 32 blocks, 24 Gated
  DeltaNet token mixers and 8 softmax-attention layers (3, 7, 11, 15, 19, 23,
  27, 31) with 16 query heads over 4 key/value heads, head dimension 256, a
  sigmoid output gate from the second half of `q_proj`, q/k RMSNorm and
  partial rotary. The full x85 dense grid (`data/runs/x85/x85_cohort.json`):
  96 prompts × 24 harm/neutral word pairs, the word appended as
  `prompt + " " + word`, 96 O + 2,304 H + 2,304 N = 4,704 forwards
  (5.6 min on one GPU). Rendering by `x85_common.format_user_chat`;
  every rendered chat's sha256 and token positions are checked against the
  x85 cohort audit, and the 9 tail tokens from `<|im_end|>` to the final
  `\n\n` are asserted per text.
- **Direction and read site.** The x87 consensus of the x86 fold directions
  (`data/runs/x86/x86_directions.npz`, read with
  `x87_common.consensus_x86_vectors`), one vector per boundary token
  (`</think>`, `\n\n`; norms 6.35, 7.34), at the block-29 output, x86's site.
- **Attention measures** (`code/x107_qkov_extract.py`), per softmax layer,
  head and tail query token: the attention mass `a` to the word's token span;
  the pre-softmax score of the word (logsumexp over its tokens) split into a
  key side `q_N·k_H − q_N·k_N` (the paired text's query, this text's key) and
  a query side `q_H·k_H − q_N·k_H`, each H/N pair compared at the same
  prompt; the head's write from the word span into the tail token
  `c = W_O^h(σ(gate) ⊙ Σ_t a_t v_t)`, its norm, and its projection on the
  direction at the two boundary tokens; attention between tail tokens; and,
  per decoder block, the token-mixer and MLP writes at the boundary tokens
  projected on the direction (forward hooks on `self_attn` / `linear_attn` /
  `mlp`). The direct-write difference `⟨c_H − c_N, μ⟩` is split into a QK term
  (mass differs, content averaged) and an OV term (content differs, mass
  averaged). Sanity in the job: the additive mask leaks 0.0 above the diagonal
  in every layer; the per-layer writes summed over blocks 0–29 reproduce the
  block-29 H − N difference exactly (13.67 = 13.67).
- **Separability** (`code/x107_value_sep_extract.py`, 4,608 H/N
  texts, 5.3 min): per attention layer the value vectors of the 4 kv heads at
  the word's last token (4 × 256), the gated OV write from the word span into
  `<|im_end|>` and into the final `\n\n` (2,560 each), the whole
  attention-module output at `<|im_end|>` (all keys), and the mass to the
  word; plus x85's stored residual states (`data/runs/x85/features/`, states
  0–32 at the word's last token, `<|im_end|>` and the two boundary tokens).
  Classifier (`code/x107_value_sep_analyze.py`): ridge regression to ±1
  (λ = 10, features standardised on the training fold, dual form when the
  dimension exceeds the sample), four held-out word folds = the x85 word
  folds (train 18 pairs × 96 prompts, test 6 pairs never fitted × 96 prompts);
  held-out accuracy and AUC; "word ρ" = Spearman over the 24 words between the
  per-word margin (mean H score − mean N score) and the word's x85 mean
  `refusal2` change `r_H − r_N` over prompts. Control: the labels of a random
  half of the word pairs flipped, pairwise.
- **Behaviour and intervals** (`code/x107_qkov_analyze.py`). `refusal2` from
  the x85 judge rows (`x85_refusal_{H,N}.jsonl`; the one N cell without a
  judgment is dropped): the 576 x85 behaviour edges, Spearman raw and after
  prompt + word fixed effects; 24-word means; two-way cluster bootstrap over
  prompts × words for every interval; prompt / word / interaction variance
  shares of the head masses on the 96 × 24 grid.

## Results

**Boundary projection** (block 29, two tokens summed): H − O +13.4, N − O
−0.3, H − N +13.7 [10.4, 16.9]; residualised Spearman with the `refusal2`
change on the 576 edges +0.65 (raw +0.77), as in x85.

**Who writes the H − N shift along the direction** (blocks 0–29):

| writer | Σ ⟨write_H − write_N, μ⟩ | share |
|---|---|---|
| 7 softmax-attention mixers (3 … 27) | +0.39 | 3 % |
| 23 Gated DeltaNet mixers | +3.24 | 24 % |
| 30 MLPs | +10.04 | 73 % |
| sum | +13.67 | the block-29 difference, +13.67 |

Blocks 0–17 together write +0.6; the rest comes from block 18 on (block 28:
mixer +0.84, MLP +1.92; block 29: +0.46, +1.61; block 22 MLP +1.06). The part
of the attention mixers' write that is read directly from the word's tokens
is +0.07 [+0.03, +0.11], 1 % of the shift, with a QK term of −0.03 [−0.07,
+0.01] and an OV term of +0.10 [+0.07, +0.13]. The heads do read the word (the
norm of their summed write from the word span into the final `\n\n` is 6.2 for
H and 9.0 for N, against a boundary shift norm of 10–13.5): what they read
lies off the direction.

**Attention mass to the word, H vs N**, summed over all 128 heads:

| tail query token | Σ a_H | Σ a_N | Δ [95 % CI] |
|---|---|---|---|
| `<\|im_end\|>` | 7.17 | 7.36 | −0.19 [−0.61, +0.28] |
| `\n` | 8.00 | 5.93 | +2.08 [−0.50, +4.46] |
| `<\|im_start\|>` | 6.33 | 5.41 | +0.92 [+0.40, +1.50] |
| `</think>` | 1.14 | 1.17 | −0.03 [−0.14, +0.10] |
| final `\n\n` | 1.72 | 2.24 | −0.52 [−0.88, −0.24] |

Of the 21 heads with mean mass > 0.1 from `<|im_end|>` to the word, 12 attend
the harm word more and 9 the neutral word more; the two largest differences
go the neutral way (L19.H14 0.46 vs 0.66, L19.H2 0.22 vs 0.36). The largest
harm-favouring head is a small-mass head (L15.H10 from `<think>`, 0.11 vs
0.02). For the top heads the variance of the mass on the 96 × 24 grid splits
0.2–0.3 prompt / 0.2–0.5 word / 0.3–0.6 interaction.

**Key side vs query side.** Averaged over heads and tail tokens the key side
is −0.39 logits and the query side +0.48; at `<|im_end|>` per layer the key
side is −0.4 to −2.1 and the query side +0.4 to +2.4. Given the same query the
harm word's key is less attractive than the neutral word's; the attention
differences that exist come from the tail token's own state, which already
differs at layer 3, the first softmax layer (query side +0.43), because the
DeltaNet blocks 0–2 have carried the word to the tail before any softmax
attention runs.

**Behaviour.** Spearman with the `refusal2` change on the 576 edges, after
prompt + word fixed effects: boundary projection +0.65; Σ direct
`⟨c_H − c_N, μ⟩` −0.05; QK term −0.08; OV term +0.02; Σ Δa at `<|im_end|>`
−0.00; Σ Δa over all tail tokens −0.02; Σ a_H over all tail tokens +0.03; the
single heads L19.H14 −0.13, L19.H2 −0.03, L19.H3 +0.00. At word level (24
words) the projection ranks the words' mean `refusal2` change at ρ = +0.93,
the attention statistics at +0.40 to +0.44.

**Separability of what attention reads**, held-out words (accuracy / AUC /
word ρ):

| source | accuracy (AUC) | word ρ |
|---|---|---|
| residual stream at the word's last token, state 0 (embedding) | 0.90 (0.98) | +0.03 |
| residual stream at the word's last token, states 4–32 | 0.92–0.99 (≥ 0.98) | −0.21 to +0.39 |
| value vectors v of the 4 kv heads at the word token, each of the 8 attention layers | 0.92–0.98 (≥ 0.98) | −0.19 to +0.41 |
| OV write from the word span into `<\|im_end\|>`, layers 3–19 | 0.83–0.90 (0.88–0.96) | +0.17 to +0.44 |
| the same, layers 23–31 (mass to the word ≤ 0.1 per layer) | 0.60–0.86 | −0.36 to +0.09 |
| whole attention output at `<\|im_end\|>` (all keys), layers 7–31 | 0.92–0.98 (≥ 0.97) | +0.19 to +0.49 |
| residual stream at `<\|im_end\|>`, states 8–28 | 0.95–0.99 | +0.38 to +0.51 |
| residual stream at the two boundary tokens, states 20–28 | 0.93–0.98 | +0.42 to +0.55 |
| control: word-pair labels flipped at random per pair, state 20 | 0.45 (0.48) | — |

Per attention layer, v at the word token: L3 0.94, L7 0.98, L11 0.94, L15
0.92, L19 0.98, L23 0.96, L27 0.94, L31 0.97. The content attention reads
from the word is linearly separable between harm and neutral words, the
separation generalises to words never fitted on, and it is present from the
token embedding onward. The separating direction is not the refusal
direction: at the word token and in v the margin barely ranks the words'
refusal effects, and it does so only at the boundary tokens in the late
layers, where the MLPs write along the boundary direction.

## Artifacts

- `data/runs/x107/x107_qkov.npz` (180 MB, in the Hugging Face dataset), `x107_qkov_meta.json`,
  `x107_qkov_analysis.{md,json}` — the attention run and its full report
  (per-head tables, tail-to-tail attention, per-layer writes, word table).
- `data/runs/x107/x107_value_sep.npz` (539 MB, in the Hugging Face dataset),
  `x107_value_sep_meta.json`, `x107_value_sep_analysis.{md,json}` (attention
  sources and residual states 0 and 20), `x107_residual_sep_analysis.{md,json}`
  (residual states 0–32 at the four positions).
- Code: `code/x107_qkov_extract.py` (+ `x107_qkov.sbatch`),
  `code/x107_qkov_analyze.py`, `code/x107_value_sep_extract.py`
  (+ `x107_value_sep.sbatch`), `code/x107_value_sep_analyze.py`. Inputs are
  x85's cohort, judge rows and hidden-state shards and x86's directions; no
  new generation and no new judging.
