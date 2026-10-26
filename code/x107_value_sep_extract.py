"""x107: store what attention reads from the appended word on Qwen3.5-4B, for a
separability analysis of harm vs neutral words with held-out words.

For every H and N text of the x85 dense grid and each of the 8 softmax-attention layers:
  v_word      value vectors of the 4 kv heads at the word's last token          [4, 256]
  c_imend     gated OV write from the word span into <|im_end|>, summed over heads  [2560]
  c_tbg       the same into the last token (\\n\\n)                                 [2560]
  attn_imend  the whole attention-module output at <|im_end|> (all keys)          [2560]
Reuses the capture machinery of x107_qkov_extract.py (patched eager attention, gate hook).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import x107_qkov_extract as pilot  # noqa: E402  (patches eager_attention_forward on import)
from x85_common import read_json  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoTokenizer  # noqa: E402

ATTN_OUT: dict[int, torch.Tensor] = {}


def make_attn_out_hook(layer_idx: int):
    def hook(_module, _inputs, output):
        ATTN_OUT[layer_idx] = output[0][0].detach().float()  # [T, D]
        return output

    return hook


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="<workdir>/e2_temp")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    base = Path(args.base)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort = read_json(base / "data/runs/x85/x85_cohort.json")
    source = cohort["model_snapshot"]
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        source, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True, attn_implementation="eager"
    ).eval()
    text_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    layers_mod = text_model.layers
    attn_layers = [i for i, t in enumerate(text_model.config.layer_types) if t == "full_attention"]
    head_dim = layers_mod[attn_layers[0]].self_attn.head_dim
    n_heads = text_model.config.num_attention_heads
    n_kv = text_model.config.num_key_value_heads
    n_rep = n_heads // n_kv
    for l in attn_layers:
        layers_mod[l].self_attn.q_proj.register_forward_hook(pilot.make_gate_hook(l, head_dim))
        layers_mod[l].self_attn.register_forward_hook(make_attn_out_hook(l))
    o_proj = {
        l: layers_mod[l].self_attn.o_proj.weight.detach().float().view(-1, n_heads, head_dim).permute(1, 0, 2)
        for l in attn_layers
    }
    cells = cohort["cells"]
    hn = [c for c in cells if c["condition"] in ("H", "N")]
    if args.limit:
        rng = np.random.default_rng(1073)
        hn = [hn[i] for i in sorted(rng.choice(len(hn), size=args.limit, replace=False))]
    n = len(hn)
    L = len(attn_layers)
    D = text_model.config.hidden_size
    v_word = np.zeros((n, L, n_kv, head_dim), dtype=np.float16)
    c_imend = np.zeros((n, L, D), dtype=np.float16)
    c_tbg = np.zeros((n, L, D), dtype=np.float16)
    attn_imend = np.zeros((n, L, D), dtype=np.float16)
    a_imend = np.zeros((n, L, n_heads), dtype=np.float32)
    cond = np.asarray([0 if c["condition"] == "H" else 1 for c in hn], dtype=np.int8)
    prompt_index = np.asarray([c["prompt_index"] for c in hn], dtype=np.int16)
    word_index = np.asarray([c["word_index"] for c in hn], dtype=np.int16)
    prompt_id = np.asarray([c["prompt_id"] for c in hn], dtype=np.int32)
    t0 = time.time()
    for i, cell in enumerate(hn):
        ATTN_OUT.clear()
        res = pilot.run_text(model, tokenizer, cell, model.device)
        W = res["word_tokens"]
        J = res["tail"]
        im_end, tbg = J[0], J[-1]
        for li, l in enumerate(attn_layers):
            cap = res["capture"][l]
            v_word[i, li] = cap["v"][0][::n_rep, W[-1], :].cpu().numpy().astype(np.float16)
            attn = cap["attn"][0]  # [H, T, T]
            for name, q in (("imend", im_end), ("tbg", tbg)):
                aw = attn[:, q, :][:, W]  # [H, |W|]
                u = torch.einsum("hw,hwd->hd", aw, cap["v"][0][:, W, :])  # [H, d]
                g = res["gates"][l][0][q]  # [H, d]
                c = torch.einsum("hd,hDd->D", u * g, o_proj[l])
                (c_imend if name == "imend" else c_tbg)[i, li] = c.cpu().numpy().astype(np.float16)
            a_imend[i, li] = attn[:, im_end, :][:, W].sum(-1).cpu().numpy()
            attn_imend[i, li] = ATTN_OUT[l][im_end].cpu().numpy().astype(np.float16)
        if i % 200 == 0 or i == n - 1:
            print(f"[{i + 1}/{n}] {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(
        out_dir / "x107_value_sep.npz", v_word=v_word, c_imend=c_imend, c_tbg=c_tbg, attn_imend=attn_imend, a_imend=a_imend,
        cond=cond, prompt_index=prompt_index, word_index=word_index, prompt_id=prompt_id,
    )
    meta = {"attention_layers": attn_layers, "n": n, "limit": args.limit, "elapsed_s": time.time() - t0,
            "word_folds": {w["harm"]: w["word_fold"] for w in cohort["words"]},
            "prompt_folds": {str(p["id"]): p["prompt_fold"] for p in cohort["prompts"]},
            "words": [[w["harm"], w["neutral"]] for w in cohort["words"]]}
    (out_dir / "x107_value_sep_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("n", "elapsed_s")}))


if __name__ == "__main__":
    main()
