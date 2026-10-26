"""x107: QK-vs-OV decomposition of the appended word's read on Qwen3.5-4B.

On the x85 dense grid (96 prompts x 24 harm/neutral pairs, word appended at the end), run
O, H and N through the model with eager attention and, in each of the 8 softmax-attention
layers (3, 7, ..., 31) and 16 heads, measure for every tail query token (the 9 tokens after
the word: <|im_end|> ... </think> \\n\\n):

  a        attention mass from the tail token to the word's token span
  s_own    logsumexp over the word tokens of the pre-softmax score q . k * scaling
  s_cross  the same with the query taken from the paired text (q_N . k_H for H, q_H . k_N for N)
  z        logsumexp over all keys (so a = exp(s_own - z))
  cnorm    norm of the head's write from the word span into the tail token's residual stream
           c = W_O^h ( sigmoid(gate) * sum_t a_t v_t )
  cproj    <c, mu_j> for the two boundary tokens, mu_j = x86 consensus direction (block 29)

plus the block-29 boundary states projected on the consensus direction for O/H/N.
Everything is written as scalars to one npz; the analysis is a separate script.
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

from x85_common import WORD_PAIRS, format_user_chat, locate_positions, read_json, sha256_text  # noqa: E402
from x87_common import consensus_x86_vectors  # noqa: E402

from transformers import AutoModelForImageTextToText, AutoTokenizer  # noqa: E402
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq  # noqa: E402

HOOK_BLOCK = 29
N_TAIL = 9  # <|im_end|> + the 8 audited tail tokens
CAPTURE: dict[int, dict[str, torch.Tensor]] = {}
GATES: dict[int, torch.Tensor] = {}
_ORIG_EAGER = mq.eager_attention_forward


def capture_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    if attention_mask is None or not attention_mask.is_floating_point():
        raise RuntimeError("eager attention needs the additive floating 4D causal mask")
    out, weights = _ORIG_EAGER(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
    k = mq.repeat_kv(key, module.num_key_value_groups)
    v = mq.repeat_kv(value, module.num_key_value_groups)
    scores = torch.matmul(query.float(), k.float().transpose(2, 3)) * scaling + attention_mask.float()
    CAPTURE[int(module.layer_idx)] = {
        "q": query.float(),  # [1, H, T, d]
        "k": k.float(),
        "v": v.float(),
        "scores": scores,  # [1, H, T, T] masked fp32 scores
        "attn": torch.softmax(scores, dim=-1),
    }
    return out, weights


mq.eager_attention_forward = capture_eager


def make_gate_hook(layer_idx: int, head_dim: int):
    def hook(_module, _inputs, output):
        proj = output.view(*output.shape[:-1], -1, head_dim * 2)
        GATES[layer_idx] = torch.sigmoid(proj[..., head_dim:].float())  # [1, T, H, d]

    return hook


WRITES: dict[str, torch.Tensor] = {}


def make_write_hook(name: str):
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        WRITES[name] = hidden[0, -N_TAIL:, :].detach().float()  # [9, D]
        return output

    return hook


class BlockReader:
    def __init__(self) -> None:
        self.last = None

    def __call__(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self.last = hidden[0, -2:, :].detach().float()
        return output


def run_text(model, tokenizer, cell: dict, device) -> dict:
    text = cell["prompt"]
    if sha256_text(text) != cell["prompt_sha256"]:
        raise SystemExit(f"FATAL: prompt hash mismatch for {cell['cell_id']}")
    rendered = format_user_chat(tokenizer, text)
    if sha256_text(rendered) != cell["rendered_chat_sha256"]:
        raise SystemExit(f"FATAL: rendered chat mismatch for {cell['cell_id']}")
    span = None if cell["inserted_span"] is None else tuple(cell["inserted_span"])
    ids, positions = locate_positions(tokenizer, rendered, text, span)
    if positions != cell["position_indices"] or len(ids) != cell["n_input_tokens"]:
        raise SystemExit(f"FATAL: position audit mismatch for {cell['cell_id']}")
    # word token span: tokens overlapping the inserted span (H/N); None for O
    word_tokens: list[int] = []
    if span is not None:
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        content_start = rendered.index(text)
        abs_start, abs_stop = content_start + span[0], content_start + span[1]
        word_tokens = [
            i for i, (a, b) in enumerate(encoded["offset_mapping"]) if b > abs_start and a < abs_stop and b > a
        ]
        decoded = tokenizer.decode([ids[i] for i in word_tokens])
        if decoded.strip() != cell["inserted_word"]:
            raise SystemExit(f"FATAL: word span decodes to {decoded!r} for {cell['cell_id']}")
        if word_tokens[-1] != positions["instruction_last"]:
            raise SystemExit(f"FATAL: word span end != instruction_last for {cell['cell_id']}")
    tail = list(range(positions["user_end"], len(ids)))
    if len(tail) != N_TAIL:
        raise SystemExit(f"FATAL: tail has {len(tail)} tokens for {cell['cell_id']}")
    CAPTURE.clear()
    GATES.clear()
    WRITES.clear()
    inputs = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(input_ids=inputs, return_dict=True)
    if len(CAPTURE) != 8 or len(GATES) != 8:
        raise SystemExit(f"FATAL: captured {len(CAPTURE)} attention layers / {len(GATES)} gates")
    return {
        "ids": ids,
        "positions": positions,
        "word_tokens": word_tokens,
        "tail": tail,
        "logits_last": out.logits[0, -1].float(),
        "capture": {l: {k: t.clone() for k, t in d.items()} for l, d in CAPTURE.items()},
        "gates": {l: g.clone() for l, g in GATES.items()},
        "writes": {k: t.clone() for k, t in WRITES.items()},
    }


def head_stats(res: dict, other: dict | None, o_proj: dict, mu: torch.Tensor, layers: list[int], scaling: float):
    """Per layer/head/tail-token scalars for one text, with the cross-query score from `other`."""
    W = res["word_tokens"]
    J = res["tail"]
    n_h = res["capture"][layers[0]]["q"].shape[1]
    a = torch.zeros(len(layers), n_h, len(J))
    s_own = torch.zeros_like(a)
    s_cross = torch.zeros_like(a)
    z = torch.zeros_like(a)
    a_prompt = torch.zeros_like(a)
    cnorm = torch.zeros_like(a)
    cproj = torch.zeros(len(layers), n_h, 2)
    t2t = torch.zeros(len(layers), n_h, len(J), len(J))
    content_end = W[0] if W else res["positions"]["instruction_last"] + 1
    for li, l in enumerate(layers):
        cap = res["capture"][l]
        attn = cap["attn"][0]  # [H, T, T]
        scores = cap["scores"][0]
        a_prompt[li] = attn[:, J, :content_end].sum(-1)
        t2t[li] = attn[:, J][:, :, J]
        if not W:
            continue
        a[li] = attn[:, J][:, :, W].sum(-1)
        s_own[li] = torch.logsumexp(scores[:, J][:, :, W], dim=-1)
        z[li] = torch.logsumexp(scores[:, J, :], dim=-1)
        if other is not None:
            q_other = other["capture"][l]["q"][0][:, J, :]  # [H, |J|, d]
            k_self = cap["k"][0][:, W, :]  # [H, |W|, d]
            s_cross[li] = torch.logsumexp(torch.einsum("hjd,hwd->hjw", q_other, k_self) * scaling, dim=-1)
        # head write from the word span into each tail token
        v_w = cap["v"][0][:, W, :]  # [H, |W|, d]
        u = torch.einsum("hjw,hwd->hjd", attn[:, J][:, :, W], v_w)  # [H, |J|, d]
        g = res["gates"][l][0][J]  # [|J|, H, d]
        u = u * g.permute(1, 0, 2)
        wo = o_proj[l]  # [H, D, d]
        c = torch.einsum("hjd,hDd->hjD", u, wo)  # [H, |J|, D]
        cnorm[li] = c.norm(dim=-1)
        cproj[li] = torch.einsum("hjD,jD->hj", c[:, -2:, :], mu)  # boundary tokens are the last two
    return {
        "a": a, "s_own": s_own, "s_cross": s_cross, "z": z, "a_prompt": a_prompt,
        "cnorm": cnorm, "cproj": cproj, "t2t": t2t,
    }


def write_stats(res: dict, n_layers: int, mu_unit: torch.Tensor) -> dict:
    out = {}
    for name in ("mixer", "mlp"):
        proj = torch.zeros(n_layers, 2)
        norm = torch.zeros(n_layers, N_TAIL)
        for i in range(n_layers):
            w = res["writes"][f"{name}{i}"]  # [9, D]
            proj[i] = (w[-2:] * mu_unit).sum(-1).cpu()
            norm[i] = w.norm(dim=-1).cpu()
        out[f"{name}_proj"] = proj
        out[f"{name}_norm"] = norm
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="<workdir>/e2_temp")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="number of pairs (prompt x word) for a debug run")
    args = parser.parse_args()
    base = Path(args.base)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort = read_json(base / "data/runs/x85/x85_cohort.json")
    mu_minus1, mu_tbg = consensus_x86_vectors(base / "data/runs/x86/x86_directions.npz")
    mu = torch.stack([mu_minus1, mu_tbg])  # [2, D], unnormalised; projections divide by norm
    mu_norm = mu.norm(dim=-1)
    mu_unit = mu / mu_norm[:, None]

    source = cohort["model_snapshot"]
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        source, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True, attn_implementation="eager"
    ).eval()
    text_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    layers_mod = text_model.layers
    attn_layers = [i for i, t in enumerate(text_model.config.layer_types) if t == "full_attention"]
    if attn_layers != [3, 7, 11, 15, 19, 23, 27, 31]:
        raise SystemExit(f"FATAL: unexpected attention layers {attn_layers}")
    head_dim = layers_mod[attn_layers[0]].self_attn.head_dim
    n_heads = text_model.config.num_attention_heads
    scaling = layers_mod[attn_layers[0]].self_attn.scaling
    for l in attn_layers:
        layers_mod[l].self_attn.q_proj.register_forward_hook(make_gate_hook(l, head_dim))
    reader = BlockReader()
    layers_mod[HOOK_BLOCK].register_forward_hook(reader)
    n_layers = len(layers_mod)
    for i, layer in enumerate(layers_mod):
        mixer = layer.linear_attn if layer.block_type == "linear_attention" else layer.self_attn
        mixer.register_forward_hook(make_write_hook(f"mixer{i}"))
        layer.mlp.register_forward_hook(make_write_hook(f"mlp{i}"))
    o_proj = {
        l: layers_mod[l].self_attn.o_proj.weight.detach().float().view(-1, n_heads, head_dim).permute(1, 0, 2)
        for l in attn_layers
    }  # [H, D, d], on the model device
    mu_cpu = mu_unit.to(model.device)

    cells = cohort["cells"]
    by_id = {c["cell_id"]: c for c in cells}
    prompts = cohort["prompts"]
    words = cohort["words"]
    pairs = [(p, w) for p in prompts for w in words]
    if args.limit:
        rng = np.random.default_rng(1071)
        pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), size=args.limit, replace=False))]
    n_pairs = len(pairs)
    L = len(attn_layers)
    shape = (n_pairs, 2, L, n_heads, N_TAIL)
    arrays = {k: np.zeros(shape, dtype=np.float32) for k in ("a", "s_own", "s_cross", "z", "a_prompt", "cnorm")}
    arrays["cproj"] = np.zeros((n_pairs, 2, L, n_heads, 2), dtype=np.float32)
    arrays["bproj"] = np.zeros((n_pairs, 2, 2), dtype=np.float32)  # <h_pos - h_O, mu_unit> per boundary token
    arrays["bproj_abs"] = np.zeros((n_pairs, 2, 2), dtype=np.float32)  # <h_pos, mu_unit>
    arrays["bshift_norm"] = np.zeros((n_pairs, 2, 2), dtype=np.float32)
    arrays["hn_proj"] = np.zeros((n_pairs, 2), dtype=np.float32)  # <h_H - h_N, mu_unit>
    arrays["o_a_prompt"] = np.zeros((n_pairs, L, n_heads, N_TAIL), dtype=np.float32)
    arrays["t2t"] = np.zeros((n_pairs, 2, L, n_heads, N_TAIL, N_TAIL), dtype=np.float16)
    arrays["o_t2t"] = np.zeros((n_pairs, L, n_heads, N_TAIL, N_TAIL), dtype=np.float16)
    n_layers_total = len(text_model.layers)
    for name in ("mixer", "mlp"):
        arrays[f"{name}_proj"] = np.zeros((n_pairs, 2, n_layers_total, 2), dtype=np.float32)  # <write, mu_unit> at boundary tokens
        arrays[f"{name}_norm"] = np.zeros((n_pairs, 2, n_layers_total, N_TAIL), dtype=np.float32)
        arrays[f"o_{name}_proj"] = np.zeros((n_pairs, n_layers_total, 2), dtype=np.float32)
    arrays["n_word_tokens"] = np.zeros((n_pairs,), dtype=np.int16)
    arrays["prompt_index"] = np.asarray([p["prompt_index"] for p, _ in pairs], dtype=np.int16)
    arrays["word_index"] = np.asarray([w["word_index"] for _, w in pairs], dtype=np.int16)
    arrays["prompt_id"] = np.asarray([p["id"] for p, _ in pairs], dtype=np.int32)
    sanity = {"causal_leak_max": 0.0, "mask_dtype": None, "logit_delta_HN_mean": 0.0, "n_forwards": 0}

    o_cache: dict[int, dict] = {}
    t0 = time.time()
    for idx, (p, w) in enumerate(pairs):
        pid = int(p["id"])
        if pid not in o_cache:
            o_res = run_text(model, tokenizer, by_id[f"O:p{pid}"], model.device)
            o_res["h29"] = reader.last.clone()
            o_stats = head_stats(o_res, None, o_proj, mu_cpu, attn_layers, scaling)
            o_w = write_stats(o_res, n_layers, mu_cpu)
            o_cache = {pid: {"h29": o_res["h29"], "a_prompt": o_stats["a_prompt"], "t2t": o_stats["t2t"], "writes": o_w}}  # one prompt at a time
            # causality check on this forward
            for l in attn_layers:
                attn = o_res["capture"][l]["attn"][0]
                leak = torch.triu(attn, diagonal=1).abs().max().item()
                sanity["causal_leak_max"] = max(sanity["causal_leak_max"], leak)
            sanity["n_forwards"] += 1
            del o_res
        wi = int(w["word_index"])
        res = {}
        for ci, cond in enumerate(("H", "N")):
            res[cond] = run_text(model, tokenizer, by_id[f"{cond}:p{pid}:w{wi:02d}"], model.device)
            res[cond]["h29"] = reader.last.clone()
            sanity["n_forwards"] += 1
        if res["H"]["word_tokens"] != res["N"]["word_tokens"]:
            raise SystemExit(f"FATAL: H/N word spans differ for p{pid} w{wi}")
        if len(res["H"]["ids"]) != len(res["N"]["ids"]):
            raise SystemExit(f"FATAL: H/N lengths differ for p{pid} w{wi}")
        arrays["n_word_tokens"][idx] = len(res["H"]["word_tokens"])
        stats = {
            "H": head_stats(res["H"], res["N"], o_proj, mu_cpu, attn_layers, scaling),
            "N": head_stats(res["N"], res["H"], o_proj, mu_cpu, attn_layers, scaling),
        }
        h_o = o_cache[pid]["h29"]
        for ci, cond in enumerate(("H", "N")):
            for k in ("a", "s_own", "s_cross", "z", "a_prompt", "cnorm"):
                arrays[k][idx, ci] = stats[cond][k].cpu().numpy()
            arrays["cproj"][idx, ci] = stats[cond]["cproj"].cpu().numpy()
            h = res[cond]["h29"]
            arrays["bproj"][idx, ci] = ((h - h_o) * mu_cpu).sum(-1).cpu().numpy()
            arrays["bproj_abs"][idx, ci] = (h * mu_cpu).sum(-1).cpu().numpy()
            arrays["bshift_norm"][idx, ci] = (h - h_o).norm(dim=-1).cpu().numpy()
        arrays["hn_proj"][idx] = ((res["H"]["h29"] - res["N"]["h29"]) * mu_cpu).sum(-1).cpu().numpy()
        arrays["o_a_prompt"][idx] = o_cache[pid]["a_prompt"].cpu().numpy()
        arrays["o_t2t"][idx] = o_cache[pid]["t2t"].cpu().numpy().astype(np.float16)
        for name in ("mixer", "mlp"):
            arrays[f"o_{name}_proj"][idx] = o_cache[pid]["writes"][f"{name}_proj"].numpy()
        for ci, cond in enumerate(("H", "N")):
            ws = write_stats(res[cond], n_layers, mu_cpu)
            arrays["t2t"][idx, ci] = stats[cond]["t2t"].cpu().numpy().astype(np.float16)
            for name in ("mixer", "mlp"):
                arrays[f"{name}_proj"][idx, ci] = ws[f"{name}_proj"].numpy()
                arrays[f"{name}_norm"][idx, ci] = ws[f"{name}_norm"].numpy()
        sanity["logit_delta_HN_mean"] += (res["H"]["logits_last"] - res["N"]["logits_last"]).abs().max().item()
        if idx % 50 == 0 or idx == n_pairs - 1:
            el = time.time() - t0
            print(f"[{idx + 1}/{n_pairs}] {el:.0f}s  mean bproj H {arrays['bproj'][: idx + 1, 0].sum(-1).mean():.2f}  N {arrays['bproj'][: idx + 1, 1].sum(-1).mean():.2f}", flush=True)
    sanity["logit_delta_HN_mean"] /= max(n_pairs, 1)
    sanity["mask_dtype"] = "float (additive); causal leak max above"
    np.savez_compressed(out_dir / "x107_qkov.npz", **arrays)
    meta = {
        "model_snapshot": source,
        "attention_layers": attn_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "scaling": scaling,
        "hook_block": HOOK_BLOCK,
        "tail_tokens": cohort["template_audit"]["tail_decoded"],
        "tail_query_order": ["<|im_end|>"] + cohort["template_audit"]["tail_decoded"],
        "mu_norms": mu_norm.tolist(),
        "n_pairs": n_pairs,
        "limit": args.limit,
        "word_pairs": [list(x) for x in WORD_PAIRS],
        "sanity": sanity,
        "elapsed_s": time.time() - t0,
    }
    (out_dir / "x107_qkov_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["sanity"], indent=2))


if __name__ == "__main__":
    main()
