"""x109 word-representation probe, GPU stage: context-free representations of every scanned word.

For each of the 71,583 scanned words (Qwen3.5-4B), with no prompt around it:
  - the static input embedding of its tokens (mean over tokens, and last token);
  - the residual stream at its last token when " " + w is fed alone, at the outputs of blocks
    3, 7, 11, 15, 19, 23, 27, 29 (the read block) and 31;
  - the value vectors (v_proj output, 4 kv heads x 256) at its last token in the eight
    softmax-attention layers;
  - the boundary projection when w alone is the whole user message, rendered with
    x106_common.render_chat (same template, date and read site as the scan).
Everything is written to data/runs/x109/wordrep/features.npz (float16) plus meta.json.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x103_common import _decoder_layers  # noqa: E402
from x109_scan_bench_gpu import Reader, forward  # noqa: E402
from x109_scan_common import RUN, SCAN_JSON, TAG, atomic_json, read_json  # noqa: E402
from x106_common import (  # noqa: E402
    assert_boundary,
    hook_site,
    inputs_path,
    load_mu,
    load_tag_model,
    load_tag_tokenizer,
    projection_of,
    projections_path,
    render_chat,
)

OUT = RUN / "wordrep"
BLOCKS = [3, 7, 11, 15, 19, 23, 27, 29, 31]
BATCH = 2048


class LastTok:
    """Forward hook: keep the last position of the module output (float16, cpu)."""

    def __init__(self) -> None:
        self.last = None

    def __call__(self, _m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        self.last = h[:, -1, :].detach().to(torch.float16).cpu()
        return output


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    entry = read_json(inputs_path())["models"][TAG]
    tok = load_tag_tokenizer(entry["snapshot"], TAG)
    date = read_json(projections_path(TAG))["rendering"]["date"]
    boundary_ids = [int(v) for v in entry["boundary_ids"]]
    scan = read_json(SCAN_JSON)
    words = scan["candidates"]["words"]
    W = len(words)
    ids = [tok(" " + w, add_special_tokens=False)["input_ids"] for w in words]
    if [len(x) for x in ids] != list(scan["candidates"]["n_tokens"]):
        raise SystemExit("FATAL: tokenization differs from the scan")

    model = load_tag_model(entry["snapshot"], TAG).eval()
    dev = model.device
    layers = _decoder_layers(model)
    attn_layers = [i for i, l in enumerate(layers) if hasattr(l, "self_attn")]
    print("attention layers", attn_layers, flush=True)
    emb = model.get_input_embeddings().weight

    feats = {
        "emb_mean": np.zeros((W, emb.shape[1]), np.float16),
        "emb_last": np.zeros((W, emb.shape[1]), np.float16),
    }
    hooks, handles = {}, []
    for b in BLOCKS:
        hooks[f"block{b}"] = LastTok()
        handles.append(layers[b].register_forward_hook(hooks[f"block{b}"]))
    for a in attn_layers:
        hooks[f"v{a}"] = LastTok()
        handles.append(layers[a].self_attn.v_proj.register_forward_hook(hooks[f"v{a}"]))

    by_len = defaultdict(list)
    for i, x in enumerate(ids):
        by_len[len(x)].append(i)
    with torch.inference_mode():
        for i, x in enumerate(ids):
            e = emb[torch.tensor(x, device=dev)].float()
            feats["emb_mean"][i] = e.mean(0).half().cpu().numpy()
            feats["emb_last"][i] = e[-1].half().cpu().numpy()
        for L, idx in sorted(by_len.items()):
            for s in range(0, len(idx), BATCH):
                chunk = idx[s:s + BATCH]
                inp = torch.tensor([ids[i] for i in chunk], device=dev)
                model(input_ids=inp, use_cache=False)
                for k, h in hooks.items():
                    if k not in feats:
                        feats[k] = np.zeros((W, h.last.shape[1]), np.float16)
                    feats[k][chunk] = h.last.numpy()
            print(f"word-alone len {L}: {len(idx)} words, {time.time() - t0:.0f}s", flush=True)
    for h in handles:
        h.remove()

    # word alone as the whole user message, read at the scan's site
    mu = load_mu(TAG)
    site = hook_site(model, int(entry["hook_block"]))
    reader = Reader(stop=True)
    handle = site.register_forward_hook(reader)
    chats = [render_chat(tok, w, date) for w in words]
    assert_boundary(tok, chats[0], boundary_ids, TAG)
    cids = [tok(c, add_special_tokens=False)["input_ids"] for c in chats]
    if any(c[-2:] != boundary_ids for c in cids):
        raise SystemExit("FATAL: a word-alone chat does not end on the boundary ids")
    chat_proj = np.zeros(W, np.float64)
    chat_states = np.zeros((W, 2, mu.shape[1]), np.float16)
    by_len = defaultdict(list)
    for i, c in enumerate(cids):
        by_len[len(c)].append(i)
    with torch.inference_mode():
        for L, idx in sorted(by_len.items()):
            for s in range(0, len(idx), BATCH // 2):
                chunk = idx[s:s + BATCH // 2]
                forward(model, reader, input_ids=torch.tensor([cids[i] for i in chunk], device=dev), use_cache=False)
                st = reader.last.cpu().numpy()
                chat_states[chunk] = st.astype(np.float16)
                chat_proj[chunk] = projection_of(st, mu)
    handle.remove()
    print(f"chat frame done {time.time() - t0:.0f}s", flush=True)

    np.savez(OUT / "features.npz", chat_proj=chat_proj, chat_states=chat_states, **feats)
    atomic_json(OUT / "meta.json", {
        "experiment": "x109", "stage": "wordrep_extract", "model": entry["model_id"], "revision": entry["revision"],
        "n_words": W, "blocks": BLOCKS, "attention_layers": attn_layers, "read_block": int(entry["hook_block"]),
        "word_alone_input": "ids(' ' + w), no template, no special tokens; last token",
        "chat_frame": "x106_common.render_chat(tokenizer, w, date); block-29 output at the two boundary tokens; projection_of on load_mu('qwen')",
        "features": {k: list(v.shape) for k, v in feats.items()}, "seconds": time.time() - t0,
    })
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
