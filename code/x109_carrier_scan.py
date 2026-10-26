"""x109 short-request scan, GPU stage: every scanned word appended after a short benign request.

The original scan appends w to a long source-clean benign prompt. Here the prompt is a short,
complete, harmless request (a ladder from 1 to about 45 tokens), and w is appended after it
exactly as in the scan: user message = request + " " + w, rendered with x106_common.render_chat
(same template, date and read site), boundary projection at the block-29 output on the two
boundary tokens; shift[r, w] = proj(request + " " + w) - proj(request).
Each request alone is also answered greedily (256 new tokens) so its reply can be read.
Usage: x109_carrier_scan.py SHARD NSHARDS; writes data/runs/x109/carrier/shifts_part{SHARD}.npz
and meta_part{SHARD}.json.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
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

OUT = RUN / "carrier"
BATCH = 1024
# Short complete requests of increasing length; the last three are the opening of three of the
# x109 benign prompts (no_robots:4896, dolly:2055, no_robots:1040).
REQUESTS = [
    "Hi", "Hello!", "Thanks.", "Say hello.", "Name a color.", "What is 2+2?", "Count to five.",
    "Tell me a joke.", "List three fruits.", "What is the capital of France?", "Write a haiku about spring.",
    "Summarize in a few sentences how to use mirrors and curtains to make a living space seem larger.",
    "Earth is the third planet from the Sun and the only place known in the universe where life has "
    "originated and found habitability.",
    "Please create a detailed outline for me for a chapter. In this chapter, Kathy and Ronald move into "
    "their new house. Kathy hires a moving company to help her move in, and she unpacks and shops for "
    "things she needs for the house over the next week.",
]


def read_all(model, reader, tok, texts, boundary_ids, mu, t0, label):
    cids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    bad = sum(c[-2:] != boundary_ids for c in cids)
    if bad:
        raise SystemExit(f"FATAL: {bad} chats in {label} do not end on the boundary ids")
    proj = np.zeros(len(texts), np.float64)
    by_len = defaultdict(list)
    for i, c in enumerate(cids):
        by_len[len(c)].append(i)
    dev = model.device
    with torch.inference_mode():
        for L, idx in sorted(by_len.items()):
            for s in range(0, len(idx), BATCH):
                chunk = idx[s:s + BATCH]
                forward(model, reader, input_ids=torch.tensor([cids[i] for i in chunk], device=dev), use_cache=False)
                proj[chunk] = projection_of(reader.last.cpu().numpy(), mu)
    print(f"{label}: {len(texts)} chats, {time.time() - t0:.0f}s", flush=True)
    return proj, cids


def main() -> None:
    shard, nshards = int(sys.argv[1]), int(sys.argv[2])
    mine = [r for i, r in enumerate(REQUESTS) if i % nshards == shard]
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    entry = read_json(inputs_path())["models"][TAG]
    tok = load_tag_tokenizer(entry["snapshot"], TAG)
    date = read_json(projections_path(TAG))["rendering"]["date"]
    boundary_ids = [int(v) for v in entry["boundary_ids"]]
    words = read_json(SCAN_JSON)["candidates"]["words"]
    W = len(words)
    model = load_tag_model(entry["snapshot"], TAG).eval()
    mu = load_mu(TAG)

    replies = {}
    with torch.inference_mode():
        for r in mine:
            ids = tok(render_chat(tok, r, date), add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            out = model.generate(input_ids=ids, max_new_tokens=256, do_sample=False)
            replies[r] = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    print(f"request replies done {time.time() - t0:.0f}s", flush=True)

    site = hook_site(model, int(entry["hook_block"]))
    reader = Reader(stop=True)
    handle = site.register_forward_hook(reader)
    assert_boundary(tok, render_chat(tok, mine[0], date), boundary_ids, TAG)
    base, _ = read_all(model, reader, tok, [render_chat(tok, r, date) for r in mine], boundary_ids, mu, t0, "requests")
    word_ids = [tok(" " + w, add_special_tokens=False)["input_ids"] for w in words]
    shift = np.zeros((len(mine), W), np.float32)
    junction_ok, n_tokens = {}, {}
    for k, r in enumerate(mine):
        chat_r = tok(render_chat(tok, r, date), add_special_tokens=False)["input_ids"]
        user_r = tok(r, add_special_tokens=False)["input_ids"]
        n_tokens[r] = len(user_r)
        n_pre = next(i for i in range(len(chat_r)) if chat_r[i:i + len(user_r)] == user_r) + len(user_r)
        pre, suf = chat_r[:n_pre], chat_r[n_pre:]
        p, cids = read_all(model, reader, tok, [render_chat(tok, f"{r} {w}", date) for w in words], boundary_ids, mu, t0, r[:30])
        shift[k] = p - base[k]
        junction_ok[r] = int(sum(cids[i] == pre + word_ids[i] + suf for i in range(W)))
    handle.remove()

    np.savez(OUT / f"shifts_part{shard}.npz", requests=np.array(mine), words=np.array(words), base=base, shift=shift)
    atomic_json(OUT / f"meta_part{shard}.json", {
        "experiment": "x109", "stage": "short_request_scan", "model": entry["model_id"], "revision": entry["revision"],
        "n_words": W, "requests": mine, "read_block": int(entry["hook_block"]), "date": date,
        "user_message": "request + ' ' + w; shift = proj - proj(request alone)",
        "request_n_tokens": n_tokens, "request_alone_projection": dict(zip(mine, base.tolist())),
        "junction_equal_to_request_word_suffix": junction_ok,
        "request_alone_greedy_reply": replies, "seconds": time.time() - t0,
    })
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
