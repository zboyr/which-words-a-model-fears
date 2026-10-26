"""x110 vocabulary scan on x100's 192 prompts (GPU, Qwen3.5-4B; sharded by prompt).

The x109 cached-prefix reader over the x109 candidate list (71,583 words, taken from the x109
scan.json and re-tokenised here): for each prompt the rendered chat prefix P + text is prefilled
once, its cache is repeated to the batch and ' ' + w + S is fed per candidate; block 29's output
at the last two positions is projected on the x87 consensus direction. shift = proj(text + ' '
+ w) - proj(text) through the same cached path. `--shard k --n-shards n` scans prompts
k::n, resumable per prompt; `--assemble` stacks the parts into scan.json + scan_shift.npz
(the prompts that have a part; `prompt_indices_scanned` lists them).
`--words key` scans only the ten chosen words and their twins (same reader, same checks) on
every prompt into key_parts/ -> key_words_scan.json + key_words_shift.npz.
`--words key --prompts topup` scans the same twenty words on the HP top-up prompts
(prompts_topup.json) into key_parts_topup/ -> key_words_scan_topup.json + key_words_shift_topup.npz.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x109_scan_bench_gpu import PLACEHOLDER, Reader, expand_cache, forward  # noqa: E402
from x110_common import (  # noqa: E402
    KEY_JSON,
    KEY_JSON_TOPUP,
    KEY_NPZ,
    KEY_NPZ_TOPUP,
    KEY_PARTS,
    KEY_PARTS_TOPUP,
    PROMPTS,
    PROMPTS_TOPUP,
    SCAN_JSON,
    SCAN_NPZ,
    SCAN_PARTS,
    TAG,
    WORDS,
    X106_HARD1K_SD,
    X109_SCAN_JSON,
    append_end,
    atomic_json,
    read_json,
    sha256_text,
)
from x103_extract import audit_site  # noqa: E402
from x106_common import (  # noqa: E402
    BoundaryReader,
    assert_boundary,
    directions_path,
    file_record,
    hard1k_rows,
    hook_site,
    inputs_path,
    load_mu,
    load_tag_model,
    load_tag_tokenizer,
    projection_of,
    projections_path,
    render_chat,
    sha256_file,
    unit_directions,
)

BATCH = 1024
JUNCTION_SAMPLE = 100


def part_path(parts_dir: Path, pi: int) -> Path:
    return parts_dir / f"{pi:03d}.npz"


def assemble(parts_dir: Path, out_json: Path, out_npz: Path, cand_meta: dict, words: list[str], n_tok: list[int], words_sha: str, plist: list[dict], meta: dict) -> None:
    parts, scanned = [], []
    for pi, p in enumerate(plist):
        if not part_path(parts_dir, pi).exists():
            continue
        z = np.load(part_path(parts_dir, pi))
        if str(z["words_sha"]) != words_sha or str(z["prompt_sha"]) != p["text_sha256"]:
            raise SystemExit(f"FATAL: part {pi} belongs to another word list or prompt")
        parts.append(z)
        scanned.append(pi)
    if not parts:
        raise SystemExit(f"FATAL: no parts in {parts_dir}")
    e_tok = np.stack([z["e"] for z in parts])  # [P, N, 2]
    o_cached = np.stack([z["o_cached"] for z in parts])
    o_full = np.stack([z["o_full"] for z in parts])
    shift = (e_tok - o_cached[:, None, :]).astype(np.float32)
    np.savez_compressed(out_npz, shift=shift, o_cached=o_cached, o_full=o_full)
    out = {
        **meta,
        "n_prompts_scanned": len(scanned),
        "prompt_indices_scanned": scanned,
        "prompt_ids_scanned": [plist[pi]["id"] for pi in scanned],
        "candidates": {**cand_meta, "n": len(words), "words_sha256": words_sha, "words": words, "n_tokens": n_tok},
        "o_proj_cached": o_cached.sum(1).tolist(),
        "o_proj_full": o_full.sum(1).tolist(),
        "o_cached_vs_full_max_abs_per_token": float(np.abs(o_cached - o_full).max()),
        "seconds_per_prompt": [float(z["seconds"]) for z in parts],
        "prefix_tokens": [int(z["prefix_tokens"]) for z in parts],
        "batch_size_final": [int(z["batch"]) for z in parts],
    }
    atomic_json(out_json, out)
    print("->", out_json, out_npz, shift.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--words", choices=("all", "key"), default="all", help="all: the x109 candidate list; key: the chosen words and twins only")
    parser.add_argument("--prompts", choices=("original", "topup"), default="original", help="key scan only: prompts.json, or the HP top-up prompts of prompts_topup.json (-> key_words_*_topup)")
    args = parser.parse_args()
    if args.prompts == "topup" and args.words != "key":
        raise SystemExit("--prompts topup scans the key words only")
    t_start = time.time()
    entry = read_json(inputs_path())["models"][TAG]
    if sha256_file(directions_path(TAG)) != entry["directions"]["sha256"]:
        raise SystemExit("FATAL: direction artifact differs from the x106 inputs")
    tokenizer = load_tag_tokenizer(entry["snapshot"], TAG)
    stage0a = read_json(projections_path(TAG))
    date = stage0a["rendering"]["date"]
    boundary_ids = [int(v) for v in entry["boundary_ids"]]
    prompts = read_json(PROMPTS_TOPUP if args.prompts == "topup" else PROMPTS)
    plist = prompts["prompts"]
    chosen = read_json(WORDS)["chosen"]
    key_words = [c["word"] for c in chosen] + [c["twin"] for c in chosen]

    def ids(s: str) -> list[int]:
        return tokenizer(s, add_special_tokens=False)["input_ids"]

    if args.words == "all":
        x109 = read_json(X109_SCAN_JSON)["candidates"]
        words = list(x109["words"])
        words_sha = sha256_text("\n".join(words))
        if words_sha != x109["words_sha256"]:
            raise SystemExit("FATAL: x109 candidate list fails its sha256")
        cont = {w: ids(" " + w) for w in words}
        n_tok = [len(cont[w]) for w in words]
        if n_tok != list(x109["n_tokens"]):
            raise SystemExit("FATAL: candidate token counts differ from x109's")
        if any(w not in cont for w in key_words):
            raise SystemExit("FATAL: a chosen word or twin is not a candidate")
        parts_dir, out_json, out_npz = SCAN_PARTS, SCAN_JSON, SCAN_NPZ
        cand_meta = {"source": "x109 scan.json candidate list, re-tokenised and checked here"}
    else:
        words = list(dict.fromkeys(key_words))
        if len(words) != len(key_words):
            raise SystemExit("FATAL: chosen words and twins are not distinct")
        words_sha = sha256_text("\n".join(words))
        cont = {w: ids(" " + w) for w in words}
        n_tok = [len(cont[w]) for w in words]
        for c in chosen:
            if n_tok[words.index(c["word"])] != int(c["n_tokens"]):
                raise SystemExit(f"FATAL: token count of {c['word']} differs from words.json")
        parts_dir, out_json, out_npz = (KEY_PARTS_TOPUP, KEY_JSON_TOPUP, KEY_NPZ_TOPUP) if args.prompts == "topup" else (KEY_PARTS, KEY_JSON, KEY_NPZ)
        cand_meta = {"source": "the ten chosen words then their twins, from words.json"}
    rendered = render_chat(tokenizer, PLACEHOLDER, date)
    P, S = rendered.split(PLACEHOLDER)
    suffix_ids = ids(S)
    if suffix_ids[-2:] != boundary_ids:
        raise SystemExit("FATAL: template suffix does not end with the boundary ids")
    meta = {
        "run": "x110",
        "model_tag": TAG,
        "model_id": entry["model_id"],
        "model_source": entry["snapshot"],
        "prompts_sha256": prompts["prompts_sha256"],
        "prompt_ids": [p["id"] for p in plist],
        "prompt_populations": [p["population"] for p in plist],
        "site": {"hook_block": int(entry["hook_block"]), "boundary_ids": boundary_ids, "read": "block output at the last two prefill positions (x106 Stage 0a), cached prefix, forward stopped after the read block"},
        "directions": {**file_record(directions_path(TAG)), "norms": [float(v) for v in unit_directions(load_mu(TAG))[1]]},
        "rendering": {"renderer": "x106_common.render_chat", "date": date, "template_prefix": P, "template_suffix": S, "suffix_ids": suffix_ids},
        "sd_unit": {"value": X106_HARD1K_SD, "source": "population sd of the x106 Stage 0a projections over the 1,319 Hard-1K prompts, Qwen3.5-4B"},
        "projection": "per boundary token <h, mu/|mu|>; shift = proj(text + ' ' + w) - proj(text), both through the cached path; npz 'shift' is [prompt, candidate, token] float32 with token order (tbg_minus1, tbg)",
        "environment": {"torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
    }
    meta["word_list"] = args.words
    meta["prompt_set"] = args.prompts
    if args.assemble:
        assemble(parts_dir, out_json, out_npz, cand_meta, words, n_tok, words_sha, plist, meta)
        return

    mine = list(range(len(plist)))[args.shard :: args.n_shards]
    todo = []
    for pi in mine:
        part = part_path(parts_dir, pi)
        if part.exists():
            z = np.load(part)
            if str(z["words_sha"]) == words_sha and str(z["prompt_sha"]) == plist[pi]["text_sha256"]:
                continue
        todo.append(pi)
    print(f"shard {args.shard}/{args.n_shards}: {len(mine)} prompts, {len(todo)} to do", flush=True)
    if not todo:
        return

    # Junction check on this shard's prompts: the chosen words, the twins and 100 random candidates.
    rng = random.Random(20261009 + args.shard)
    junction = {"n_checked": 0, "n_fail": 0, "fail_examples": []}
    for pi in todo:
        p = plist[pi]
        pre = ids(P + p["text"])
        chat_o = render_chat(tokenizer, p["text"], date)
        assert_boundary(tokenizer, chat_o, boundary_ids, TAG)
        if ids(chat_o) != pre + suffix_ids:
            raise SystemExit(f"FATAL: O junction fails on {p['id']}")
        for w in list(dict.fromkeys(key_words + rng.sample(words, min(JUNCTION_SAMPLE, len(words))))):
            chat = render_chat(tokenizer, append_end(p["text"], w), date)
            junction["n_checked"] += 1
            if chat != P + p["text"] + " " + w + S or ids(chat) != pre + cont[w] + suffix_ids:
                junction["n_fail"] += 1
                junction["fail_examples"].append([p["id"], w])
    print("junction", junction["n_checked"], junction["n_fail"], flush=True)
    if junction["n_fail"]:
        raise SystemExit(f"FATAL: junction check failed: {junction['fail_examples'][:5]}")

    model = load_tag_model(entry["snapshot"], TAG)
    model.eval()
    device = model.device
    block = int(entry["hook_block"])
    site = hook_site(model, block)
    mu = load_mu(TAG)
    units, _ = unit_directions(mu)
    units_gpu = torch.from_numpy(units.astype(np.float32)).to(device)

    def render(text: str) -> str:
        chat = render_chat(tokenizer, text, date)
        assert_boundary(tokenizer, chat, boundary_ids, TAG)
        return chat

    audit = audit_site(model, tokenizer, site, block, torch.from_numpy(mu[0].astype(np.float32)), torch.from_numpy(mu[1].astype(np.float32)), [render(plist[pi]["text"]) for pi in todo[:2]])
    print("site audit", audit["status"], flush=True)
    if audit["status"] != "pass":
        raise SystemExit("FATAL: read-site audit failed")
    breader = BoundaryReader()
    handle = site.register_forward_hook(breader)
    s0 = {r["id"]: r["proj"] for r in stage0a["rows"]}
    scale = []
    with torch.inference_mode():
        for row in hard1k_rows(TAG)[:2]:
            chat = render(row["prompt"])
            if sha256_text(chat) != row["rendered_chat_sha256"]:
                raise SystemExit("FATAL: Hard-1K rendering does not reproduce the base row")
            enc = tokenizer(chat, return_tensors="pt", add_special_tokens=False).to(device)
            model(**enc, use_cache=False)
            value = float(projection_of(breader.last[0].numpy()[None], mu)[0])
            scale.append({"id": row["id"], "proj_here": value, "proj_x106_stage0a": s0[row["id"]], "abs_diff": abs(value - s0[row["id"]])})
    handle.remove()
    print("scale", scale, flush=True)
    if max(s["abs_diff"] for s in scale) > 0.5:
        raise SystemExit("FATAL: scale check against x106 Stage 0a failed")

    reader = Reader(stop=True)
    handle = site.register_forward_hook(reader)

    @torch.inference_mode()
    def prefix_cache(prefix: list[int]):
        reader.stop = False
        try:
            out = model(input_ids=torch.tensor([prefix], device=device), use_cache=True)
        finally:
            reader.stop = True
        return out.past_key_values

    @torch.inference_mode()
    def cached_batch(cache, conts: list[list[int]]) -> np.ndarray:
        c = expand_cache(cache, len(conts))
        x = torch.tensor([cn + suffix_ids for cn in conts], device=device)
        forward(model, reader, input_ids=x, past_key_values=c, use_cache=True)
        return torch.einsum("btd,td->bt", reader.last, units_gpu).cpu().numpy()

    @torch.inference_mode()
    def full_o(prefix: list[int]) -> np.ndarray:
        forward(model, reader, input_ids=torch.tensor([prefix + suffix_ids], device=device), use_cache=False)
        return torch.einsum("btd,td->bt", reader.last, units_gpu).cpu().numpy()[0]

    groups: dict[int, list[int]] = {}
    for j, w in enumerate(words):
        groups.setdefault(n_tok[j], []).append(j)
    parts_dir.mkdir(parents=True, exist_ok=True)
    batch = BATCH
    t0 = time.time()
    for k, pi in enumerate(todo):
        p = plist[pi]
        tp = time.time()
        pre = ids(P + p["text"])
        cache = prefix_cache(pre)
        o_cached = cached_batch(cache, [[]])[0]
        o_full = full_o(pre)
        e = np.empty((len(words), 2), dtype=np.float32)
        for _, js in sorted(groups.items()):
            start = 0
            while start < len(js):
                chunk = js[start : start + batch]
                try:
                    e[chunk] = cached_batch(cache, [cont[words[j]] for j in chunk])
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    batch = max(64, batch // 4)
                    print(f"OOM -> batch {batch}", flush=True)
                    continue
                start += len(chunk)
        np.savez(part_path(parts_dir, pi), e=e, o_cached=o_cached, o_full=o_full, words_sha=words_sha, prompt_sha=p["text_sha256"], prefix_tokens=len(pre), seconds=time.time() - tp, batch=batch, junction_checked=junction["n_checked"], site_audit=audit["status"], scale_max_abs_diff=max(s["abs_diff"] for s in scale))
        print(f"prompt {k + 1}/{len(todo)} [{pi}] {p['id']} {p['population']} {len(pre)} tok {time.time() - tp:.0f}s total {(time.time() - t0) / 60:.1f} min peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GB", flush=True)
        del cache
    handle.remove()
    print(f"shard {args.shard} done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
