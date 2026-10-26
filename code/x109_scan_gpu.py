"""Vocabulary scan, projection pass (GPU, Qwen3.5-4B).

Candidates: every vocabulary token that decodes to ' ' + [a-z]{3,} and round-trips as one
token, union the lowercase 3+ letter words of the wordfreq top-50,000 list, union the x83
lexicon, the x85 neutrals and the fixed twin-noun lists. For each of the 40 prompts the
rendered chat P + text is prefilled once; its cache (attention keys/values, DeltaNet conv and
recurrent states) is repeated to the batch and ' ' + w + S is fed per candidate, grouped by
continuation length; block 29's output at the last two positions is projected on the x87
consensus direction (x106 reader). shift = proj(text + ' ' + w) - proj(text), with proj(text)
read through the same cached path (empty continuation). Resumable per prompt.
"""

from __future__ import annotations

import importlib.metadata
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x109_scan_bench_gpu import PLACEHOLDER, Reader, expand_cache, forward  # noqa: E402
from x109_scan_common import (  # noqa: E402
    PILOT,
    PILOT_TWIN_CANDIDATES,
    PROMPTS,
    SCAN_JSON,
    SCAN_NPZ,
    SCAN_PARTS,
    TAG,
    TWIN_CANDIDATES,
    WORDFREQ,
    X106_HARD1K_SD,
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
from x83_common import lexicon_words  # noqa: E402
from x85_common import WORD_PAIRS  # noqa: E402

WORD_RE = re.compile(r"[a-z]{3,}")
SPACE_WORD_RE = re.compile(r" [a-z]{3,}")
BATCH = 1024
JUNCTION_SAMPLE = 300


def spearman(x, y) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    t_start = time.time()
    entry = read_json(inputs_path())["models"][TAG]
    if sha256_file(directions_path(TAG)) != entry["directions"]["sha256"]:
        raise SystemExit("FATAL: direction artifact differs from the x106 inputs")
    tokenizer = load_tag_tokenizer(entry["snapshot"], TAG)
    stage0a = read_json(projections_path(TAG))
    date = stage0a["rendering"]["date"]
    boundary_ids = [int(v) for v in entry["boundary_ids"]]
    prompts = read_json(PROMPTS)
    plist = prompts["prompts"]

    def ids(s: str) -> list[int]:
        return tokenizer(s, add_special_tokens=False)["input_ids"]

    # ---- candidates -------------------------------------------------------------------------
    special = set(tokenizer.all_special_ids) | set(getattr(tokenizer, "added_tokens_decoder", {}).keys())
    single = {}
    for i in range(len(tokenizer)):
        if i in special:
            continue
        s = tokenizer.decode([i])
        if SPACE_WORD_RE.fullmatch(s):
            single[s[1:]] = i
    single_rt = {w for w, i in single.items() if ids(" " + w) == [i]}
    wf_alpha = [w for w in read_json(WORDFREQ)["words"] if WORD_RE.fullmatch(w)]
    base = single_rt | set(wf_alpha)
    lex = [w for w, _ in lexicon_words()]
    extras_req = list(dict.fromkeys(lex + [n for _, _, n in WORD_PAIRS] + list(PILOT_TWIN_CANDIDATES) + list(TWIN_CANDIDATES)))
    extras_added = [w for w in extras_req if w not in base]
    words = sorted(base | set(extras_req))
    cont = {w: ids(" " + w) for w in words}
    n_tok = [len(cont[w]) for w in words]
    print(f"candidates {len(words)} (base {len(base)}, extras added {extras_added})", flush=True)

    # ---- template split and junction check ----------------------------------------------------
    rendered = render_chat(tokenizer, PLACEHOLDER, date)
    P, S = rendered.split(PLACEHOLDER)
    suffix_ids = ids(S)
    if suffix_ids[-2:] != boundary_ids:
        raise SystemExit("FATAL: template suffix does not end with the boundary ids")
    rng = random.Random(20261009)
    pilot_words = read_json(PILOT / "projections.json")["words"]
    junction = {"n_checked": 0, "n_fail": 0, "fail_examples": []}
    for p in plist:
        pre = ids(P + p["text"])
        chat_o = render_chat(tokenizer, p["text"], date)
        assert_boundary(tokenizer, chat_o, boundary_ids, TAG)
        if ids(chat_o) != pre + suffix_ids:
            raise SystemExit(f"FATAL: O junction fails on {p['id']}")
        for w in list(dict.fromkeys(pilot_words + extras_req + rng.sample(words, JUNCTION_SAMPLE))):
            chat = render_chat(tokenizer, append_end(p["text"], w), date)
            junction["n_checked"] += 1
            if chat != P + p["text"] + " " + w + S or ids(chat) != pre + cont[w] + suffix_ids:
                junction["n_fail"] += 1
                junction["fail_examples"].append([p["id"], w])
    junction["rule"] = "ids(render(text + ' ' + w)) == ids(P + text) + ids(' ' + w) + ids(S), checked per prompt on the pilot words, the extras and 300 random candidates"
    print("junction", junction["n_checked"], junction["n_fail"], flush=True)
    if junction["n_fail"]:
        raise SystemExit("FATAL: junction check failed")

    # ---- model, audit, scale ------------------------------------------------------------------
    model = load_tag_model(entry["snapshot"], TAG)
    model.eval()
    device = model.device
    block = int(entry["hook_block"])
    site = hook_site(model, block)
    mu = load_mu(TAG)
    units, norms = unit_directions(mu)
    units_gpu = torch.from_numpy(units.astype(np.float32)).to(device)

    def render(text: str) -> str:
        chat = render_chat(tokenizer, text, date)
        assert_boundary(tokenizer, chat, boundary_ids, TAG)
        return chat

    audit = audit_site(model, tokenizer, site, block, torch.from_numpy(mu[0].astype(np.float32)), torch.from_numpy(mu[1].astype(np.float32)), [render(p["text"]) for p in plist[:2]])
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
    words_sha = sha256_text("\n".join(words))
    SCAN_PARTS.mkdir(parents=True, exist_ok=True)
    batch = BATCH
    t0 = time.time()
    timing = []
    for pi, p in enumerate(plist):
        part = SCAN_PARTS / f"{pi:02d}.npz"
        if part.exists():
            z = np.load(part)
            if str(z["words_sha"]) == words_sha and str(z["prompt_sha"]) == p["text_sha256"]:
                continue
        tp = time.time()
        pre = ids(P + p["text"])
        cache = prefix_cache(pre)
        o_cached = cached_batch(cache, [[]])[0]
        o_full = full_o(pre)
        e = np.empty((len(words), 2), dtype=np.float32)
        for _, js in sorted(groups.items()):
            k = 0
            while k < len(js):
                chunk = js[k : k + batch]
                try:
                    e[chunk] = cached_batch(cache, [cont[words[j]] for j in chunk])
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    batch = max(64, batch // 4)
                    print(f"OOM -> batch {batch}", flush=True)
                    continue
                k += len(chunk)
        np.savez(part, e=e, o_cached=o_cached, o_full=o_full, words_sha=words_sha, prompt_sha=p["text_sha256"], prefix_tokens=len(pre), seconds=time.time() - tp, batch=batch)
        timing.append(time.time() - tp)
        print(f"prompt {pi + 1}/{len(plist)} {p['id']} {time.time() - tp:.0f}s total {(time.time() - t0) / 60:.1f} min peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GB", flush=True)
        del cache
    handle.remove()

    # ---- assemble -------------------------------------------------------------------------
    parts = [np.load(SCAN_PARTS / f"{pi:02d}.npz") for pi in range(len(plist))]
    e_tok = np.stack([z["e"] for z in parts])  # [P, N, 2]
    o_cached = np.stack([z["o_cached"] for z in parts])
    o_full = np.stack([z["o_full"] for z in parts])
    shift = (e_tok - o_cached[:, None, :]).astype(np.float32)
    np.savez_compressed(SCAN_NPZ, shift=shift, o_cached=o_cached, o_full=o_full)

    # Comparison with the pilot's full-prefill batch-1 shifts on set A.
    pilot = read_json(PILOT / "projections.json")
    widx = {w: j for j, w in enumerate(words)}
    pw = [w for w in pilot["words"] if w in widx]
    pshift = np.asarray(pilot["shift"])[:, [pilot["words"].index(w) for w in pw]]
    rows_a = [i for i, p in enumerate(plist) if p["id"] in pilot["prompt_ids"]]
    order = [pilot["prompt_ids"].index(plist[i]["id"]) for i in rows_a]
    sshift = shift[rows_a][:, [widx[w] for w in pw]].sum(axis=2)
    pshift = pshift[order]
    vs_pilot = {
        "n_prompts": len(rows_a),
        "n_words": len(pw),
        "max_abs_diff_summed_shift": float(np.abs(sshift - pshift).max()),
        "mean_abs_diff_summed_shift": float(np.abs(sshift - pshift).mean()),
        "spearman_per_cell": spearman(sshift.ravel(), pshift.ravel()),
        "spearman_word_means": spearman(sshift.mean(0), pshift.mean(0)),
        "o_full_vs_pilot_o_max_abs": float(np.abs(o_full[rows_a].sum(1) - np.asarray(pilot["o_proj"])[order]).max()),
    }
    print("vs pilot", vs_pilot, flush=True)
    out = {
        "run": "x109",
        "model_tag": TAG,
        "model_id": entry["model_id"],
        "model_source": entry["snapshot"],
        "prompts_sha256": prompts["prompts_sha256"],
        "prompt_ids": [p["id"] for p in plist],
        "prompt_sets": [p["set"] for p in plist],
        "site": {"hook_block": block, "boundary_ids": boundary_ids, "read": "block output at the last two prefill positions (x106 Stage 0a), cached prefix, forward stopped after the read block"},
        "directions": {**file_record(directions_path(TAG)), "norms": [float(v) for v in norms]},
        "rendering": {"renderer": "x106_common.render_chat", "date": date, "template_prefix": P, "template_suffix": S, "suffix_ids": suffix_ids},
        "site_audit": audit,
        "scale_check_vs_x106_stage0a": scale,
        "junction": junction,
        "sd_unit": {"value": X106_HARD1K_SD, "source": "population sd of the x106 Stage 0a projections over the 1,319 Hard-1K prompts, Qwen3.5-4B"},
        "projection": "per boundary token <h, mu/|mu|>; shift = proj(text + ' ' + w) - proj(text), both through the cached path; npz 'shift' is [prompt, candidate, token] float32 with token order (tbg_minus1, tbg)",
        "candidates": {
            "rule": "vocabulary tokens decoding to ' ' + [a-z]{3,} that round-trip as one token, union lowercase [a-z]{3,} words of wordfreq top-50,000, union x83 lexicon, x85 neutrals and the twin-noun lists; sorted",
            "n": len(words),
            "n_base": len(base),
            "n_single_token_roundtrip": len(single_rt),
            "n_wordfreq_alpha": len(wf_alpha),
            "extras_added": extras_added,
            "lexicon_all_in_base": all(w in base for w in lex),
            "token_length_hist": dict(sorted(Counter(n_tok).items())),
            "words_sha256": words_sha,
            "words": words,
            "n_tokens": n_tok,
        },
        "o_proj_cached": o_cached.sum(1).tolist(),
        "o_proj_full": o_full.sum(1).tolist(),
        "o_cached_vs_full_max_abs_per_token": float(np.abs(o_cached - o_full).max()),
        "vs_pilot_projections": vs_pilot,
        "batch_size": BATCH,
        "batch_size_final": batch,
        "seconds_per_prompt": [float(z["seconds"]) for z in parts],
        "prefix_tokens": [int(z["prefix_tokens"]) for z in parts],
        "scan_seconds_this_job": time.time() - t0,
        "job_seconds": time.time() - t_start,
        "environment": {"torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "device": torch.cuda.get_device_name(0)},
    }
    atomic_json(SCAN_JSON, out)
    print("->", SCAN_JSON, SCAN_NPZ)


if __name__ == "__main__":
    main()
