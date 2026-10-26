"""x112 boundary-shift scan (GPU, one model per job).

`--words all`: the x109 candidate list (71,583 words) on the eight scan prompts -> the model's
own table. `--words key`: the model's 40 chosen words on its 32 held-out prompts, run after
words and prompts are fixed (the in-context shift). For each prompt the rendered chat P + text is
prefilled once; its cache is repeated to the batch (every per-layer tensor state: attention
keys / values, DeltaNet conv / recurrent states; only the layers up to the read block) and
' ' + w + S is fed per word, grouped by continuation length so no batch is padded; the read
block's output at the last two positions is projected on the model's x106 direction.
shift = proj(text + ' ' + w) - proj(text), proj(text) read through the same path.
`--mode full` replaces the cached prefix by a full prefill of P + text + ' ' + w + S per word.
Audit on every prompt (cached mode): the O projection cached vs full prefill, and the shift of
AUDIT_N sampled words cached vs full prefill (batched by length, and batch 1 on a few).
Junction: every word's continuation ids are read off the tokenized P + text + ' ' + w + S, so a
tokenizer that merges across the junction still gets the exact rendered ids (such words are
counted and run through full prefill). Resumable per prompt; `--assemble` writes the outputs.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x109_scan_bench_gpu import PLACEHOLDER, VALID_WORDS, Reader, forward  # noqa: E402
from x112_common import CANDIDATES, ROOT, SCAN_PROMPTS, X111_TABLES_NPZ, append_end, atomic_json, paths, read_json, sd_unit, sha256_text  # noqa: E402
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
AUDIT_N = 128
AUDIT_BS1 = 8
RENDER_SAMPLE = 200


def expand_cache(cache, batch: int, n_layers: int):
    """A batch-`batch` copy of a batch-1 cache: every tensor state of the first n_layers layers
    repeated along the batch dimension (tensors directly on the layer or inside a dict / list)."""

    def rep(v):
        if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == 1 and v.numel():
            return v.repeat(batch, *([1] * (v.dim() - 1)))
        if isinstance(v, dict):
            return {k: rep(x) for k, x in v.items()}
        if isinstance(v, list):
            return [rep(x) for x in v]
        return v

    c = copy.copy(cache)
    c.layers = []
    for i, layer in enumerate(cache.layers):
        new = copy.copy(layer)
        if i < n_layers:
            for name, v in list(vars(layer).items()):
                setattr(new, name, rep(v))
        else:
            for name, v in list(vars(layer).items()):
                if isinstance(v, (dict, list)):
                    setattr(new, name, copy.copy(v))
        c.layers.append(new)
    return c


def cache_summary(cache) -> list:
    out = []
    for layer in cache.layers:
        rec = {"type": type(layer).__name__}
        for name, v in vars(layer).items():
            if isinstance(v, torch.Tensor):
                rec[name] = list(v.shape)
            elif isinstance(v, dict):
                rec[name] = {str(k): list(x.shape) for k, x in v.items() if isinstance(x, torch.Tensor)} or {str(k): repr(x)[:40] for k, x in v.items()}
            elif isinstance(v, (int, float, bool, str)) or v is None:
                rec[name] = v
        out.append(rec)
    return out


def spearman(x, y) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--words", choices=("all", "key"), default="all")
    parser.add_argument("--prompts", choices=("original", "topup"), default="original", help="key scan: the 32 prompts of prompts.json, or the top-up prompts of prompts_topup.json (-> key_words_*_topup)")
    parser.add_argument("--mode", choices=("cached", "full"), default="cached")
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--prompt-limit", type=int, default=0, help="debug: scan only the first n prompts")
    parser.add_argument("--word-limit", type=int, default=0, help="debug: only the first n words")
    args = parser.parse_args()
    tag = args.tag
    P_ = paths(tag)
    t_start = time.time()
    entry = read_json(inputs_path())["models"][tag]
    if sha256_file(directions_path(tag)) != entry["directions"]["sha256"]:
        raise SystemExit("FATAL: direction artifact differs from the x106 inputs")
    sd = sd_unit(tag)
    tokenizer = load_tag_tokenizer(entry["snapshot"], tag)
    stage0a = read_json(projections_path(tag))
    date = stage0a["rendering"]["date"]
    boundary_ids = [int(v) for v in entry["boundary_ids"]]
    if args.words == "all":
        prompts = read_json(SCAN_PROMPTS)
        plist = [{**p, "population": "BP"} for p in prompts["prompts"]]
        cand = read_json(CANDIDATES)
        words = cand["words"]
        if sha256_text("\n".join(words)) != cand["words_sha256"]:
            raise SystemExit("FATAL: candidate list fails its sha256")
        parts_dir, out_json, out_npz = P_["scan_parts"], P_["scan_json"], P_["scan_npz"]
    elif args.prompts == "topup":
        prompts = read_json(P_["prompts_topup"])
        plist = prompts["prompts"]
        words = [c["word"] for c in read_json(P_["words"])["chosen"]]
        parts_dir, out_json, out_npz = P_["key_parts_topup"], P_["key_json_topup"], P_["key_npz_topup"]
    else:
        prompts = read_json(P_["prompts"])
        plist = prompts["prompts"]
        words = [c["word"] for c in read_json(P_["words"])["chosen"]]
        parts_dir, out_json, out_npz = P_["key_parts"], P_["key_json"], P_["key_npz"]
    if args.prompt_limit:
        plist = plist[: args.prompt_limit]
    if args.word_limit:
        words = words[: args.word_limit]
    words_sha = sha256_text("\n".join(words))

    def ids(s: str) -> list[int]:
        return tokenizer(s, add_special_tokens=False)["input_ids"]

    rendered = render_chat(tokenizer, PLACEHOLDER, date)
    P, S = rendered.split(PLACEHOLDER)
    suffix_ids = ids(S)
    if suffix_ids[-2:] != boundary_ids:
        raise SystemExit("FATAL: template suffix does not end with the boundary ids")
    n_tok_plain = [len(ids(" " + w)) for w in words]
    meta = {
        "run": "x112",
        "model_tag": tag,
        "model_id": entry["model_id"],
        "model_source": entry["snapshot"],
        "word_list": args.words,
        "prompt_set": "scan prompts" if args.words == "all" else args.prompts,
        "mode": args.mode,
        "prompts_sha256": prompts["prompts_sha256"],
        "prompt_ids": [p["id"] for p in plist],
        "prompt_populations": [p["population"] for p in plist],
        "site": {"hook_block": int(entry["hook_block"]), "boundary_ids": boundary_ids, "read": "block output at the last two prefill positions (x106 Stage 0a), forward stopped after the read block"},
        "directions": {**file_record(directions_path(tag)), "norms": [float(v) for v in unit_directions(load_mu(tag))[1]]},
        "rendering": {"renderer": "x106_common.render_chat", "date": date, "template_prefix": P, "template_suffix": S, "suffix_ids": suffix_ids},
        "sd_unit": {"value": sd, "source": "population sd (ddof 0) of the model's x106 Stage 0a projections over the 1,319 Hard-1K prompts"},
        "projection": "per boundary token <h, mu/|mu|>; shift = proj(text + ' ' + w) - proj(text) through the same path; npz 'shift' is [prompt, word, token] float32 in raw units with token order (tbg_minus1, tbg)",
    }
    if args.assemble:
        assemble(parts_dir, out_json, out_npz, plist, words, words_sha, n_tok_plain, meta, tag)
        return

    parts_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for pi, p in enumerate(plist):
        part = parts_dir / f"{pi:03d}.npz"
        if part.exists():
            z = np.load(part)
            if str(z["words_sha"]) == words_sha and str(z["prompt_sha"]) == p["text_sha256"] and str(z["mode"]) == args.mode:
                continue
        todo.append(pi)
    print(f"{tag} {args.words} {args.mode}: {len(plist)} prompts, {len(todo)} to do, {len(words)} words", flush=True)
    if not todo:
        return

    model = load_tag_model(entry["snapshot"], tag)
    model.eval()
    device = model.device
    block = int(entry["hook_block"])
    site = hook_site(model, block)
    mu = load_mu(tag)
    units, _ = unit_directions(mu)
    units_gpu = torch.from_numpy(units.astype(np.float32)).to(device)

    def render(text: str) -> str:
        chat = render_chat(tokenizer, text, date)
        assert_boundary(tokenizer, chat, boundary_ids, tag)
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
        for row in hard1k_rows(tag)[:2]:
            chat = render(row["prompt"])
            if sha256_text(chat) != row["rendered_chat_sha256"]:
                raise SystemExit("FATAL: Hard-1K rendering does not reproduce the base row")
            enc = tokenizer(chat, return_tensors="pt", add_special_tokens=False).to(device)
            model(**enc, use_cache=False)
            value = float(projection_of(breader.last[0].numpy()[None], mu)[0])
            scale.append({"id": row["id"], "proj_here": value, "proj_x106_stage0a": s0[row["id"]], "abs_diff_sd": abs(value - s0[row["id"]]) / sd})
    handle.remove()
    print("scale", scale, flush=True)
    if max(s["abs_diff_sd"] for s in scale) > 0.1:
        raise SystemExit("FATAL: scale check against x106 Stage 0a failed")

    reader = Reader(stop=True)
    handle = site.register_forward_hook(reader)
    n_expand = block + 1
    layout = {}

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
        c = expand_cache(cache, len(conts), n_expand)
        x = torch.tensor([cn + suffix_ids for cn in conts], device=device)
        forward(model, reader, input_ids=x, past_key_values=c, use_cache=True)
        return torch.einsum("btd,td->bt", reader.last, units_gpu).float().cpu().numpy()

    @torch.inference_mode()
    def full_batch(prefix: list[int], conts: list[list[int]]) -> np.ndarray:
        x = torch.tensor([prefix + cn + suffix_ids for cn in conts], device=device)
        forward(model, reader, input_ids=x, use_cache=False)
        return torch.einsum("btd,td->bt", reader.last, units_gpu).float().cpu().numpy()

    def run_grouped(fn, conts: dict[int, list[int]], batch: int) -> tuple[dict[int, np.ndarray], int]:
        groups: dict[int, list[int]] = {}
        for j, c in conts.items():
            groups.setdefault(len(c), []).append(j)
        res = {}
        for _, js in sorted(groups.items()):
            k = 0
            while k < len(js):
                chunk = js[k : k + batch]
                try:
                    vals = fn([conts[j] for j in chunk])
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if batch <= 8:
                        raise
                    batch = max(8, batch // 4)
                    print(f"OOM -> batch {batch}", flush=True)
                    continue
                for j, v in zip(chunk, vals):
                    res[j] = v
                k += len(chunk)
        return res, batch

    batch = args.batch
    rng = random.Random(112)
    t0 = time.time()
    for k, pi in enumerate(todo):
        p = plist[pi]
        tp = time.time()
        if render(p["text"]) != P + p["text"] + S:
            raise SystemExit(f"FATAL: template is not a plain substitution on {p['id']}")
        for w in rng.sample(words, min(RENDER_SAMPLE, len(words))):
            if render(append_end(p["text"], w)) != P + p["text"] + " " + w + S:
                raise SystemExit(f"FATAL: template is not a plain substitution on {p['id']} + {w}")
        pre = ids(P + p["text"])
        if ids(P + p["text"] + S) != pre + suffix_ids:
            raise SystemExit(f"FATAL: O junction fails on {p['id']}")
        # Exact ids of every rendered E text, in tokenizer batches.
        conts: dict[int, list[int]] = {}
        odd: dict[int, list[int]] = {}
        texts = [P + p["text"] + " " + w + S for w in words]
        for s0_ in range(0, len(texts), 4096):
            enc = tokenizer(texts[s0_ : s0_ + 4096], add_special_tokens=False)["input_ids"]
            for j, full in enumerate(enc, start=s0_):
                if full[: len(pre)] == pre and full[-len(suffix_ids) :] == suffix_ids and len(full) > len(pre) + len(suffix_ids):
                    conts[j] = full[len(pre) : len(full) - len(suffix_ids)]
                else:
                    odd[j] = full
        n_cont_differs = sum(1 for j, c in conts.items() if len(c) != n_tok_plain[j])
        e = np.full((len(words), 2), np.nan, dtype=np.float32)
        audit_rec = {}
        if args.mode == "cached":
            cache = prefix_cache(pre)
            if not layout:
                layout = {"class": type(cache).__name__, "layers": cache_summary(cache), "n_layers_expanded": n_expand}
            o_cached = cached_batch(cache, [[]])[0]
            o_full = full_batch(pre, [[]])[0]
            res, batch = run_grouped(lambda cs: cached_batch(cache, cs), conts, batch)
            for j, v in res.items():
                e[j] = v
            # Audit: sampled words through full prefill.
            arng = np.random.default_rng([112, pi])
            pool = sorted(conts)
            sample = sorted(arng.choice(pool, size=min(AUDIT_N, len(pool)), replace=False).tolist())
            fres, _ = run_grouped(lambda cs: full_batch(pre, cs), {j: conts[j] for j in sample}, 256)
            f_e = np.stack([fres[j] for j in sample])
            c_e = e[sample]
            bs1 = {j: full_batch(pre, [conts[j]])[0] for j in sample[:AUDIT_BS1]}
            sh_c = (c_e - o_cached).sum(1)
            sh_f = (f_e - o_full).sum(1)
            audit_rec = {
                "n_words": len(sample),
                "o_cached_vs_full_max_abs_per_token_sd": float(np.abs(o_cached - o_full).max() / sd),
                "e_cached_vs_full_batched_max_abs_per_token_sd": float(np.abs(c_e - f_e).max() / sd),
                "e_cached_vs_full_batched_mean_abs_per_token_sd": float(np.abs(c_e - f_e).mean() / sd),
                "e_full_batched_vs_bs1_max_abs_per_token_sd": float(max(np.abs(fres[j] - v).max() for j, v in bs1.items()) / sd),
                "e_cached_vs_full_bs1_max_abs_per_token_sd": float(max(np.abs(e[j] - v).max() for j, v in bs1.items()) / sd),
                "summed_shift_sd_of_sample_full": float(sh_f.std() / sd),
                "summed_shift_spearman_cached_vs_full": spearman(sh_c, sh_f),
                "summed_shift_pearson_cached_vs_full": float(np.corrcoef(sh_c, sh_f)[0, 1]),
                "summed_shift_max_abs_diff_sd": float(np.abs(sh_c - sh_f).max() / sd),
            }
            del cache
        else:
            o_full = full_batch(pre, [[]])[0]
            o_cached = o_full.copy()
            res, batch = run_grouped(lambda cs: full_batch(pre, cs), conts, min(batch, 256))
            for j, v in res.items():
                e[j] = v
        if odd:
            ores, _ = run_grouped(lambda cs: full_batch([], cs), odd, 256)
            for j, v in ores.items():
                e[j] = v
        if np.isnan(e).any():
            raise SystemExit("FATAL: unread words")
        np.savez(parts_dir / f"{pi:03d}.npz", e=e, o_cached=o_cached, o_full=o_full, words_sha=words_sha, prompt_sha=p["text_sha256"], mode=args.mode, prefix_tokens=len(pre), seconds=time.time() - tp, batch=batch, n_junction_odd=len(odd), n_cont_len_differs=n_cont_differs, audit=np.asarray(str(audit_rec)), scale_max_abs_diff_sd=max(s["abs_diff_sd"] for s in scale), site_audit=audit["status"], layout=np.asarray(str(layout)))
        print(f"prompt {k + 1}/{len(todo)} [{pi}] {p['id']} {len(pre)} tok {time.time() - tp:.0f}s odd {len(odd)} batch {batch} audit {audit_rec} total {(time.time() - t0) / 60:.1f} min peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GB", flush=True)
    handle.remove()
    if layout:
        atomic_json(parts_dir / "cache_layout.json", layout)
    print(f"{tag} done in {(time.time() - t_start) / 60:.1f} min", flush=True)
    if not args.prompt_limit and not args.word_limit:
        assemble(parts_dir, out_json, out_npz, plist, words, words_sha, n_tok_plain, meta, tag)


def assemble(parts_dir: Path, out_json: Path, out_npz: Path, plist, words, words_sha, n_tok, meta, tag) -> None:
    import ast

    parts, scanned = [], []
    for pi, p in enumerate(plist):
        f = parts_dir / f"{pi:03d}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        if str(z["words_sha"]) != words_sha or str(z["prompt_sha"]) != p["text_sha256"]:
            raise SystemExit(f"FATAL: part {pi} belongs to another word list or prompt")
        parts.append(z)
        scanned.append(pi)
    if len(parts) != len(plist):
        raise SystemExit(f"FATAL: {len(parts)} of {len(plist)} prompts scanned")
    e_tok = np.stack([z["e"] for z in parts])
    o_cached = np.stack([z["o_cached"] for z in parts])
    o_full = np.stack([z["o_full"] for z in parts])
    shift = (e_tok - o_cached[:, None, :]).astype(np.float32)
    np.savez_compressed(out_npz, shift=shift, o_cached=o_cached, o_full=o_full)
    sd = meta["sd_unit"]["value"]
    audits = [ast.literal_eval(str(z["audit"])) for z in parts]
    modes = sorted({str(z["mode"]) for z in parts})
    out = {
        **meta,
        "mode": modes[0] if len(modes) == 1 else modes,
        "n_prompts_scanned": len(scanned),
        "candidates": {"source": "x112 candidates.json (x109 list)" if meta["word_list"] == "all" else "the 40 chosen words, from words.json", "n": len(words), "words_sha256": words_sha, "words": words, "n_tokens": n_tok},
        "o_proj_cached_sd": (o_cached.sum(1) / sd).tolist(),
        "o_proj_full_sd": (o_full.sum(1) / sd).tolist(),
        "o_cached_vs_full_max_abs_per_token": float(np.abs(o_cached - o_full).max()),
        "o_cached_vs_full_max_abs_per_token_sd": float(np.abs(o_cached - o_full).max() / sd),
        "audit_per_prompt": audits,
        "n_junction_odd_per_prompt": [int(z["n_junction_odd"]) for z in parts],
        "n_cont_len_differs_per_prompt": [int(z["n_cont_len_differs"]) for z in parts],
        "scale_max_abs_diff_sd": float(max(float(z["scale_max_abs_diff_sd"]) for z in parts)),
        "site_audit": sorted({str(z["site_audit"]) for z in parts}),
        "seconds_per_prompt": [float(z["seconds"]) for z in parts],
        "prefix_tokens": [int(z["prefix_tokens"]) for z in parts],
        "batch_size_final": [int(z["batch"]) for z in parts],
        "cache_layout": read_json(parts_dir / "cache_layout.json") if (parts_dir / "cache_layout.json").exists() else None,
    }
    atomic_json(out_json, out)
    print("->", out_json, out_npz, shift.shape)
    if meta["word_list"] == "all":
        write_table(tag, shift, words, words_sha, n_tok, sd, out_json)


def write_table(tag: str, shift: np.ndarray, words, words_sha, n_tok, sd, scan_json: Path) -> None:
    q = np.load(X111_TABLES_NPZ)
    eligible = q["eligible"]
    own = shift.astype(np.float64).sum(axis=2).mean(0) / sd
    pct = np.full(len(words), np.nan)
    idx = np.where(eligible)[0]
    order = idx[np.argsort(own[idx], kind="stable")]
    pct[order] = 100.0 * np.arange(len(order)) / len(order)
    P_ = paths(tag)
    np.savez_compressed(P_["table_npz"], own=own.astype(np.float32), pct_own=pct, eligible=eligible, n_tokens=np.asarray(n_tok, np.int16), per_prompt_sd=(shift.astype(np.float64).sum(axis=2) / sd).astype(np.float32))
    e = own[eligible]
    out = {
        "run": "x112",
        "model_tag": tag,
        "scan": str(scan_json.relative_to(ROOT)),
        "candidates": {"n": len(words), "words_sha256": words_sha},
        "statistic": f"per word: mean over the eight scan prompts of the shift summed over the two boundary tokens, divided by the model's x106 Hard-1K sd ({sd:.6f}); percentile = 100 * ascending rank / n over x111's eligible words (0 = most negative)",
        "n_eligible": int(eligible.sum()),
        "distribution_sd_eligible": {k: float(np.quantile(e, v)) for k, v in (("p1", 0.01), ("p5", 0.05), ("p25", 0.25), ("median", 0.5), ("p75", 0.75), ("p95", 0.95), ("p99", 0.99))},
        "npz_arrays": "own (sd units, candidate order), pct_own (nan for ineligible), eligible, n_tokens (this model's tokenizer, ' ' + w), per_prompt_sd [8, N]",
    }
    atomic_json(P_["table_json"], out)
    print("->", P_["table_json"], out["distribution_sd_eligible"])


if __name__ == "__main__":
    main()
