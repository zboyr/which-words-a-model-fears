"""Vocabulary-scan benchmark (GPU, Qwen3.5-4B): candidate counts, KV-cache validity, throughput.

1. Candidate counts under the Qwen3.5-4B tokenizer: vocabulary tokens that decode to
   ' ' + [a-z]{3,}, and the token length of ' ' + w for the wordfreq top-50,000 list.
2. Cached-prefix validity: the rendered chat of (text + ' ' + w) as token ids equals
   ids(P + text) + ids(' ' + w) + ids(S) for the template prefix P and suffix S; the boundary
   projections from a cached prefix (prefill P + text once, then feed ' ' + w + S) are compared
   with the batch-1 full prefill and with the pilot's projections.json.
3. Throughput on 2 prompts x 2,000 candidates: full prefill and cached prefix at several batch
   sizes. Rows are bucketed by continuation length, so no batch is padded. Every forward stops
   after block 29 (the read site); the cached variant copies the batch-1 prefix cache to the
   batch (attention keys/values and DeltaNet conv/recurrent states) before each forward.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x109_pilot_common import PROJECTIONS, PROMPTS, RUN, TAG, append_end, atomic_json, read_json  # noqa: E402
from x106_common import (  # noqa: E402
    assert_boundary,
    hook_site,
    inputs_path,
    load_mu,
    load_tag_model,
    load_tag_tokenizer,
    projections_path,
    render_chat,
    unit_directions,
)

BENCH = RUN / "bench"
WORDLIST = BENCH / "wordfreq_en_top50000.json"
WORD_RE = re.compile(r"[a-z]{3,}")
SPACE_WORD_RE = re.compile(r" [a-z]{3,}")
PLACEHOLDER = "QXZPLACEHOLDERQXZ"
VALID_WORDS = (
    "pedophile", "racist", "porn", "bomb", "garden", "portrait",
    "explosive", "anthrax", "murder", "genocide", "nazi", "phishing", "rape", "nude", "molest", "prostitution",
    "harmful", "illegal", "bypass", "unethical", "dangerous",
    "lamp", "pineapple", "refrigerator", "chrysanthemum", "constellation", "melody", "compass", "village", "lemonade",
)


class Stop(Exception):
    pass


class Reader:
    """Block-output hook: keep the last two positions (float32) and stop the forward."""

    def __init__(self, stop: bool) -> None:
        self.stop = stop
        self.last = None

    def __call__(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self.last = hidden[:, -2:, :].float()
        if self.stop:
            raise Stop
        return output


def forward(model, reader: Reader, **kwargs):
    reader.last = None
    try:
        out = model(**kwargs)
    except Stop:
        out = None
    if reader.last is None:
        raise SystemExit("FATAL: reader did not fire")
    return out


def expand_cache(cache, batch: int):
    """A batch-`batch` copy of a batch-1 cache: every key/value and conv/recurrent state repeated."""
    c = copy.copy(cache)
    c.layers = []
    for layer in cache.layers:
        new = copy.copy(layer)
        for name in ("keys", "values"):
            t = getattr(layer, name, None)
            if isinstance(t, torch.Tensor) and t.numel():
                setattr(new, name, t.repeat(batch, *([1] * (t.dim() - 1))))
        for name in ("conv_states", "recurrent_states"):
            d = getattr(layer, name, None)
            if isinstance(d, dict):
                setattr(new, name, {k: (v.repeat(batch, *([1] * (v.dim() - 1))) if isinstance(v, torch.Tensor) else v) for k, v in d.items()})
        for name in ("has_previous_state", "is_conv_states_initialized", "is_recurrent_states_initialized", "conv_kernel_size"):
            d = getattr(layer, name, None)
            if isinstance(d, dict):
                setattr(new, name, dict(d))
        c.layers.append(new)
    return c


def cache_summary(cache) -> list[dict]:
    out = []
    for layer in cache.layers:
        rec = {"type": type(layer).__name__}
        for name in ("keys", "values"):
            t = getattr(layer, name, None)
            if isinstance(t, torch.Tensor):
                rec[name] = [list(t.shape), str(t.dtype)]
        for name in ("conv_states", "recurrent_states"):
            d = getattr(layer, name, None)
            if isinstance(d, dict):
                rec[name] = {str(k): [list(v.shape), str(v.dtype)] for k, v in d.items() if isinstance(v, torch.Tensor)}
        out.append(rec)
    return out


def cache_bytes(cache) -> int:
    total = 0
    for layer in cache.layers:
        for name in ("keys", "values"):
            t = getattr(layer, name, None)
            if isinstance(t, torch.Tensor):
                total += t.numel() * t.element_size()
        for name in ("conv_states", "recurrent_states"):
            d = getattr(layer, name, None)
            if isinstance(d, dict):
                total += sum(v.numel() * v.element_size() for v in d.values() if isinstance(v, torch.Tensor))
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-words", type=int, default=2000)
    parser.add_argument("--full-bs", default="1,16,64,256")
    parser.add_argument("--cached-bs", default="64,256,1024,2048")
    parser.add_argument("--bs1-words", type=int, default=200)
    args = parser.parse_args()

    entry = read_json(inputs_path())["models"][TAG]
    tokenizer = load_tag_tokenizer(entry["snapshot"], TAG)
    stage0a = read_json(projections_path(TAG))
    date = stage0a["rendering"]["date"]
    boundary_ids = [int(v) for v in entry["boundary_ids"]]
    prompts = read_json(PROMPTS)["prompts"]
    pilot = read_json(PROJECTIONS)
    report: dict = {"environment": {"torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "device": torch.cuda.get_device_name(0)}}

    def ids(s: str) -> list[int]:
        return tokenizer(s, add_special_tokens=False)["input_ids"]

    # ---- 1. candidate counts ------------------------------------------------------------
    t0 = time.time()
    special = set(tokenizer.all_special_ids) | set(getattr(tokenizer, "added_tokens_decoder", {}).keys())
    single = {}
    for i in range(len(tokenizer)):
        if i in special:
            continue
        s = tokenizer.decode([i])
        if SPACE_WORD_RE.fullmatch(s):
            single[s[1:]] = i
    single_roundtrip = {w for w, i in single.items() if ids(" " + w) == [i]}
    wf = read_json(WORDLIST)["words"]
    wf_alpha = [w for w in wf if WORD_RE.fullmatch(w)]
    lens = {w: len(ids(" " + w)) for w in wf_alpha}
    multi = [w for w in wf_alpha if lens[w] > 1]
    union = sorted(single_roundtrip | set(wf_alpha))
    report["candidates"] = {
        "vocab_size": len(tokenizer),
        "single_token_space_lower_ge3": len(single),
        "single_token_roundtrip": len(single_roundtrip),
        "wordfreq_top50000": len(wf),
        "wordfreq_alpha_lower_ge3": len(wf_alpha),
        "wordfreq_alpha_single_token": len(wf_alpha) - len(multi),
        "wordfreq_alpha_multi_token": len(multi),
        "wordfreq_multi_token_length_hist": dict(sorted(Counter(lens[w] for w in multi).items())),
        "wordfreq_alpha_in_single_set": len(set(wf_alpha) & single_roundtrip),
        "union_single_roundtrip_or_wordfreq_alpha": len(union),
        "union_token_length_hist": dict(sorted(Counter(1 if w in single_roundtrip else lens[w] for w in union).items())),
        "seconds": time.time() - t0,
    }
    print(json.dumps(report["candidates"]), flush=True)

    # ---- template split and junction check --------------------------------------------------
    rendered = render_chat(tokenizer, PLACEHOLDER, date)
    P, S = rendered.split(PLACEHOLDER)
    suffix_ids = ids(S)
    rng = random.Random(20261008)
    bench_words = rng.sample(union, args.n_words)
    junction_fail = []
    n_checked = 0
    for p in prompts:
        pre = ids(P + p["text"])
        o_ids = ids(render_chat(tokenizer, p["text"], date))
        if o_ids != pre + suffix_ids:
            junction_fail.append({"prompt": p["id"], "word": None})
        for w in list(dict.fromkeys(pilot["words"] + list(VALID_WORDS) + (bench_words if p is prompts[0] or p is prompts[1] else []))):
            chat = render_chat(tokenizer, append_end(p["text"], w), date)
            if chat != P + p["text"] + " " + w + S:
                junction_fail.append({"prompt": p["id"], "word": w, "kind": "string"})
                continue
            n_checked += 1
            if ids(chat) != pre + ids(" " + w) + suffix_ids:
                junction_fail.append({"prompt": p["id"], "word": w, "kind": "ids"})
    report["junction"] = {
        "template_prefix": P,
        "template_suffix": S,
        "suffix_ids": suffix_ids,
        "boundary_ids": boundary_ids,
        "suffix_ends_with_boundary": suffix_ids[-2:] == boundary_ids,
        "n_checked": n_checked,
        "n_fail": len(junction_fail),
        "fail_examples": junction_fail[:20],
        "rule": "ids(render(text + ' ' + w)) == ids(P + text) + ids(' ' + w) + ids(S)",
    }
    print(json.dumps({k: v for k, v in report["junction"].items() if k != "template_prefix"}), flush=True)

    # ---- model ------------------------------------------------------------------------------
    model = load_tag_model(entry["snapshot"], TAG)
    model.eval()
    device = model.device
    site = hook_site(model, int(entry["hook_block"]))
    mu = load_mu(TAG)
    units, _ = unit_directions(mu)
    units_gpu = torch.from_numpy(units.astype(np.float32)).to(device)

    def proj_gpu(states: torch.Tensor) -> np.ndarray:
        return torch.einsum("btd,td->bt", states, units_gpu).cpu().numpy().astype(np.float64)

    def proj_cpu(states: torch.Tensor) -> np.ndarray:
        return np.einsum("btd,td->bt", states.cpu().numpy().astype(np.float64), units)

    full_reader, stop_reader = Reader(stop=False), Reader(stop=True)

    @torch.inference_mode()
    def full_batch(prefix: list[int], conts: list[list[int]], reader: Reader, cpu=False) -> np.ndarray:
        x = torch.tensor([prefix + c + suffix_ids for c in conts], device=device)
        forward(model, reader, input_ids=x, use_cache=False)
        return proj_cpu(reader.last) if cpu else proj_gpu(reader.last)

    @torch.inference_mode()
    def prefix_cache(prefix: list[int]):
        x = torch.tensor([prefix], device=device)
        stop, stop_reader.stop = stop_reader.stop, False
        try:
            out = model(input_ids=x, use_cache=True)
        finally:
            stop_reader.stop = stop
        return out.past_key_values

    @torch.inference_mode()
    def cached_batch(cache, conts: list[list[int]], reader: Reader, cpu=False) -> np.ndarray:
        c = expand_cache(cache, len(conts))
        x = torch.tensor([cn + suffix_ids for cn in conts], device=device)
        forward(model, reader, input_ids=x, past_key_values=c, use_cache=True)
        return proj_cpu(reader.last) if cpu else proj_gpu(reader.last)

    # ---- 2. validity on prompt 0 x 30 words -------------------------------------------------
    handle = site.register_forward_hook(full_reader)
    p0 = prompts[0]
    pre0 = ids(P + p0["text"])
    pi = pilot["prompt_ids"].index(p0["id"])
    o_tok = np.asarray(pilot["o_proj_by_token"][pi])
    pilot_e = {w: o_tok + np.asarray(pilot["shift_by_token"][pi][j]) for j, w in enumerate(pilot["words"])}
    words30 = list(VALID_WORDS)
    full1 = {w: full_batch(pre0, [ids(" " + w)], full_reader, cpu=True)[0] for w in words30}
    o_here = full_batch(pre0, [[]], full_reader, cpu=True)[0]
    cache0 = prefix_cache(pre0)
    report["cache_layout"] = {"class": type(cache0).__name__, "layers": cache_summary(cache0), "bytes_batch1": cache_bytes(cache0), "prefix_tokens": len(pre0)}
    cached1 = {w: cached_batch(cache0, [ids(" " + w)], full_reader, cpu=True)[0] for w in words30}
    # Batched cached: all 30 continuations, bucketed by length.
    cachedB = {}
    by_len: dict[int, list[str]] = {}
    for w in words30:
        by_len.setdefault(len(ids(" " + w)), []).append(w)
    for _, ws in by_len.items():
        vals = cached_batch(cache0, [ids(" " + w) for w in ws], full_reader, cpu=True)
        cachedB.update(dict(zip(ws, vals)))
    fullB = {}
    for _, ws in by_len.items():
        vals = full_batch(pre0, [ids(" " + w) for w in ws], full_reader, cpu=True)
        fullB.update(dict(zip(ws, vals)))
    handle.remove()

    def maxdiff(a: dict, b: dict) -> float:
        return float(max(np.abs(a[w] - b[w]).max() for w in words30))

    per_word = {w: {"n_tokens": len(ids(" " + w)), "pilot": pilot_e[w].tolist(), "full_bs1": full1[w].tolist(), "cached_bs1": cached1[w].tolist(), "cached_batched": cachedB[w].tolist(), "full_batched": fullB[w].tolist()} for w in words30}
    report["validity"] = {
        "prompt": p0["id"],
        "words": words30,
        "o_here_vs_pilot_max_abs": float(np.abs(o_here - o_tok).max()),
        "max_abs_per_token": {
            "full_bs1_vs_pilot": maxdiff(full1, pilot_e),
            "cached_bs1_vs_full_bs1": maxdiff(cached1, full1),
            "cached_bs1_vs_pilot": maxdiff(cached1, pilot_e),
            "cached_batched_vs_full_bs1": maxdiff(cachedB, full1),
            "cached_batched_vs_pilot": maxdiff(cachedB, pilot_e),
            "full_batched_vs_full_bs1": maxdiff(fullB, full1),
        },
        "shift_scale": {"mean_abs_pilot_shift_tbg_minus1": float(np.mean([abs(pilot_e[w][0] - o_tok[0]) for w in words30])), "mean_abs_pilot_shift_tbg": float(np.mean([abs(pilot_e[w][1] - o_tok[1]) for w in words30]))},
        "per_word": per_word,
    }
    print(json.dumps(report["validity"]["max_abs_per_token"]), flush=True)

    # ---- 3. throughput -------------------------------------------------------------------
    handle = site.register_forward_hook(stop_reader)
    bench_prompts = prompts[:2]
    bench_len = {w: ids(" " + w) for w in bench_words}
    report["bench_words"] = {"n": len(bench_words), "token_length_hist": dict(sorted(Counter(len(v) for v in bench_len.values()).items()))}
    groups: dict[int, list[str]] = {}
    for w in bench_words:
        groups.setdefault(len(bench_len[w]), []).append(w)

    def run(variant: str, bs: int, words_limit: int | None = None) -> dict:
        ws_all = bench_words[:words_limit] if words_limit else bench_words
        grp: dict[int, list[str]] = {}
        for w in ws_all:
            grp.setdefault(len(bench_len[w]), []).append(w)
        results = {}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        # warm-up: one batch of the largest group
        first = max(grp.values(), key=len)[:bs]
        pre = ids(P + bench_prompts[0]["text"])
        if variant == "full":
            full_batch(pre, [bench_len[w] for w in first], stop_reader)
        else:
            cached_batch(prefix_cache(pre), [bench_len[w] for w in first], stop_reader)
        torch.cuda.synchronize()
        t = time.time()
        n = 0
        for p in bench_prompts:
            pre = ids(P + p["text"])
            cache = prefix_cache(pre) if variant == "cached" else None
            for _, ws in grp.items():
                for k in range(0, len(ws), bs):
                    chunk = ws[k : k + bs]
                    conts = [bench_len[w] for w in chunk]
                    vals = full_batch(pre, conts, stop_reader) if variant == "full" else cached_batch(cache, conts, stop_reader)
                    for w, v in zip(chunk, vals):
                        results[(p["id"], w)] = v
                    n += len(chunk)
        torch.cuda.synchronize()
        sec = time.time() - t
        return {"variant": variant, "batch_size": bs, "n_candidates": n, "seconds": sec, "candidates_per_second": n / sec, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "_results": results}

    runs = []
    reference = None
    for variant, sizes in (("full", [int(v) for v in args.full_bs.split(",")]), ("cached", [int(v) for v in args.cached_bs.split(",")])):
        for bs in sizes:
            try:
                r = run(variant, bs, args.bs1_words if bs == 1 else None)
            except torch.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                runs.append({"variant": variant, "batch_size": bs, "status": "OOM", "error": str(exc)[:200]})
                print(f"{variant} bs={bs} OOM", flush=True)
                continue
            res = r.pop("_results")
            if reference is None and variant == "full" and bs > 1:
                reference = res
            if reference is not None:
                common = [k for k in res if k in reference]
                r["max_abs_vs_full_reference"] = float(max(np.abs(res[k] - reference[k]).max() for k in common)) if common else None
            r["status"] = "ok"
            runs.append(r)
            print(json.dumps(r), flush=True)
            torch.cuda.empty_cache()
    handle.remove()
    report["throughput"] = {"prompts": [p["id"] for p in bench_prompts], "prefix_tokens": [len(ids(P + p["text"])) for p in bench_prompts], "suffix_tokens": len(suffix_ids), "runs": runs, "stop_after_block": int(entry["hook_block"])}
    device_tag = "b200" if "B200" in report["environment"]["device"] else "rtx"
    out = BENCH / f"bench_{device_tag}.json"
    atomic_json(out, report)
    print("->", out)


if __name__ == "__main__":
    main()
