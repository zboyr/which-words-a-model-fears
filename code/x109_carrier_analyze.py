"""x109 short-request scan, analysis: how short can the benign prompt be and still rank the vocabulary like y?

Rankings: y (mean shift over the 40 benign prompts / 8.645, wordrep/oof.npz), the word-alone
chat projection (carrier/chat_proj.npy, copied from wordrep), each short request's shift
(carrier/shifts_part*.npz), each single benign prompt's shift (scan_shift.npz), and means of the
per-request z-scored shifts over pairs and triples of the short requests.
Metrics of a ranking r against a reference t: Spearman over all words, Spearman within t's top
1,000, AUC for t's top 1 %, and how many of t's top 10/50/100/716 fall in r's top 100/500/1000/2000.
Also the sd over words of each shift (x106 sd units), the rank of ten violence/crime verbs, and
cross-request agreement. Writes carrier/results.json, carrier/results.md, carrier/shifts.npz.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x109_scan_common import RUN, SCAN_NPZ, atomic_json  # noqa: E402

OUT = RUN / "carrier"
SD = 8.645
TOPS = [10, 50, 100, 716]
CUTS = [100, 500, 1000, 2000]
VERBS = ["bomb", "bombs", "gun", "kill", "steal", "exploit", "crack", "hack", "assassinate", "shoot"]
COLS = ["n_tok", "sd", "spearman", "spearman_ref_top1000", "auc_ref_top1pct", "t10_in_r100", "t10_in_r500",
        "t50_in_r500", "t50_in_r2000", "t100_in_r1000", "t100_in_r2000", "t716_in_r1000", "t716_in_r2000"]


def ranks_desc(x):
    r = np.empty(len(x), np.int64)
    r[np.argsort(-x, kind="stable")] = np.arange(1, len(x) + 1)
    return r


def auc(score, pos):
    rk = stats.rankdata(score)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((rk[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def compare(r, t):
    top_t = np.argsort(-t)[:1000]
    pos = t >= np.quantile(t, 0.99)
    order_t, order_r = np.argsort(-t), np.argsort(-r)
    hits = {f"t{a}_in_r{b}": int(np.isin(order_t[:a], order_r[:b]).sum()) for a in TOPS for b in CUTS}
    return {"sd": float(r.std()), "spearman": float(stats.spearmanr(r, t).statistic),
            "spearman_ref_top1000": float(stats.spearmanr(r[top_t], t[top_t]).statistic),
            "auc_ref_top1pct": auc(r, pos), **hits}


def fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def table(title, block, cols=COLS):
    out = ["", f"## {title}", "", "| ranking | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for n, m in block.items():
        out.append(f"| {n} | " + " | ".join(fmt(m.get(c, "")) for c in cols) + " |")
    return out


def main() -> None:
    parts = sorted(OUT.glob("shifts_part*.npz"))
    reqs, shifts, base, meta = [], [], [], {"request_n_tokens": {}, "request_alone_greedy_reply": {},
                                            "junction_equal_to_request_word_suffix": {}}
    words = None
    for p in parts:
        d = np.load(p)
        words = [str(w) for w in d["words"]] if words is None else words
        assert [str(w) for w in d["words"]] == words
        reqs += [str(r) for r in d["requests"]]
        shifts.append(d["shift"])
        base += d["base"].tolist()
        m = json.loads((OUT / p.name.replace("shifts_", "meta_").replace(".npz", ".json")).read_text())
        for k in meta:
            meta[k].update(m[k])
    shift = np.concatenate(shifts) / SD
    order = np.argsort([meta["request_n_tokens"][r] for r in reqs], kind="stable")
    reqs = [reqs[i] for i in order]
    shift, base = shift[order], [base[i] for i in order]
    np.savez(OUT / "shifts.npz", requests=np.array(reqs), words=np.array(words), base=np.array(base),
             shift_sd_units=shift.astype(np.float32))

    oof = np.load(RUN / "wordrep" / "oof.npz", allow_pickle=True)
    assert [str(w) for w in oof["words"]] == words
    y = oof["y"].astype(np.float64)
    chat = np.load(OUT / "chat_proj.npy").astype(np.float64)
    single = np.load(SCAN_NPZ)["shift"].sum(-1) / SD
    assert np.allclose(single.mean(0), y, atol=1e-3)
    ntok = meta["request_n_tokens"]
    W = len(words)

    rank = {r: shift[k].astype(np.float64) for k, r in enumerate(reqs)}
    res = {"requests": reqs, "n_tokens": ntok, "request_alone_projection": dict(zip(reqs, base)),
           "junction_ok": meta["junction_equal_to_request_word_suffix"], "n_words": W,
           "request_alone_greedy_reply": meta["request_alone_greedy_reply"],
           "vs_y": {}, "vs_word_alone": {}, "single_prompt_vs_y": [], "combos_vs_y": {}, "verb_ranks": {}}
    for r, v in rank.items():
        res["vs_y"][r] = {"n_tok": ntok[r], **compare(v, y)}
        res["vs_word_alone"][r] = {"n_tok": ntok[r], **compare(v, chat)}
    res["vs_y"]["word_alone (chat_proj)"] = compare(chat, y)
    for p in range(single.shape[0]):
        res["single_prompt_vs_y"].append(compare(single[p], y))
    # leave-one-out version: the prompt against the mean of the other 39
    res["single_prompt_vs_y_loo"] = [compare(single[p], (single.sum(0) - single[p]) / 39) for p in range(single.shape[0])]

    z = (shift - shift.mean(1, keepdims=True)) / shift.std(1, keepdims=True)
    short = [k for k, r in enumerate(reqs) if ntok[r] <= 8]
    for size in (2, 3):
        best = None
        for c in itertools.combinations(short, size):
            m = compare(z[list(c)].mean(0), y)
            if best is None or m["spearman"] > best[1]["spearman"]:
                best = (c, m)
        res["combos_vs_y"][f"best {size} of requests <=8 tok: " + " + ".join(reqs[i] for i in best[0])] = best[1]
    res["combos_vs_y"][f"mean of all {len(short)} requests <=8 tok"] = compare(z[short].mean(0), y)
    res["combos_vs_y"][f"mean of all {len(reqs)} requests"] = compare(z.mean(0), y)

    idx = {w: i for i, w in enumerate(words)}
    for n, v in {"y": y, "word_alone": chat, **rank}.items():
        rk = ranks_desc(v)
        res["verb_ranks"][n] = {w: int(rk[idx[w]]) for w in VERBS if w in idx}
    cs = np.array([[stats.spearmanr(rank[a], rank[b]).statistic for b in reqs] for a in reqs])
    tops = {a: set(np.argsort(-rank[a])[:100]) for a in reqs}
    co = np.array([[len(tops[a] & tops[b]) for b in reqs] for a in reqs])
    res["cross_request_spearman"], res["cross_request_top100_overlap"] = cs.tolist(), co.tolist()
    res["top30"] = {n: [words[i] for i in np.argsort(-v)[:30]] for n, v in {"y": y, **rank}.items()}
    atomic_json(OUT / "results.json", res)

    L = ["# x109 short-request scan (Qwen3.5-4B)", "",
         "User message `request + ' ' + w`; shift = (proj − proj(request alone)) / 8.645. "
         "y = 40-prompt mean shift; word_alone = w as the whole user message. n_tok = request tokens; "
         "sd = sd of the shift over the 71,583 words.", "", "## Requests", ""]
    for r, b in zip(reqs, base):
        rep = res["request_alone_greedy_reply"][r].replace("\n", " ")[:160]
        L.append(f"- `{r[:60]}` ({ntok[r]} tok): proj alone {b:.1f}; junction ok {res['junction_ok'][r]}/{W}; reply: {rep}")
    L += table("Against y", res["vs_y"])
    sp = res["single_prompt_vs_y"]
    agg = {}
    for lab, fn in [("best", np.max), ("median", np.median), ("worst", np.min)]:
        for nm, src in [("single prompt", sp), ("single prompt vs mean of other 39", res["single_prompt_vs_y_loo"])]:
            agg[f"{nm}, {lab}"] = {c: (float(fn([m[c] for m in src])) if isinstance(src[0][c], float) else int(fn([m[c] for m in src]))) for c in COLS if c != "n_tok"}
    L += table("Single long benign prompt against y (column-wise best/median/worst over the 40)", agg)
    L += table("Means of z-scored short-request shifts against y", res["combos_vs_y"])
    L += table("Against the word-alone read", res["vs_word_alone"])
    L += ["", "## Violence/crime verb ranks (1 = top of 71,583)", "", "| ranking | " + " | ".join(VERBS) + " |", "|---" * (len(VERBS) + 1) + "|"]
    for n, d in res["verb_ranks"].items():
        L.append(f"| {n[:40]} | " + " | ".join(str(d.get(v, "")) for v in VERBS) + " |")
    L += ["", "## Cross-request agreement: Spearman / top-100 overlap", "", "| | " + " | ".join(str(i) for i in range(len(reqs))) + " |", "|---" * (len(reqs) + 1) + "|"]
    for i, r in enumerate(reqs):
        L.append(f"| {i} {r[:25]} | " + " | ".join(f"{cs[i, j]:.2f} / {co[i, j]}" for j in range(len(reqs))) + " |")
    L += ["", "## Top 30 words per ranking", ""]
    L += [f"- {n[:40]}: " + ", ".join(ws) for n, ws in res["top30"].items()]
    (OUT / "results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
