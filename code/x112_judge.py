"""x112 judging (local): refusal2 and the hijack label, per model.

x111's judges, imported: each cell's twenty responses are judged as two units of ten (responses
0-9 and 10-19; unit id `cell_id#h0` / `cell_id#h1`). `--judge refusal2` judges every generated
cell of the model (the pool O cells and the grid E cells), x64 contract (DeepSeek V4 Flash on
OpenRouter); `--per-response-fallback` is x100's one-response-per-call fallback.
`--judge hijack --no-reasoning` is x111's reasoning-disabled hijack pass on the grid E cells
(hijack_fast.jsonl). Both resume on failed rows.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x112_common import N_HALF, N_SAMPLES, TAGS, paths, read_json, sha256_text  # noqa: E402
from x111_judge import ATTEMPT_SEEDS, CostMeter, hijack_row, judge_row, judge_row_per_response, load_api_key  # noqa: E402


def load_generations(tag: str) -> tuple[list[dict], str]:
    """Every generated cell of the model (pool O cells and grid E shards), keyed by cell id."""
    rows: dict[str, dict] = {}
    shas = []
    for path in sorted(glob.glob(str(paths(tag)["gen"] / "generations_*.json"))):
        obj = read_json(path)
        shas.append(sha256_text(Path(path).read_text(encoding="utf-8")))
        for r in obj["rows"]:
            if r["cell_id"] in rows:
                raise SystemExit(f"FATAL: {r['cell_id']} generated twice")
            if sha256_text(r["prompt"]) != r["prompt_sha256"] or len(r["llm_responses"]) != N_SAMPLES:
                raise SystemExit(f"FATAL: invalid generation row {r['cell_id']}")
            rows[r["cell_id"]] = r
    out = [{**r, "index": i} for i, r in enumerate(sorted(rows.values(), key=lambda r: r["cell_id"]))]
    return out, sha256_text("\n".join(shas))


def split_units(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        for h in range(N_SAMPLES // N_HALF):
            sl = slice(h * N_HALF, (h + 1) * N_HALF)
            out.append({**r, "cell_id": f"{r['cell_id']}#h{h}", "parent_cell_id": r["cell_id"], "half": h, "index": 2 * r["index"] + h, "llm_responses": r["llm_responses"][sl], "response_sha256": r["response_sha256"][sl]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, choices=TAGS)
    parser.add_argument("--judge", required=True, choices=("refusal2", "hijack"))
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--cap", type=float, default=30.0)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--per-response-fallback", action="store_true")
    parser.add_argument("--no-reasoning", action="store_true")
    args = parser.parse_args()
    tag = args.tag
    jdir = paths(tag)["judge"]
    rows, gen_sha = load_generations(tag)
    rows = split_units(rows)
    if args.judge == "hijack":
        rows = [r for r in rows if r["kind"] == "E"]
    out_path = jdir / ("hijack_fast.jsonl" if args.judge == "hijack" and args.no_reasoning else f"{args.judge}.jsonl")
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if not rec.get("failed"):
                done.add((rec["cell_id"], tuple(rec["response_sha256"])))
    todo = [r for r in rows if (r["cell_id"], tuple(r["response_sha256"])) not in done]
    print(f"{tag} {args.judge}: {len(todo)} to judge ({len(done)} done)", flush=True)
    if not todo:
        return
    api_key = load_api_key()
    meter = CostMeter(args.cap)
    jdir.mkdir(parents=True, exist_ok=True)
    grid_tag = f"x112|{tag}"

    def work(row):
        if args.judge == "hijack":
            return hijack_row(api_key, row, meter, tuple(s + args.seed_offset for s in ATTEMPT_SEEDS), reasoning=not args.no_reasoning)
        judged = {"id": row["index"], "condition": row["condition"], "prompt": row["prompt"], "prompt_sha256": row["prompt_sha256"], "response_sha256": row["response_sha256"], "arms_sha256": grid_tag, "llm_responses": row["llm_responses"]}
        rec = (judge_row_per_response if args.per_response_fallback else judge_row)(api_key, judged, meter, gen_sha)
        if rec is not None:
            rec["cell_id"] = row["cell_id"]
        return rec

    n_ok = n_failed = 0
    with out_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(args.threads) as pool:
        futures = [pool.submit(work, r) for r in todo]
        for future in as_completed(futures):
            rec = future.result()
            if rec is None:
                continue
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            n_failed += bool(rec.get("failed"))
            n_ok += not rec.get("failed")
    usd, pt, ct = meter.snapshot()
    print(f"{tag} {args.judge}: +{n_ok} ok, {n_failed} failed; ${usd:.2f}, prompt_tokens={pt}, completion_tokens={ct}", flush=True)
    with (jdir / "cost_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"judge": args.judge + ("_fast" if args.no_reasoning else ""), "ok": n_ok, "failed": n_failed, "usd": usd, "prompt_tokens": pt, "completion_tokens": ct}) + "\n")
    if meter.aborted:
        raise SystemExit("COST CAP HIT")


if __name__ == "__main__":
    main()
