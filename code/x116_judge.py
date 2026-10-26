"""x116 Stage 4 (local): `refusal2` on every R_rand cell of one model.

x113's refusal2 judge unchanged (`x81_judge_refusal.judge_row`, DeepSeek V4 Flash on OpenRouter,
ten responses per call), with the OpenRouter upstream pinned to one provider, no fallback
(`x114_judge_refusal.pin_provider`; DeepInfra, see knowledge/infrastructure.md), recorded as
`judge_provider` on every record. Resumes on failed rows; `--seed-offset` shifts the attempt seeds
of a resume pass and `--per-response-fallback` is x100's one-response-per-call fallback.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x110_judge import judge_row_per_response  # noqa: E402
from x114_judge_refusal import pin_provider  # noqa: E402
import x81_judge_refusal  # noqa: E402
from x81_judge_refusal import ATTEMPT_SEEDS, CostMeter, load_api_key  # noqa: E402
from x116_common import ALL_TAGS, N_RESPONSES, cells, cells_sha256, load_texts, paths, read_json, sha256_text  # noqa: E402


def load_generations(tag: str) -> tuple[list[dict], str, str]:
    texts = load_texts(tag)
    grid = cells(tag, texts)
    grid_sha = cells_sha256(grid)
    rows: dict[str, dict] = {}
    shard_shas = []
    for path in sorted(glob.glob(str(paths(tag)["gen"] / "generations_shard*of*.json"))):
        obj = read_json(path)
        if obj.get("experiment") != "x116" or obj.get("model_tag") != tag or obj.get("cells_sha256") != grid_sha:
            raise SystemExit(f"FATAL: {path} belongs to another tag or grid")
        shard_shas.append(sha256_text(Path(path).read_text(encoding="utf-8")))
        for r in obj["rows"]:
            rows[r["cell_id"]] = r
    out = []
    for index, c in enumerate(grid):
        r = rows.get(c["cell_id"])
        if r is None:
            continue
        if r["prompt_sha256"] != c["prompt_sha256"] or len(r["llm_responses"]) != N_RESPONSES:
            raise SystemExit(f"FATAL: invalid generation row {c['cell_id']}")
        out.append({**r, "index": index})
    print(f"{tag}: {len(out)}/{len(grid)} cells generated")
    return out, grid_sha, sha256_text("\n".join(shard_shas))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, choices=ALL_TAGS)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--cap", type=float, default=20.0)
    parser.add_argument("--provider", default="DeepInfra")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--per-response-fallback", action="store_true")
    args = parser.parse_args()
    rows, grid_sha, gen_sha = load_generations(args.tag)
    out_path = paths(args.tag)["refusal2"]
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if not rec.get("failed"):
                done.add((rec["cell_id"], tuple(rec["response_sha256"])))
    todo = [r for r in rows if (r["cell_id"], tuple(r["response_sha256"])) not in done]
    print(f"{args.tag}/refusal2: {len(todo)} to judge ({len(done)} done)")
    if not todo:
        return
    pin_provider(args.provider)
    # judge_row looks `call` up in x81_judge_refusal's namespace at call time, so the pin applies
    import x110_judge

    x110_judge.call = x81_judge_refusal.call  # the per-response fallback imported `call` by name
    api_key = load_api_key()
    meter = CostMeter(args.cap)
    seeds = tuple(s + args.seed_offset for s in ATTEMPT_SEEDS)
    if args.seed_offset:
        x81_judge_refusal.ATTEMPT_SEEDS = seeds

    def work(row):
        judged = {"id": row["index"], "condition": row["arm"], "prompt": row["prompt"], "prompt_sha256": row["prompt_sha256"], "response_sha256": row["response_sha256"], "arms_sha256": grid_sha, "llm_responses": row["llm_responses"]}
        fn = judge_row_per_response if args.per_response_fallback else x81_judge_refusal.judge_row
        rec = fn(api_key, judged, meter, gen_sha)
        if rec is not None:
            rec["cell_id"] = row["cell_id"]
            rec["prompt_id"] = row["id"]
            rec["judge_provider"] = args.provider
        return rec

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_failed = 0
    with out_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(args.threads) as pool:
        for future in as_completed([pool.submit(work, r) for r in todo]):
            rec = future.result()
            if rec is None:
                continue
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            n_failed += bool(rec.get("failed"))
            n_ok += not rec.get("failed")
    usd, pt, ct = meter.snapshot()
    print(f"{args.tag}/refusal2: +{n_ok} ok, {n_failed} failed; ${usd:.2f}")
    with paths(args.tag)["cost_log"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"tag": args.tag, "judge": "refusal2" + ("_per_response" if args.per_response_fallback else ""), "provider": args.provider, "seed_offset": args.seed_offset, "ok": n_ok, "failed": n_failed, "usd": usd, "prompt_tokens": pt, "completion_tokens": ct}) + "\n")
    if meter.aborted:
        raise SystemExit("COST CAP HIT")


if __name__ == "__main__":
    main()
