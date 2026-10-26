"""x106 Stage 0b (GPU, sharded): leave-one-word-out and leave-one-pair-out deletion on one
model, and the merge that turns the deltas into one nominated unit per prompt.

For each prompt the full text and every deletion variant are rendered with the model's
renderer (the date Stage 0a recovered), forwarded in unpadded batches (variants grouped by exact
token length), and read at each row's last two tokens through the x88 write site; `delta = proj(full) - proj(variant)`
with both projections from the same batched pass. One batch-1 pass on the first prompt of each
shard records the batched-vs-single projection difference. `--text S` runs the same pass on the
cohort's substituted texts; `--cohort` restricts the prompts to the cohort's.

`--merge` concatenates a tag's shards and writes the nomination record per prompt
(`x106_common.nominate`), with the answered-prompt mean projection from Stage 0a and x104.
"""

from __future__ import annotations

import argparse
import glob
import importlib.metadata
import time
from pathlib import Path

import numpy as np
import torch

from x106_common import (
    ALL_TAGS,
    IMPORTED_TAG,
    TEXT_CONDITIONS,
    BatchBoundaryReader,
    all_prompts,
    assert_boundary,
    atomic_json,
    deletion_variants,
    directions_path,
    file_record,
    hook_block_for,
    hook_site,
    inputs_path,
    is_xstest,
    lexicon_words,
    load_cohort,
    load_mu,
    load_tag_model,
    load_tag_tokenizer,
    n_blocks_of,
    nominate,
    nominations_path,
    nominations_shard_path,
    projection_of,
    projections_path,
    read_json,
    render_chat,
    sha256_file,
    sha256_text,
    unit_directions,
    units_with_text,
    x104_labels,
)

N_AUDIT_VARIANTS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, choices=ALL_TAGS)
    parser.add_argument("--text", default="O", choices=TEXT_CONDITIONS)
    parser.add_argument("--cohort", action="store_true", help="only the cohort's prompts (required for --text != O)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--inputs", default=str(inputs_path()))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--projection-only", action="store_true", help="no deletion variants: only proj(full) per text (C and R)")
    parser.add_argument("--merge", action="store_true", help="merge the shards of (tag, text) into the nomination file")
    return parser.parse_args()


def prompt_list(tag: str, text: str, cohort_only: bool) -> list[dict]:
    if text == "O" and not cohort_only:
        return [{"id": r["id"], "prompt": r["prompt"], "prompt_sha256": r["prompt_sha256"]} for r in all_prompts(tag)]
    cohort = load_cohort()
    return [
        {"id": u["id"], "prompt": u["prompts"][text], "prompt_sha256": u["prompt_sha256"][text]}
        for u in units_with_text(cohort, text)
    ]


@torch.inference_mode()
def batched_projections(model, tokenizer, reader: BatchBoundaryReader, mu: np.ndarray, chats: list[str], batch_size: int) -> np.ndarray:
    """Chats grouped by exact token length, so every batch is unpadded and carries no attention mask."""
    ids = [tokenizer(chat, add_special_tokens=False)["input_ids"] for chat in chats]
    groups: dict[int, list[int]] = {}
    for index, seq in enumerate(ids):
        groups.setdefault(len(seq), []).append(index)
    out = np.empty(len(chats), dtype=np.float64)
    for length, members in groups.items():
        for start in range(0, len(members), batch_size):
            chunk = members[start : start + batch_size]
            input_ids = torch.tensor([ids[k] for k in chunk], dtype=torch.long, device=model.device)
            reader.lengths = [length] * len(chunk)
            calls = reader.calls
            model(input_ids=input_ids, use_cache=False)
            if reader.calls != calls + 1:
                raise SystemExit("FATAL: batch reader fired more than once")
            out[chunk] = projection_of(reader.last.numpy(), mu)
    return out


@torch.inference_mode()
def single_projections(model, tokenizer, reader: BatchBoundaryReader, mu: np.ndarray, chats: list[str]) -> np.ndarray:
    """The same reader, one unpadded prompt per forward."""
    out = []
    for chat in chats:
        encoded = tokenizer(chat, return_tensors="pt", add_special_tokens=False).to(model.device)
        reader.lengths = [int(encoded["input_ids"].shape[1])]
        model(**encoded, use_cache=False)
        out.append(float(projection_of(reader.last.numpy(), mu)[0]))
    return np.asarray(out)


def run_shard(args: argparse.Namespace) -> None:
    inputs = read_json(args.inputs)
    if inputs.get("experiment") != "x106" or inputs.get("status") != "frozen":
        raise SystemExit("FATAL: invalid x106 inputs")
    entry = inputs["models"][args.tag]
    if args.text != "O" and not args.cohort:
        raise SystemExit("FATAL: --text S/C/R needs --cohort")
    stage0a = read_json(projections_path(args.tag))
    if stage0a.get("experiment") != "x106" or stage0a.get("model_tag") != args.tag:
        raise SystemExit("FATAL: Stage 0a projections are missing for this tag")
    date = stage0a["rendering"]["date"]
    expected_sha = {r["id"]: r for r in stage0a["rows"]}

    prompts = prompt_list(args.tag, args.text, args.cohort)
    prompts = prompts[args.shard :: args.n_shards]
    if args.limit:
        prompts = prompts[: args.limit]
    print(f"x106 nominate {args.tag}/{args.text} shard {args.shard}/{args.n_shards}: {len(prompts)} prompts", flush=True)

    dir_path = directions_path(args.tag)
    if sha256_file(dir_path) != entry["directions"]["sha256"]:
        raise SystemExit(f"FATAL: {args.tag}: direction artifact differs from the frozen inputs")
    mu = load_mu(args.tag, dir_path)
    tokenizer = load_tag_tokenizer(entry["snapshot"], args.tag)
    model = load_tag_model(entry["snapshot"], args.tag)
    model.eval()
    n_blocks = n_blocks_of(model)
    block = int(entry["hook_block"])
    if entry["n_blocks"] is not None and n_blocks != int(entry["n_blocks"]):
        raise SystemExit(f"FATAL: {args.tag}: model has {n_blocks} blocks, the inputs record {entry['n_blocks']}")
    if args.tag != IMPORTED_TAG and hook_block_for(n_blocks) != block:
        raise SystemExit(f"FATAL: {args.tag}: block {block} is not the x88 rule on {n_blocks} blocks")
    hidden_dim = int(model.config.get_text_config().hidden_size)
    if hidden_dim != mu.shape[1]:
        raise SystemExit(f"FATAL: {args.tag}: direction dim {mu.shape[1]} != hidden {hidden_dim}")
    site = hook_site(model, block)
    boundary_ids = [int(v) for v in entry["boundary_ids"]]
    reader = BatchBoundaryReader()
    handle = site.register_forward_hook(reader)

    out_path = nominations_shard_path(args.tag, args.text, args.shard, args.n_shards)
    done: dict[str, dict] = {}
    if out_path.exists():
        prior = read_json(out_path)
        if prior.get("model_tag") == args.tag and prior.get("text") == args.text:
            done = {r["id"]: r for r in prior.get("rows", []) if r.get("prompt_sha256")}
    expected_text_sha = {r["id"]: r["prompt_sha256"] for r in prompts}
    done = {i: r for i, r in done.items() if expected_text_sha.get(i) == r["prompt_sha256"]}
    t0 = time.time()
    n_forwards = 0
    audit = None
    rows = []
    try:
        for index, row in enumerate(prompts):
            if row["id"] in done:
                rows.append(done[row["id"]])
                continue
            if sha256_text(row["prompt"]) != row["prompt_sha256"]:
                raise SystemExit(f"FATAL: prompt sha mismatch at {row['id']}")
            variants = deletion_variants(row["prompt"])
            if args.projection_only:
                variants = {**variants, "unigram_texts": [], "bigram_texts": []}
            texts = [row["prompt"]] + variants["unigram_texts"] + variants["bigram_texts"]
            chats = [render_chat(tokenizer, t, date) for t in texts]
            if args.text == "O" and row["id"] in expected_sha and expected_sha[row["id"]].get("prompt_sha256") == row["prompt_sha256"]:
                pass  # the full text is the Stage 0a prompt; its rendering was verified there
            for chat in chats[:1]:
                assert_boundary(tokenizer, chat, boundary_ids, args.tag)
            proj = batched_projections(model, tokenizer, reader, mu, chats, args.batch_size)
            n_forwards += len(chats)
            n_uni = len(variants["unigram_texts"])
            full = float(proj[0])
            if audit is None:
                single = single_projections(model, tokenizer, reader, mu, chats[: 1 + N_AUDIT_VARIANTS])
                audit = {
                    "id": row["id"],
                    "batched": [float(v) for v in proj[: 1 + N_AUDIT_VARIANTS]],
                    "single": [float(v) for v in single],
                    "max_abs_diff": float(np.max(np.abs(proj[: 1 + N_AUDIT_VARIANTS] - single))),
                    "stage0a_proj": expected_sha.get(row["id"], {}).get("proj") if args.text == "O" else None,
                }
                print(f"x106 nominate {args.tag}: batched-vs-single max abs diff {audit['max_abs_diff']:.4f}", flush=True)
            rows.append(
                {
                    "id": row["id"],
                    "text": args.text,
                    "prompt": row["prompt"],
                    "prompt_sha256": row["prompt_sha256"],
                    "rendered_chat_sha256": sha256_text(chats[0]),
                    "words": variants["words"],
                    "spans": variants["spans"],
                    "proj_full": full,
                    "delta_unigram": [full - float(v) for v in proj[1 : 1 + n_uni]],
                    "delta_bigram": [full - float(v) for v in proj[1 + n_uni :]],
                }
            )
            if index % 20 == 0 or index == len(prompts) - 1:
                atomic_json(out_path, shard_record(args, entry, block, n_blocks, hidden_dim, date, audit, n_forwards, t0, rows))
                print(f"x106 nominate {args.tag}/{args.text} shard {args.shard}: {index + 1}/{len(prompts)}, {n_forwards} forwards, {(time.time() - t0) / 60:.1f} min", flush=True)
    finally:
        handle.remove()
    atomic_json(out_path, shard_record(args, entry, block, n_blocks, hidden_dim, date, audit, n_forwards, t0, rows))
    print(f"x106 nominate {args.tag}/{args.text} shard {args.shard}: {len(rows)} rows -> {out_path}")


def shard_record(args, entry, block, n_blocks, hidden_dim, date, audit, n_forwards, t0, rows) -> dict:
    return {
        "experiment": "x106",
        "stage": "0b",
        "model_tag": args.tag,
        "model_id": entry["model_id"],
        "model_source": entry["snapshot"],
        "text": args.text,
        "shard": args.shard,
        "n_shards": args.n_shards,
        "site": {"hook_block": block, "state_index": block + 1, "n_blocks": n_blocks, "hidden_dim": hidden_dim, "batch_size": args.batch_size, "padding": "none; variants batched by exact token length, each row read at its last two tokens"},
        "directions": entry["directions"],
        "rendering": {"date": date},
        "deletion": "one whitespace word (or two adjacent words) and one adjacent run of whitespace removed from the prompt; re-rendered; delta = proj(full) - proj(variant), both from the batched pass",
        "batched_vs_single_audit": audit,
        "n_forwards": n_forwards,
        "seconds": time.time() - t0,
        "environment": {"torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
        "rows": rows,
    }


def merge(args: argparse.Namespace) -> None:
    pattern = str(nominations_shard_path(args.tag, args.text, 0, 1)).replace("shard0of1", "shard*of*")
    shards = sorted(glob.glob(pattern))
    if not shards:
        raise SystemExit(f"FATAL: no shards match {pattern}")
    rows: dict[str, dict] = {}
    audits = []
    n_forwards = 0
    seconds = 0.0
    for path in shards:
        obj = read_json(path)
        if obj.get("model_tag") != args.tag or obj.get("text") != args.text:
            raise SystemExit(f"FATAL: {path} belongs to another tag/text")
        if obj.get("batched_vs_single_audit"):
            audits.append(obj["batched_vs_single_audit"])
        n_forwards += int(obj.get("n_forwards", 0))
        seconds += float(obj.get("seconds", 0.0))
        for row in obj["rows"]:
            if row["id"] in rows:
                raise SystemExit(f"FATAL: duplicate row {row['id']} across shards")
            rows[row["id"]] = row
    expected = prompt_list(args.tag, args.text, args.cohort)
    missing = [r["id"] for r in expected if r["id"] not in rows]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} prompts missing from the shards (first {missing[:5]})")
    stage0a = read_json(projections_path(args.tag))
    labels = x104_labels(args.tag)["base"]
    proj_by_id = {r["id"]: r["proj"] for r in stage0a["rows"]}
    answered = [proj_by_id[i] for i, rec in labels.items() if int(rec["refusal"]) == 0 and i in proj_by_id]
    answered_mean = float(np.mean(answered))
    lexicon = lexicon_words()
    out_rows = []
    for row in expected:
        r = rows[row["id"]]
        if not r["delta_unigram"]:
            record = {"projection_only": True}
        else:
            record = nominate(r["words"], r["delta_unigram"], r["delta_bigram"], r["proj_full"], answered_mean)
            record["unit_in_lexicon"] = all(w in lexicon for w in record["unit_norm"].split())
            record["top_in_lexicon"] = record["top_norm"] in lexicon
        out_rows.append({**{k: r[k] for k in ("id", "text", "prompt", "prompt_sha256", "rendered_chat_sha256", "words", "spans", "proj_full", "delta_unigram", "delta_bigram")}, "proj_stage0a": proj_by_id.get(row["id"]), **record})
    out = {
        "experiment": "x106",
        "stage": "0b",
        "model_tag": args.tag,
        "model_id": stage0a["model_id"],
        "text": args.text,
        "cohort_only": bool(args.cohort),
        "n_prompts": len(out_rows),
        "answered_mean_proj": answered_mean,
        "answered_mean_source": "Stage 0a projection averaged over the Hard-1K prompts GuardReasoner-8B (x104) labels answered at base",
        "shards": [file_record(p) for p in shards],
        "batched_vs_single_audits": audits,
        "n_forwards": n_forwards,
        "gpu_seconds": seconds,
        "nomination": nominate.__doc__.strip(),
        "rows": out_rows,
    }
    path = nominations_path(args.tag, args.text)
    atomic_json(path, out)
    z = np.asarray([r.get("top_z", np.nan) for r in out_rows], dtype=float)
    print(f"x106 nominate {args.tag}/{args.text}: {len(out_rows)} prompts -> {path}; median top z {np.nanmedian(z):.2f}, pair wins {np.mean([r.get('pair_wins', False) for r in out_rows]):.2f}")


def main() -> None:
    args = parse_args()
    if args.merge:
        merge(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
