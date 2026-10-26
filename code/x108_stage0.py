"""x108 Stage 0 (GPU) on one model (`--tag`): the boundary projection of every prompt
(`--project`), the leave-one-word-out and leave-one-pair-out deletion pass in shards (default;
`--text S` / `--text R` on the cohort's texts), and the merge into one nominated unit per prompt
(`--merge`).

The prompts are the 655 OR-Bench Toxic prompts rendered exactly as that model's x87 / x88 base
generation rows record them (`rendered_chat_sha256` checked per row; Llama's template date
recovered from the hash as in x106) and the 100 XSTest contrast prompts of the four lexical
categories. The reader, the write site (the model's x88 hook block, the two trailing prefill
positions), the consensus direction, the batched deletion pass, the batched-vs-single audit and
the nomination rule are x106's, imported. The nomination's share-of-excess reference is x106's
answered-Hard-1K mean projection of the same model, reused unchanged.
"""

from __future__ import annotations

import argparse
import glob
import importlib.metadata
import time
from pathlib import Path

import numpy as np
import torch

from x103_extract import audit_site
from x106_common import BOUNDARY_POSITIONS, BatchBoundaryReader, BoundaryReader, assert_boundary, deletion_variants, lexicon_words, nominate, projection_of, recover_date, render_chat, unit_directions
from x106_stage0_nominate import batched_projections, single_projections
from x106_stage0_project import forward
from x108_common import (
    ALL_TAGS,
    N_CONTRAST,
    N_TOXIC,
    TAG,
    TEXT_CONDITIONS,
    X102_XSTEST,
    X104_LABELS,
    all_prompts,
    atomic_json,
    base_generations_path,
    file_record,
    hard1k_reference,
    is_toxic,
    load_cohort,
    load_model,
    model_entry,
    nominations_path,
    nominations_shard_path,
    population_ids,
    projections_path,
    read_json,
    sha256_text,
    states_path,
    units_with_text,
    x104_toxic_labels,
    x106_inputs,
)

N_AUDIT_PROMPTS = 2
N_AUDIT_VARIANTS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=TAG, choices=ALL_TAGS)
    parser.add_argument("--project", action="store_true", help="Stage 0a: states and projections of every prompt")
    parser.add_argument("--merge", action="store_true", help="merge the deletion-pass shards of --text into the nomination file")
    parser.add_argument("--text", default="O", choices=TEXT_CONDITIONS)
    parser.add_argument("--cohort", action="store_true", help="only the cohort's prompts (required for --text != O)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--projection-only", action="store_true", help="no deletion variants: only proj(full) per text (R)")
    parser.add_argument("--limit", type=int, default=0, help="debug: only the first N prompts")
    return parser.parse_args()


# --- Stage 0a ------------------------------------------------------------------------------


def project(tag: str, limit: int) -> None:
    entry = model_entry(tag)
    prompts = all_prompts(tag)
    if limit:
        prompts = prompts[:limit]
    labels = x104_toxic_labels(tag)
    population = population_ids(labels)
    tokenizer, model, site, mu, site_record = load_model(tag, entry)
    minus1 = torch.from_numpy(mu[0].astype(np.float32))
    tbg = torch.from_numpy(mu[1].astype(np.float32))
    _, mu_norms = unit_directions(mu)
    hidden_dim = site_record["hidden_dim"]
    boundary_ids = site_record["boundary_ids"]

    first = next(r for r in prompts if is_toxic(r["id"]))
    date = recover_date(tokenizer, first["prompt"], first["rendered_chat_sha256"])
    print(f"x108 project {tag}: renderer date {date!r}", flush=True)

    def render(row: dict) -> str:
        chat = render_chat(tokenizer, row["prompt"], date)
        if row["rendered_chat_sha256"] is not None and sha256_text(chat) != row["rendered_chat_sha256"]:
            raise SystemExit(f"FATAL: rendered chat does not reproduce the base row's sha256 at {row['id']}")
        assert_boundary(tokenizer, chat, boundary_ids, tag)
        return chat

    audit = audit_site(model, tokenizer, site, site_record["hook_block"], minus1, tbg, [render(r) for r in prompts[:N_AUDIT_PROMPTS]])
    print(f"x108 project {tag}: site audit {audit['status']}", flush=True)
    if audit["status"] != "pass":
        atomic_json(projections_path(tag).with_name("x108_site_audit_FAILED.json"), audit)
        raise SystemExit("FATAL: hook-site audit failed")

    reader = BoundaryReader()
    handle = site.register_forward_hook(reader)
    states = np.empty((len(prompts), 2, hidden_dim), dtype=np.float32)
    n_tokens = []
    for index, row in enumerate(prompts):
        chat = render(row)
        encoded = tokenizer(chat, return_tensors="pt", add_special_tokens=False).to(model.device)
        calls = reader.calls
        forward(model, encoded)
        if reader.calls != calls + 1 or reader.last.shape != (1, 2, hidden_dim):
            raise SystemExit(f"FATAL: reader fired {reader.calls - calls} times for {row['id']}")
        states[index] = reader.last[0].numpy()
        n_tokens.append(int(encoded["input_ids"].shape[1]))
        if index % 100 == 0 or index == len(prompts) - 1:
            print(f"x108 project {tag}: {index + 1}/{len(prompts)} prompts", flush=True)
    handle.remove()

    proj = projection_of(states, mu)
    per_token = np.einsum("ntd,td->nt", states.astype(np.float64), mu / mu_norms[:, None])
    norms = np.linalg.norm(states.astype(np.float64), axis=2)
    rows = []
    for index, row in enumerate(prompts):
        label = labels.get(row["id"])
        rows.append(
            {
                "id": row["id"],
                "source": row["source"],
                "category": row["category"],
                "prompt_sha256": row["prompt_sha256"],
                "n_input_tokens": n_tokens[index],
                "proj": float(proj[index]),
                "proj_by_token": {p: float(v) for p, v in zip(BOUNDARY_POSITIONS, per_token[index])},
                "norm_by_token": {p: float(v) for p, v in zip(BOUNDARY_POSITIONS, norms[index])},
                "base_refusal_guardreasoner": int(label["refusal"]) if label else None,
                "base_request_guardreasoner": label["fields"].get("request") if label else None,
                "in_population": row["id"] in population if is_toxic(row["id"]) else None,
            }
        )
    out_states = states_path(tag)
    out_states.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_states, states=states.astype(np.float16), ids=np.asarray([r["id"] for r in prompts]), boundary_positions=np.asarray(BOUNDARY_POSITIONS), mu=mu.astype(np.float32))
    fp16_err = float(np.max(np.abs(states.astype(np.float16).astype(np.float32) - states)) / (np.abs(states).max() + 1e-12))
    out = {
        "experiment": "x108",
        "stage": "0a",
        "model_tag": tag,
        "model_id": entry["model_id"],
        "model_source": entry["snapshot"],
        "inputs": {
            "x106_inputs": file_record(x106_inputs()),
            "base_generations": file_record(base_generations_path(tag)),
            "x104_labels": file_record(X104_LABELS),
            "x102_xstest": file_record(X102_XSTEST),
            "directions": entry["directions"],
        },
        "population": {
            "rule": "OR-Bench Toxic prompts GuardReasoner-8B (x104) labels refused at base whose request it reads as harmful",
            "n_toxic": N_TOXIC,
            "n_population": len(population),
            "n_contrast": N_CONTRAST,
            "population_ids_sha256": sha256_text("\n".join(population)),
        },
        "n_prompts": len(prompts),
        "site": {**site_record, "read": "forward hook on the decoder block the x88 boundary write hooks; block output at the last two prefill positions", "batch_size": 1},
        "directions": {**entry["directions"], "kind": entry["directions_kind"], "norms": {p: float(v) for p, v in zip(BOUNDARY_POSITIONS, mu_norms)}},
        "rendering": {"renderer": "x87_common.format_user_chat, verified against every Toxic base row's rendered_chat_sha256", "date": date},
        "texts_from": TAG,
        "site_audit": audit,
        "states": {**file_record(out_states), "dtype": "float16", "fp16_max_relative_error": fp16_err},
        "environment": {"torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
        "rows": rows,
    }
    atomic_json(projections_path(tag), out)
    print(f"x108 project {tag}: {len(rows)} projections -> {projections_path(tag)}; states -> {out_states}")


# --- Stage 0b: the deletion pass ----------------------------------------------------------


def prompt_list(tag: str, text: str, cohort_only: bool) -> list[dict]:
    if text == "O" and not cohort_only:
        return [{"id": r["id"], "prompt": r["prompt"], "prompt_sha256": r["prompt_sha256"]} for r in all_prompts(tag)]
    cohort = load_cohort()
    return [{"id": u["id"], "prompt": u["prompts"][text], "prompt_sha256": u["prompt_sha256"][text]} for u in units_with_text(cohort, text)]


def run_shard(args: argparse.Namespace) -> None:
    tag = args.tag
    entry = model_entry(tag)
    if args.text != "O" and not args.cohort:
        raise SystemExit("FATAL: --text S/R needs --cohort")
    stage0a = read_json(projections_path(tag))
    if stage0a.get("experiment") != "x108" or stage0a.get("model_tag") != tag:
        raise SystemExit("FATAL: Stage 0a projections are missing")
    date = stage0a["rendering"]["date"]
    expected = {r["id"]: r for r in stage0a["rows"]}

    prompts = prompt_list(tag, args.text, args.cohort)[args.shard :: args.n_shards]
    if args.limit:
        prompts = prompts[: args.limit]
    print(f"x108 nominate {tag} {args.text} shard {args.shard}/{args.n_shards}: {len(prompts)} prompts", flush=True)

    tokenizer, model, site, mu, site_record = load_model(tag, entry)
    boundary_ids = site_record["boundary_ids"]
    reader = BatchBoundaryReader()
    handle = site.register_forward_hook(reader)

    out_path = nominations_shard_path(tag, args.text, args.shard, args.n_shards)
    done: dict[str, dict] = {}
    if out_path.exists():
        prior = read_json(out_path)
        if prior.get("experiment") == "x108" and prior.get("text") == args.text:
            done = {r["id"]: r for r in prior.get("rows", []) if r.get("prompt_sha256")}
    expected_text_sha = {r["id"]: r["prompt_sha256"] for r in prompts}
    done = {i: r for i, r in done.items() if expected_text_sha.get(i) == r["prompt_sha256"]}
    t0 = time.time()
    n_forwards = 0
    audit = None
    rows = []

    def record() -> dict:
        return {
            "experiment": "x108",
            "stage": "0b",
            "model_tag": tag,
            "model_id": entry["model_id"],
            "model_source": entry["snapshot"],
            "text": args.text,
            "shard": args.shard,
            "n_shards": args.n_shards,
            "site": {**site_record, "batch_size": args.batch_size, "padding": "none; variants batched by exact token length, each row read at its last two tokens"},
            "directions": entry["directions"],
            "rendering": {"date": date},
            "deletion": "one whitespace word (or two adjacent words) and one adjacent run of whitespace removed from the prompt; re-rendered; delta = proj(full) - proj(variant), both from the batched pass",
            "batched_vs_single_audit": audit,
            "n_forwards": n_forwards,
            "seconds": time.time() - t0,
            "environment": {"torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
            "rows": rows,
        }

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
            assert_boundary(tokenizer, chats[0], boundary_ids, tag)
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
                    "stage0a_proj": expected.get(row["id"], {}).get("proj") if args.text == "O" else None,
                }
                print(f"x108 nominate {tag}: batched-vs-single max abs diff {audit['max_abs_diff']:.4f}", flush=True)
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
                atomic_json(out_path, record())
                print(f"x108 nominate {tag} {args.text} shard {args.shard}: {index + 1}/{len(prompts)}, {n_forwards} forwards, {(time.time() - t0) / 60:.1f} min", flush=True)
    finally:
        handle.remove()
    atomic_json(out_path, record())
    print(f"x108 nominate {tag} {args.text} shard {args.shard}: {len(rows)} rows -> {out_path}")


# --- merge ---------------------------------------------------------------------------------


def merge(args: argparse.Namespace) -> None:
    tag = args.tag
    pattern = str(nominations_shard_path(tag, args.text, 0, 1)).replace("shard0of1", "shard*of*")
    shards = sorted(glob.glob(pattern))
    if not shards:
        raise SystemExit(f"FATAL: no shards match {pattern}")
    rows: dict[str, dict] = {}
    audits = []
    n_forwards = 0
    seconds = 0.0
    for path in shards:
        obj = read_json(path)
        if obj.get("experiment") != "x108" or obj.get("text") != args.text or obj.get("model_tag") != tag:
            raise SystemExit(f"FATAL: {path} belongs to another experiment/model/text")
        if obj.get("batched_vs_single_audit"):
            audits.append(obj["batched_vs_single_audit"])
        n_forwards += int(obj.get("n_forwards", 0))
        seconds += float(obj.get("seconds", 0.0))
        for row in obj["rows"]:
            if row["id"] in rows:
                raise SystemExit(f"FATAL: duplicate row {row['id']} across shards")
            rows[row["id"]] = row
    expected = prompt_list(tag, args.text, args.cohort)
    missing = [r["id"] for r in expected if r["id"] not in rows]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} prompts missing from the shards (first {missing[:5]})")
    stage0a = read_json(projections_path(tag))
    if stage0a.get("model_tag") != tag:
        raise SystemExit("FATAL: Stage 0a projections belong to another model")
    proj_by_id = {r["id"]: r["proj"] for r in stage0a["rows"]}
    reference = hard1k_reference(tag)
    answered_mean = reference["answered_mean"]
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
        "experiment": "x108",
        "stage": "0b",
        "model_tag": tag,
        "model_id": stage0a["model_id"],
        "text": args.text,
        "cohort_only": bool(args.cohort),
        "n_prompts": len(out_rows),
        "answered_mean_proj": answered_mean,
        "answered_mean_source": f"x106's {tag} Stage 0a projection averaged over the Hard-1K prompts GuardReasoner-8B (x104) labels answered at base, reused unchanged",
        "x106_reference": {k: reference[k] for k in ("answered_mean", "refused_mean", "sd", "n_hard1k", "n_refused", "n_answered", "projections", "nominations")},
        "shards": [file_record(p) for p in shards],
        "batched_vs_single_audits": audits,
        "n_forwards": n_forwards,
        "gpu_seconds": seconds,
        "nomination": nominate.__doc__.strip(),
        "rows": out_rows,
    }
    path = nominations_path(tag, args.text)
    atomic_json(path, out)
    z = np.asarray([r.get("top_z", np.nan) for r in out_rows], dtype=float)
    print(f"x108 nominate {tag} {args.text}: {len(out_rows)} prompts -> {path}; median top z {np.nanmedian(z):.2f}, pair wins {np.mean([r.get('pair_wins', False) for r in out_rows]):.2f}")


def main() -> None:
    args = parse_args()
    if args.project:
        project(args.tag, args.limit)
    elif args.merge:
        merge(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
