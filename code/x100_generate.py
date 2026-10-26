"""x100 Stages 1/3: generate behavior (or pilot) cells under the exact x83/x85 contract.

One chunk of the 2,880 behavior conditions per call (behavior_id % num_chunks),
resumable by id, so a Slurm array can be resubmitted freely. --pilot generates
the 40 pilot cells (pilot-namespaced IDs and seeds) into data/runs/x100/pilot/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from x81_generate import MAX_NEW, RETRY_NEW, TEMPERATURE, TOP_K, TOP_P, eos_id_set, generate_row
from x85_generate import valid_completed
from x100_common import (
    GEN_DIR,
    MODEL_ID,
    N_GEN_CHUNKS,
    N_RESPONSES,
    PILOT,
    atomic_json,
    cohort_path,
    file_record,
    format_user_chat,
    load_causal_lm,
    read_json,
    sha256_text,
    stable_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default=str(cohort_path()))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--chunk", type=int)
    group.add_argument("--pilot", action="store_true")
    parser.add_argument("--num-chunks", type=int, default=N_GEN_CHUNKS)
    parser.add_argument("--out")
    return parser.parse_args()


def chunk_path(chunk: int, num_chunks: int) -> Path:
    return GEN_DIR / f"x100_generations_chunk_{chunk:02d}-of-{num_chunks:02d}.json"


def pilot_path() -> Path:
    return PILOT / "x100_pilot_generations.json"


def main() -> None:
    args = parse_args()
    cohort_file = Path(args.cohort)
    manifest = read_json(cohort_file)
    if manifest.get("experiment") != "x100" or manifest.get("model_id") != MODEL_ID or manifest.get("status") != "frozen":
        raise SystemExit("FATAL: invalid x100 cohort")
    cohort_sha = manifest["cohort_sha256"]
    if args.pilot:
        cells = [dict(c, behavior_id=c["pilot_id"]) for c in manifest["pilot_cells"]]
        out_path = Path(args.out) if args.out else pilot_path()
        scope = "pilot"
    else:
        if not 0 <= args.chunk < args.num_chunks:
            raise SystemExit("FATAL: chunk out of range")
        cells = [c for c in manifest["cells"] if c["behavior"] and int(c["behavior_id"]) % args.num_chunks == args.chunk]
        out_path = Path(args.out) if args.out else chunk_path(args.chunk, args.num_chunks)
        scope = f"chunk {args.chunk}/{args.num_chunks}"
    cells.sort(key=lambda c: int(c["behavior_id"]))
    expected = {int(c["behavior_id"]): c for c in cells}

    done: dict[int, dict] = {}
    if out_path.exists():
        prior = read_json(out_path)
        if prior.get("cohort_sha256") != cohort_sha or prior.get("scope") != scope:
            raise SystemExit("FATAL: resume output belongs to another scope/cohort")
        for row in prior.get("rows", []):
            bid = int(row["id"])
            if bid not in expected:
                raise SystemExit(f"FATAL: unexpected resume row {bid}")
            if not valid_completed(row, expected[bid], cohort_sha):
                raise SystemExit(f"FATAL: invalid completed generation row {bid}")
            if bid in done:
                raise SystemExit(f"FATAL: duplicate completed generation row {bid}")
            done[bid] = row

    todo = [c for c in cells if int(c["behavior_id"]) not in done]
    if not todo:
        print(f"x100 generate {scope}: nothing new ({len(done)} rows) -> {out_path}")
        return
    model_source = manifest["model_snapshot"]
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = load_causal_lm(model_source)
    eos_ids = eos_id_set(tokenizer, model)

    def checkpoint() -> None:
        atomic_json(
            out_path,
            {
                "experiment": "x100",
                "stage": 1 if args.pilot else 3,
                "scope": scope,
                "model_id": MODEL_ID,
                "model_source": model_source,
                "cohort_sha256": cohort_sha,
                "arms_sha256": cohort_sha,
                "input": file_record(cohort_file),
                "config": {
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                    "num_return_sequences": N_RESPONSES,
                    "max_new_tokens": MAX_NEW,
                    "retry_max_new_tokens": RETRY_NEW,
                    "thinking": False,
                    "seed_policy": "x85 stable_seed form, x100 namespace, base 100: sha256('x100|100|<cell_id>|<condition>')",
                },
                "rows": [done[key] for key in sorted(done)],
            },
        )

    for index, cell in enumerate(todo, start=1):
        bid = int(cell["behavior_id"])
        text = cell["prompt"]
        if sha256_text(text) != cell["prompt_sha256"]:
            raise SystemExit(f"FATAL: prompt hash mismatch for {cell['cell_id']}")
        chat = format_user_chat(tokenizer, text)
        if sha256_text(chat) != cell["rendered_chat_sha256"]:
            raise SystemExit(f"FATAL: rendered chat mismatch for {cell['cell_id']}")
        condition = cell["condition"]
        seed = stable_seed(cell["cell_id"], condition)
        texts, lengths, truncated = generate_row(model, tokenizer, eos_ids, chat, MAX_NEW, seed)
        used_retry = any(truncated)
        retry_seed = None
        if used_retry:
            retry_seed = stable_seed(cell["cell_id"], f"{condition}:retry4096")
            texts, lengths, truncated = generate_row(model, tokenizer, eos_ids, chat, RETRY_NEW, retry_seed)
        if len(texts) != N_RESPONSES or any(not value.strip() for value in texts):
            raise SystemExit(f"FATAL: incomplete/empty generation row {cell['cell_id']}")
        done[bid] = {
            "id": bid,
            "cell_id": cell["cell_id"],
            "population": cell["population"],
            "condition": condition,
            "prompt_key": cell["prompt_key"],
            "word_index": cell["word_index"],
            "word": cell["word"],
            "neutral": cell["neutral"],
            "prompt": text,
            "prompt_sha256": cell["prompt_sha256"],
            "rendered_chat_sha256": cell["rendered_chat_sha256"],
            "arms_sha256": cohort_sha,
            "seed": seed,
            "used_retry": used_retry,
            "retry_seed": retry_seed,
            "llm_responses": texts,
            "response_sha256": [sha256_text(v) for v in texts],
            "gen_tokens": lengths,
            "truncated": truncated,
        }
        checkpoint()
        print(f"x100 generate {scope}: {index}/{len(todo)} new", flush=True)
    checkpoint()
    print(f"x100 generate {scope}: {len(done)} rows -> {out_path}")


if __name__ == "__main__":
    main()
