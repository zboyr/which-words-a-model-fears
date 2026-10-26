"""x101 Stage 0b: per-model preflight with one real model load (GPU), and manifest assembly (CPU).

Per fresh tag (``--tag``):
  1. resolve the frozen x88 snapshot and load tokenizer + model with the x88 loaders;
  2. render the two audit prompts and two cohort prompts with ``format_user_chat``;
     the scaffold around the user turn must be one template constant; record it,
     its sha256, and whether it differs from the campaign renderer's scaffold
     (``extract_features_v2``: same call with a pinned ``date_string``);
  3. locate the gate scoring position: greedy-decode the x83 detection prompt for
     ``N_GATE_PROBES`` distinct-word units at positions S and E; the longest common
     token prefix is the template's fixed pre-answer emission (empty when the first
     generated token is the answer). It must decode to scaffolding only;
  4. score P_detect at that position for the probes and run one short sampled
     generation row, so the exact detection and generation paths are exercised.

``--assemble`` (CPU) merges the per-tag preflights into data/runs/x101/x101_manifest.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import platform
import re
from pathlib import Path

from x101_common import (
    ALL_TAGS,
    AUDIT_PROMPTS,
    DETECT_INSTRUCTION,
    FRESH_TAGS,
    GREEDY_TOKENS,
    MODELS,
    N_GATE_PROBES,
    QWEN4B_MODEL_ID,
    X83_RUN,
    X88_RUN,
    atomic_json,
    cohort_path,
    file_record,
    fixed_pre_answer_ids,
    format_user_chat,
    load_cohort,
    load_model,
    load_tokenizer,
    manifest_path,
    preflight_path,
    read_json,
    scoring_prefix_ids,
    sha256_text,
    surface_variants,
    template_like,
)

CAMPAIGN_DATE_STRING = "09 Jul 2026"  # extract_features_v2.DATE_STRING
SMOKE_NEW_TOKENS = 48
PROMPT_SLOT = "{PROMPT}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tag", choices=FRESH_TAGS)
    group.add_argument("--assemble", action="store_true")
    parser.add_argument("--x88-manifest", default=str(X88_RUN / "x88_manifest.json"))
    return parser.parse_args()


def resolve_snapshot(tag: str, x88_manifest_path: Path) -> dict:
    manifest = read_json(x88_manifest_path)
    entry = manifest["models"][tag]
    if entry.get("model_id") != MODELS[tag]["model_id"]:
        raise SystemExit(f"FATAL: x88 manifest model id for {tag} is not the registry's")
    snapshot = Path(entry["snapshot"])
    if not (snapshot / "config.json").exists():
        raise SystemExit(f"FATAL: snapshot {snapshot} is not on disk (config.json missing)")
    weights = sorted(p.name for p in snapshot.glob("*.safetensors"))
    if not weights:
        raise SystemExit(f"FATAL: snapshot {snapshot} has no safetensors shards")
    return {
        "model_id": entry["model_id"],
        "snapshot": str(snapshot),
        "revision": entry["revision"],
        "x88_boundary_ids": entry["boundary_ids"],
        "weight_files": weights,
    }


def scaffold_of(rendered: str, prompt: str) -> str:
    if rendered.count(prompt) != 1:
        raise SystemExit("FATAL: the rendered chat does not contain the user prompt exactly once")
    return rendered.replace(prompt, PROMPT_SLOT)


def campaign_render(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False, date_string=CAMPAIGN_DATE_STRING,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def thinking_block_state(scaffold: str) -> str:
    if "<think>" not in scaffold:
        return "none"
    return "empty" if re.search(r"<think>\s*</think>", scaffold) else "non_empty"


def greedy_ids(model, tokenizer, prefix_ids: list[int], n_tokens: int) -> list[int]:
    import torch

    input_ids = torch.tensor([prefix_ids], dtype=torch.long).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=n_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return [int(t) for t in out[0, input_ids.shape[1]:].tolist()]


def probe_units(units: list[dict], n: int) -> list[dict]:
    chosen, seen = [], set()
    for unit in units:
        if unit["word"] in seen:
            continue
        seen.add(unit["word"])
        chosen.append(unit)
        if len(chosen) == n:
            return chosen
    raise SystemExit("FATAL: not enough distinct-word units for the gate probes")


def run_tag(tag: str, x88_manifest_path: Path) -> None:
    import torch

    from x81_generate import eos_id_set, generate_row
    from x83_detect import variant_log_probs

    cohort = load_cohort()
    units = list(cohort["units"])
    source = resolve_snapshot(tag, x88_manifest_path)
    tokenizer = load_tokenizer(source["snapshot"], tag)
    model = load_model(source["snapshot"], tag)
    model.eval()
    status = "pass"
    problems: list[str] = []

    # 1. Chat-template audit: the scaffold around the user turn is one constant.
    audit_prompts = list(AUDIT_PROMPTS) + [units[0]["prompts"]["O"], units[1]["prompts"]["O"]]
    scaffolds = {scaffold_of(format_user_chat(tokenizer, p), p) for p in audit_prompts}
    if len(scaffolds) != 1:
        raise SystemExit(f"FATAL: {tag} scaffold is not a template constant: {len(scaffolds)} variants")
    scaffold = scaffolds.pop()
    campaign_scaffolds = {scaffold_of(campaign_render(tokenizer, p), p) for p in audit_prompts}
    campaign_scaffold = campaign_scaffolds.pop() if len(campaign_scaffolds) == 1 else None
    thinking = thinking_block_state(scaffold)
    if thinking == "non_empty":
        status = "fail_thinking_block"
        problems.append("rendered scaffold carries a non-empty thinking block")
    scaffold_ids = tokenizer(format_user_chat(tokenizer, audit_prompts[0]), add_special_tokens=False)["input_ids"]
    if list(scaffold_ids[-2:]) != list(source["x88_boundary_ids"]):
        problems.append(f"trailing ids {scaffold_ids[-2:]} differ from the x88 audit {source['x88_boundary_ids']}")

    # 2. Gate scoring position from greedy detection answers to distinct-word probes.
    probes = probe_units(units, N_GATE_PROBES)
    greedy: dict[str, list[list[int]]] = {}
    chats: dict[str, list[str]] = {}
    for position in ("S", "E"):
        chats[position] = [
            format_user_chat(tokenizer, DETECT_INSTRUCTION.format(text=unit["prompts"][position])) for unit in probes
        ]
        greedy[position] = [
            greedy_ids(model, tokenizer, tokenizer(chat, add_special_tokens=False)["input_ids"], GREEDY_TOKENS)
            for chat in chats[position]
        ]
    fixed_by_position = {p: fixed_pre_answer_ids(greedy[p]) for p in greedy}
    if fixed_by_position["S"] != fixed_by_position["E"]:
        status = "fail_fixed_prefix_disagrees"
        problems.append(f"fixed prefix differs between S {fixed_by_position['S']} and E {fixed_by_position['E']}")
    fixed = fixed_by_position["E"]
    fixed_text = tokenizer.decode(fixed, skip_special_tokens=False) if fixed else ""
    if fixed and not template_like(fixed_text):
        status = "fail_fixed_prefix_not_template"
        problems.append(f"fixed prefix decodes to {fixed_text!r}, which is not scaffolding")

    # 3. P_detect at the scoring position for the probes (exercises the exact gate path).
    probe_records = []
    for position in ("S", "E"):
        for unit, chat, ids in zip(probes, chats[position], greedy[position]):
            chat_ids = tokenizer(chat, add_special_tokens=False)["input_ids"]
            prefix = scoring_prefix_ids(chat_ids, fixed)
            variants = variant_log_probs(model, tokenizer, prefix, surface_variants(unit["word"]))
            probe_records.append(
                {
                    "id": unit["id"],
                    "word": unit["word"],
                    "position": position,
                    "greedy_token_ids": ids,
                    "greedy_text_raw": tokenizer.decode(ids, skip_special_tokens=False),
                    "greedy_answer": tokenizer.decode(ids[len(fixed):], skip_special_tokens=True),
                    "p_detect": float(sum(v["prob"] for v in variants)),
                    "scoring_position": len(prefix),
                }
            )

    # 4. One short sampled generation row through the x83 contract.
    eos_ids = eos_id_set(tokenizer, model)
    smoke_chat = format_user_chat(tokenizer, units[0]["prompts"]["O"])
    texts, lengths, truncated = generate_row(model, tokenizer, eos_ids, smoke_chat, SMOKE_NEW_TOKENS, 101)
    if len(texts) != 10 or any(not value.strip() for value in texts):
        status = "fail_generation_smoke"
        problems.append("generation smoke produced an empty response")

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    output = {
        "experiment": "x101",
        "stage": "0b",
        "status": status,
        "problems": problems,
        "model_tag": tag,
        **source,
        "trust_remote_code": MODELS[tag]["trust_remote_code"],
        "cohort_sha256": cohort["cohort_sha256"],
        "cohort": file_record(cohort_path()),
        "rendered_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "chat_template": {
            "renderer": "x87_common.format_user_chat (user turn only, add_generation_prompt, enable_thinking=False)",
            "scaffold": scaffold,
            "scaffold_sha256": sha256_text(scaffold),
            "thinking_block": thinking,
            "campaign_renderer_scaffold_sha256": sha256_text(campaign_scaffold) if campaign_scaffold else None,
            "differs_from_campaign_renderer": campaign_scaffold != scaffold,
            "trailing_ids": [int(v) for v in scaffold_ids[-2:]],
            "n_scaffold_tokens": len(scaffold_ids) - len(tokenizer(audit_prompts[0], add_special_tokens=False)["input_ids"]),
        },
        "gate": {
            "rule": "score the inserted word at the first position after the fixed pre-answer tokens (longest common greedy prefix over distinct-word probes)",
            "fixed_pre_answer_ids": fixed,
            "fixed_pre_answer_text": fixed_text,
            "scoring_offset": len(fixed),
            "probes": probe_records,
        },
        "generation_smoke": {
            "max_new_tokens": SMOKE_NEW_TOKENS,
            "eos_ids": sorted(eos_ids),
            "gen_tokens": lengths,
            "truncated": truncated,
            "response_heads": [value[:160] for value in texts[:3]],
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "accelerate": importlib.metadata.version("accelerate"),
            "device": device,
        },
    }
    atomic_json(preflight_path(tag), output)
    print(
        f"x101 preflight {tag}: {status}; thinking={thinking}; fixed prefix {fixed} {fixed_text!r}; "
        f"probe p_detect {[round(r['p_detect'], 3) for r in probe_records]} -> {preflight_path(tag)}"
    )
    if status != "pass":
        raise SystemExit(f"PREFLIGHT FAILED: {problems}")


def assemble() -> None:
    cohort = load_cohort()
    x83_detect = read_json(X83_RUN / "x83_detect.json")
    models = {
        "qwen": {
            "model_id": QWEN4B_MODEL_ID,
            "snapshot": x83_detect["model_source"],
            "source": "x83 detection, generations and judgments restricted to the x101 cohort; not rerun",
            "x83_cohort_sha256": cohort["x83_cohort_sha256"],
        }
    }
    for tag in FRESH_TAGS:
        preflight = read_json(preflight_path(tag))
        if preflight.get("status") != "pass":
            raise SystemExit(f"FATAL: {tag} preflight status {preflight.get('status')}")
        models[tag] = {
            key: preflight[key]
            for key in ("model_id", "snapshot", "revision", "trust_remote_code", "chat_template", "gate", "environment", "rendered_at_utc")
        }
        models[tag]["gate"] = {k: v for k, v in preflight["gate"].items() if k != "probes"}
        models[tag]["preflight"] = file_record(preflight_path(tag))
    manifest = {
        "experiment": "x101",
        "status": "frozen",
        "cohort_sha256": cohort["cohort_sha256"],
        "cohort": file_record(cohort_path()),
        "tags": list(ALL_TAGS),
        "models": models,
        "notes": [
            "The x88 venv and the seven model snapshots had been swept from the cluster scratch; the frozen "
            "revisions were re-downloaded into the shared cache and run in the x100 venv (transformers 5.15).",
            "No model emits fixed tokens before the answer, so the gate scoring offset is 0 everywhere.",
            "Judge: one qwen27b/M row (prompt 21446) returned malformed JSON on all 36 attempts; the analysis "
            "drops that prompt from qwen27b in all four conditions (counts.dropped_unjudged_prompts).",
        ],
    }
    atomic_json(manifest_path(), manifest)
    print(f"x101 manifest: {len(models)} models -> {manifest_path()}")


def main() -> None:
    args = parse_args()
    if args.assemble:
        assemble()
    else:
        run_tag(args.tag, Path(args.x88_manifest))


if __name__ == "__main__":
    main()
