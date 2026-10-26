"""x103 Stage 0/1: freeze the inputs (local), then read boundary states per model (GPU).

`--prepare` writes data/runs/x103/x103_inputs.json from local artifacts: per model the
x101 snapshot, the x88 hook block and boundary ids, the direction artifact and its
sha256, the rendered-chat sha256 of every x101 (x83 for Qwen3.5-4B) generation row,
and the x85 prompt ids and harm words for the leakage check.

`--tag T` runs 520 forward passes (130 prompts x O/S/M/E) on one model, reading the
block output at the two trailing prefill positions through a forward hook on the x88
write site, audits that site against the x88 hook and against `output_hidden_states`,
and writes per-(prompt, position) projections, shift norms and fp16 boundary deltas.
"""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

import numpy as np
import torch

from x103_common import (
    ALL_TAGS,
    BOUNDARY_POSITIONS,
    CONDITIONS,
    IMPORTED_TAG,
    MODELS,
    N_AUDIT_UNITS,
    POSITIONS,
    QWEN4B_DIRECTIONS,
    QWEN4B_HOOK_BLOCK,
    QWEN4B_MODEL_ID,
    X85_COHORT,
    X86_EXPECTED_BOUNDARY_IDS,
    X88_RUN,
    BoundaryReader,
    ExternalBoundaryHook,
    assert_boundary,
    atomic_json,
    bf16_ulp_bound,
    cohort_path,
    consensus_x86_vectors,
    date_candidates,
    deltas_path,
    directions_path,
    expected_renderings,
    file_record,
    hook_block_for,
    hook_site,
    inputs_path,
    leakage_sets,
    load_cohort,
    load_tag_model,
    load_tag_tokenizer,
    manifest_path,
    n_blocks_of,
    projection_record,
    projections_path,
    read_json,
    render_like_x101,
    sha256_file,
    sha256_text,
    unit_directions,
)

AUDIT_DOSE = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="write x103_inputs.json from local artifacts")
    parser.add_argument("--tag", choices=ALL_TAGS)
    parser.add_argument("--inputs", default=str(inputs_path()))
    parser.add_argument("--limit", type=int, default=0, help="debug: only the first N cohort units")
    return parser.parse_args()


def prepare() -> None:
    cohort = load_cohort()
    x101_manifest = read_json(manifest_path())
    x88_manifest = read_json(X88_RUN / "x88_manifest.json")
    if x101_manifest.get("cohort_sha256") != cohort["cohort_sha256"] or x88_manifest.get("status") != "frozen":
        raise SystemExit("FATAL: x101 manifest / x88 manifest are not the frozen artifacts")
    x85 = read_json(X85_COHORT)
    prompt_ids, harm_words = leakage_sets(x85)
    models = {}
    for tag in ALL_TAGS:
        entry = x101_manifest["models"][tag]
        if tag == IMPORTED_TAG:
            if entry["model_id"] != QWEN4B_MODEL_ID:
                raise SystemExit("FATAL: imported row is not Qwen3.5-4B")
            x88_entry = None
            record = {
                "model_id": QWEN4B_MODEL_ID,
                "snapshot": entry["snapshot"],
                "n_blocks": None,  # recorded at run time; the site is x86/x87's block 29
                "hook_block": int(QWEN4B_HOOK_BLOCK),
                "boundary_ids": list(X86_EXPECTED_BOUNDARY_IDS),
                "directions_kind": "x87 consensus of the four x86 fold directions (Qwen3.5-4B)",
                "behaviour_source": "x83 rows restricted to the x101 cohort (as x101 reported them)",
                "date_candidates": [],
            }
        else:
            x88_entry = x88_manifest["models"][tag]
            if x88_entry["snapshot"] != entry["snapshot"] or x88_entry["model_id"] != MODELS[tag]["model_id"]:
                raise SystemExit(f"FATAL: {tag}: x101 and x88 snapshots differ")
            if hook_block_for(int(x88_entry["n_blocks"])) != int(x88_entry["hook_block"]):
                raise SystemExit(f"FATAL: {tag}: x88 hook block is not the frozen relative-depth rule")
            record = {
                "model_id": MODELS[tag]["model_id"],
                "snapshot": entry["snapshot"],
                "revision": entry.get("revision"),
                "n_blocks": int(x88_entry["n_blocks"]),
                "hook_block": int(x88_entry["hook_block"]),
                "boundary_ids": [int(v) for v in x88_entry["boundary_ids"]],
                "directions_kind": "x88 consensus (mean of four fold directions) at the x88 hook block",
                "behaviour_source": "x101 refusal2 rows",
                "date_candidates": date_candidates(entry["chat_template"]["scaffold"]),
            }
        record["state_index"] = record["hook_block"] + 1
        record["directions"] = file_record(directions_path(tag))
        record["rendered_chat_sha256"] = expected_renderings(tag, cohort)
        models[tag] = record
    out = {
        "experiment": "x103",
        "stage": 0,
        "status": "frozen",
        "pre_registered": False,
        "cohort_sha256": cohort["cohort_sha256"],
        "cohort": file_record(cohort_path()),
        "x101_manifest": file_record(manifest_path()),
        "x88_manifest": file_record(X88_RUN / "x88_manifest.json"),
        "leakage": {
            "x85_cohort": file_record(X85_COHORT),
            "x85_cohort_sha256": x85.get("cohort_sha256"),
            "x85_prompt_ids": sorted(prompt_ids),
            "x85_harm_words": sorted(harm_words),
            "x101_prompts_in_x85": sorted(int(u["id"]) for u in cohort["units"] if int(u["id"]) in prompt_ids),
            "x101_words_in_x85": sorted({u["word"] for u in cohort["units"]} & harm_words),
            "n_units_free_of_both": sum(1 for u in cohort["units"] if int(u["id"]) not in prompt_ids and u["word"] not in harm_words),
        },
        "models": models,
    }
    atomic_json(inputs_path(), out)
    print(f"x103 inputs -> {inputs_path()}: {len(models)} models, leakage-free units {out['leakage']['n_units_free_of_both']}")


@torch.inference_mode()
def forward(model, encoded, **kwargs):
    return model(**encoded, use_cache=False, **kwargs)


def audit_site(model, tokenizer, site, block, minus1, tbg, chats: list[str]) -> dict:
    """The read tensor is (a) `output_hidden_states[block + 1]` at the last two positions, the
    site x88's direction extraction read, and (b) moved by exactly dose * mu (to bf16 rounding)
    when the x88 write hook is registered ahead of the reader, so what we read is what it edits."""
    mu = torch.stack([minus1, tbg]).float()
    reports = []
    for chat in chats:
        encoded = tokenizer(chat, return_tensors="pt", add_special_tokens=False).to(model.device)

        reader = BoundaryReader()
        handle = site.register_forward_hook(reader)
        out = forward(model, encoded, output_hidden_states=True)
        handle.remove()
        base = reader.last.clone()
        via_states = out.hidden_states[block + 1][:, -2:, :].float().cpu()
        base_logits = out.logits[:, -1, :].float().cpu()

        # Forward hooks run in registration order: the reader registered after the x88 write
        # sees the edited output, exactly as the next decoder block does.
        hook = ExternalBoundaryHook(AUDIT_DOSE, minus1, tbg)
        write_handle = site.register_forward_hook(hook)
        reader = BoundaryReader()
        handle = site.register_forward_hook(reader)
        out_edited = forward(model, encoded)
        handle.remove()
        write_handle.remove()
        edited = reader.last.clone()

        reader = BoundaryReader()
        handle = site.register_forward_hook(reader)
        forward(model, encoded)
        handle.remove()
        restored = reader.last.clone()

        diff = (edited - base)[0]
        bound = bf16_ulp_bound(edited[0], AUDIT_DOSE * mu)
        within_ulp = bool(torch.all((diff - AUDIT_DOSE * mu).abs() <= bound))
        cos = [float(torch.nn.functional.cosine_similarity(diff[t], mu[t], dim=0)) for t in range(2)]
        norm_ratio = [float(diff[t].norm() / mu[t].norm()) for t in range(2)]
        reports.append(
            {
                "n_input_tokens": int(encoded["input_ids"].shape[1]),
                "reader_equals_output_hidden_states": bool(torch.equal(base, via_states)),
                "reader_vs_hidden_states_max_abs": float((base - via_states).abs().max()),
                "write_hook_prefill_calls": hook.prefill_calls,
                "write_hook_decode_calls": hook.decode_calls,
                "edit_minus_base_equals_dose_mu_within_bf16_ulp": within_ulp,
                "edit_minus_base_vs_dose_mu_max_abs": float((diff - AUDIT_DOSE * mu).abs().max()),
                "edit_minus_base_cosine_to_mu": cos,
                "edit_minus_base_norm_over_mu_norm": norm_ratio,
                "edit_minus_base_geometry_ok": bool(min(cos) >= 0.999 and max(abs(r - 1.0) for r in norm_ratio) <= 0.01),
                "logits_changed_by_write": bool((out_edited.logits[:, -1, :].float().cpu() - base_logits).abs().max() > 0),
                "restored_after_removal": bool(torch.equal(restored, base)),
                "reader_calls_per_forward": reader.calls,
            }
        )
    passed = all(
        r["reader_equals_output_hidden_states"]
        and r["edit_minus_base_equals_dose_mu_within_bf16_ulp"]
        and r["edit_minus_base_geometry_ok"]
        and r["logits_changed_by_write"]
        and r["restored_after_removal"]
        and r["write_hook_prefill_calls"] == 1
        and r["write_hook_decode_calls"] == 0
        and r["reader_calls_per_forward"] == 1
        for r in reports
    )
    return {"status": "pass" if passed else "fail", "dose": AUDIT_DOSE, "texts": reports}


def extract(tag: str, inputs_file: Path, limit: int) -> None:
    inputs = read_json(inputs_file)
    if inputs.get("experiment") != "x103" or inputs.get("status") != "frozen":
        raise SystemExit("FATAL: invalid x103 inputs")
    entry = inputs["models"][tag]
    cohort = load_cohort()
    if cohort["cohort_sha256"] != inputs["cohort_sha256"]:
        raise SystemExit("FATAL: cohort differs from the frozen inputs")
    units = list(cohort["units"])
    if limit:
        units = units[:limit]

    dir_path = Path(entry["directions"]["path"])
    if not dir_path.exists():
        dir_path = directions_path(tag)
    if sha256_file(dir_path) != entry["directions"]["sha256"]:
        raise SystemExit(f"FATAL: {tag}: direction artifact sha256 differs from the frozen inputs")
    minus1, tbg = consensus_x86_vectors(dir_path)
    mu = np.stack([minus1.numpy(), tbg.numpy()]).astype(np.float64)
    _, mu_norms = unit_directions(mu)

    tokenizer = load_tag_tokenizer(entry["snapshot"], tag)
    model = load_tag_model(entry["snapshot"], tag)
    model.eval()
    n_blocks = n_blocks_of(model)
    block = int(entry["hook_block"])
    if entry["n_blocks"] is not None and n_blocks != int(entry["n_blocks"]):
        raise SystemExit(f"FATAL: {tag}: model has {n_blocks} blocks, x88 recorded {entry['n_blocks']}")
    if tag != IMPORTED_TAG and hook_block_for(n_blocks) != block:
        raise SystemExit(f"FATAL: {tag}: hook block {block} is not the x88 rule on {n_blocks} blocks")
    hidden_dim = int(model.config.get_text_config().hidden_size)
    if hidden_dim != mu.shape[1]:
        raise SystemExit(f"FATAL: {tag}: direction dim {mu.shape[1]} != hidden {hidden_dim}")
    site = hook_site(model, block)
    boundary_ids = [int(v) for v in entry["boundary_ids"]]
    expected = entry["rendered_chat_sha256"]
    dates = list(entry.get("date_candidates") or [])

    audit_chats = []
    for unit in units[:N_AUDIT_UNITS]:
        for condition in ("O", "E"):
            chat, _ = render_like_x101(tokenizer, unit["prompts"][condition], expected[condition][str(unit["id"])], dates)
            assert_boundary(tokenizer, chat, boundary_ids, tag)
            audit_chats.append(chat)
    audit = audit_site(model, tokenizer, site, block, minus1, tbg, audit_chats)
    print(f"x103 {tag}: site audit {audit['status']}", flush=True)
    if audit["status"] != "pass":
        atomic_json(projections_path(tag).with_name("x103_site_audit_FAILED.json"), audit)
        raise SystemExit(f"FATAL: {tag}: hook-site audit failed")
    reader = BoundaryReader()
    handle = site.register_forward_hook(reader)

    states = {c: np.empty((len(units), 2, hidden_dim), dtype=np.float32) for c in CONDITIONS}
    row_meta = []
    dates_used = {}
    for index, unit in enumerate(units):
        prompt_id = int(unit["id"])
        meta = {"id": prompt_id, "word": unit["word"], "category": unit["category"], "n_input_tokens": {}}
        for condition in CONDITIONS:
            prompt = unit["prompts"][condition]
            if sha256_text(prompt) != unit["prompt_sha256"][condition]:
                raise SystemExit(f"FATAL: prompt hash mismatch for {prompt_id}/{condition}")
            chat, date_used = render_like_x101(tokenizer, prompt, expected[condition][str(prompt_id)], dates)
            if date_used is not None:
                dates_used[date_used] = dates_used.get(date_used, 0) + 1
            assert_boundary(tokenizer, chat, boundary_ids, tag)
            encoded = tokenizer(chat, return_tensors="pt", add_special_tokens=False).to(model.device)
            calls = reader.calls
            forward(model, encoded)
            if reader.calls != calls + 1 or reader.last.shape != (1, 2, hidden_dim):
                raise SystemExit(f"FATAL: reader fired {reader.calls - calls} times for {prompt_id}/{condition}")
            states[condition][index] = reader.last[0].numpy()
            meta["n_input_tokens"][condition] = int(encoded["input_ids"].shape[1])
        row_meta.append(meta)
        if index % 10 == 0 or index == len(units) - 1:
            print(f"x103 {tag}: {index + 1}/{len(units)} prompts", flush=True)
    handle.remove()

    deltas = np.stack([states[p] - states["O"] for p in POSITIONS], axis=1)  # [n, 3, 2, hidden]
    rows = []
    for index, meta in enumerate(row_meta):
        o_proj = np.einsum("td,td->t", states["O"][index].astype(np.float64), mu / mu_norms[:, None])
        for k, position in enumerate(POSITIONS):
            record = projection_record(deltas[index, k], mu)
            rows.append(
                {
                    "id": meta["id"],
                    "word": meta["word"],
                    "category": meta["category"],
                    "position": position,
                    **record,
                    "o_proj_by_token": {p: float(v) for p, v in zip(BOUNDARY_POSITIONS, o_proj)},
                    "o_norm_by_token": {p: float(v) for p, v in zip(BOUNDARY_POSITIONS, np.linalg.norm(states["O"][index], axis=1))},
                    "n_input_tokens": {"O": meta["n_input_tokens"]["O"], position: meta["n_input_tokens"][position]},
                }
            )

    out_deltas = deltas_path(tag)
    out_deltas.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_deltas,
        deltas=deltas.astype(np.float16),
        ids=np.asarray([m["id"] for m in row_meta], dtype=np.int64),
        positions=np.asarray(POSITIONS),
        boundary_positions=np.asarray(BOUNDARY_POSITIONS),
        mu=mu.astype(np.float32),
    )
    fp16_max_rel_error = float(np.max(np.abs(deltas.astype(np.float16).astype(np.float32) - deltas)) / (np.abs(deltas).max() + 1e-12))
    out = {
        "experiment": "x103",
        "stage": 1,
        "model_tag": tag,
        "model_id": entry["model_id"],
        "model_source": entry["snapshot"],
        "cohort_sha256": cohort["cohort_sha256"],
        "inputs": file_record(inputs_file),
        "n_units": len(units),
        "site": {
            "hook_block": block,
            "state_index": block + 1,
            "n_blocks": n_blocks,
            "relative_depth_rule_block": hook_block_for(n_blocks),
            "hidden_dim": hidden_dim,
            "boundary_ids": boundary_ids,
            "read": "forward hook on the decoder block the x88 boundary write hooks; block output at the last two prefill positions",
            "batch_size": 1,
        },
        "directions": {
            **entry["directions"],
            "kind": entry["directions_kind"],
            "norms": {p: float(v) for p, v in zip(BOUNDARY_POSITIONS, mu_norms)},
        },
        "rendering": {
            "renderer": "x87_common.format_user_chat, verified against every generation row's rendered_chat_sha256",
            "date_strings_recovered": dates_used,
        },
        "site_audit": audit,
        "deltas": {**file_record(out_deltas), "dtype": "float16", "fp16_max_relative_error": fp16_max_rel_error},
        "environment": {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "rows": rows,
    }
    atomic_json(projections_path(tag), out)
    print(f"x103 {tag}: {len(rows)} cells -> {projections_path(tag)}; deltas -> {out_deltas}")


def main() -> None:
    args = parse_args()
    if args.prepare:
        prepare()
        return
    if not args.tag:
        raise SystemExit("--tag or --prepare is required")
    extract(args.tag, Path(args.inputs), args.limit)


if __name__ == "__main__":
    main()
