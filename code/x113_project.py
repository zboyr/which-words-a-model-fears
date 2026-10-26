"""x113 Stage 5 (GPU, one model per job): the projection of O, S_T and C_T on the model's
direction at x106's read site.

x106 Stage 0a's path: every text rendered with the model's renderer (the date x106 recovered),
the read site audited against the x88 write hook (x103's `audit_site`), the block output at the
two trailing prefill positions read one text per forward and projected on the model's consensus
direction, summed over the two positions; the O projections are checked against x106's Stage 0a
rows. Writes per text the raw projection and its value in the model's x106 Hard-1K sd unit.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x103_extract import audit_site  # noqa: E402
from x106_common import (  # noqa: E402
    BOUNDARY_POSITIONS,
    IMPORTED_TAG,
    BoundaryReader,
    assert_boundary,
    directions_path,
    hook_block_for,
    hook_site,
    load_mu,
    load_tag_model,
    load_tag_tokenizer,
    n_blocks_of,
    projection_of,
    sha256_file,
    unit_directions,
)
from x113_common import (  # noqa: E402
    ALL_TAGS,
    ARMS,
    atomic_json,
    file_record,
    load_texts,
    paths,
    read_json,
    render_chat,
    sd_unit,
    x106_inputs_path,
    x106_projections_path,
)

N_AUDIT = 2


@torch.inference_mode()
def forward(model, encoded):
    return model(**encoded, use_cache=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, choices=ALL_TAGS)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    tag = args.tag
    texts = load_texts(tag)
    units = texts["units"][: args.limit] if args.limit else texts["units"]
    inputs = read_json(x106_inputs_path())
    entry = inputs["models"][tag]
    stage0a = read_json(x106_projections_path(tag))
    if stage0a.get("model_tag") != tag:
        raise SystemExit("FATAL: x106 Stage 0a projections are missing for this tag")
    date = stage0a["rendering"]["date"]
    x106_proj = {r["id"]: float(r["proj"]) for r in stage0a["rows"]}
    unit = sd_unit(tag)
    dir_path = directions_path(tag)
    if sha256_file(dir_path) != entry["directions"]["sha256"]:
        raise SystemExit(f"FATAL: {tag}: direction artifact differs from the frozen x106 inputs")
    mu = load_mu(tag, dir_path)
    minus1 = torch.from_numpy(mu[0].astype(np.float32))
    tbg = torch.from_numpy(mu[1].astype(np.float32))
    _, mu_norms = unit_directions(mu)

    tokenizer = load_tag_tokenizer(entry["snapshot"], tag)
    model = load_tag_model(entry["snapshot"], tag)
    model.eval()
    n_blocks = n_blocks_of(model)
    block = int(entry["hook_block"])
    if entry["n_blocks"] is not None and n_blocks != int(entry["n_blocks"]):
        raise SystemExit(f"FATAL: {tag}: model has {n_blocks} blocks, the inputs record {entry['n_blocks']}")
    if tag != IMPORTED_TAG and hook_block_for(n_blocks) != block:
        raise SystemExit(f"FATAL: {tag}: block {block} is not the x88 rule on {n_blocks} blocks")
    hidden_dim = int(model.config.get_text_config().hidden_size)
    site = hook_site(model, block)
    boundary_ids = [int(v) for v in entry["boundary_ids"]]

    def render(text: str) -> str:
        chat = render_chat(tokenizer, text, date)
        assert_boundary(tokenizer, chat, boundary_ids, tag)
        return chat

    audit = audit_site(model, tokenizer, site, block, minus1, tbg, [render(u["prompt"]) for u in units[:N_AUDIT]])
    print(f"x113 project {tag}: site audit {audit['status']}", flush=True)
    if audit["status"] != "pass":
        atomic_json(paths(tag)["projections"].with_name("x113_site_audit_FAILED.json"), audit)
        raise SystemExit(f"FATAL: {tag}: hook-site audit failed")

    reader = BoundaryReader()
    handle = site.register_forward_hook(reader)
    jobs = []
    for u in units:
        jobs.append((u["id"], "O", u["prompt"]))
        for arm in ARMS:
            if u["texts"].get(arm) is not None:
                jobs.append((u["id"], arm, u["texts"][arm]))
    states = np.empty((len(jobs), 2, hidden_dim), dtype=np.float32)
    for index, (_, _, text) in enumerate(jobs):
        encoded = tokenizer(render(text), return_tensors="pt", add_special_tokens=False).to(model.device)
        calls = reader.calls
        forward(model, encoded)
        if reader.calls != calls + 1 or reader.last.shape != (1, 2, hidden_dim):
            raise SystemExit(f"FATAL: reader fired {reader.calls - calls} times for {jobs[index][:2]}")
        states[index] = reader.last[0].numpy()
        if index % 200 == 0 or index == len(jobs) - 1:
            print(f"x113 project {tag}: {index + 1}/{len(jobs)} texts", flush=True)
    handle.remove()
    proj = projection_of(states, mu)
    per_token = np.einsum("ntd,td->nt", states.astype(np.float64), mu / mu_norms[:, None])

    rows: dict[str, dict] = {}
    for index, (prompt_id, text, _) in enumerate(jobs):
        row = rows.setdefault(prompt_id, {"id": prompt_id, "proj": {}, "proj_sd": {}, "proj_by_token": {}})
        row["proj"][text] = float(proj[index])
        row["proj_sd"][text] = float(proj[index] / unit)
        row["proj_by_token"][text] = {p: float(v) for p, v in zip(BOUNDARY_POSITIONS, per_token[index])}
    o_diff = [abs(rows[u["id"]]["proj"]["O"] - x106_proj[u["id"]]) for u in units if u["id"] in x106_proj]
    out = {
        "experiment": "x113",
        "stage": 5,
        "model_tag": tag,
        "model_id": entry["model_id"],
        "model_source": entry["snapshot"],
        "texts": file_record(paths(tag)["texts"]),
        "texts_sha256": texts["texts_sha256"],
        "site": {"hook_block": block, "state_index": block + 1, "n_blocks": n_blocks, "hidden_dim": hidden_dim, "boundary_ids": boundary_ids, "read": "forward hook on the x88 write-site block; block output at the last two prefill positions, batch size 1"},
        "directions": {**entry["directions"], "kind": entry["directions_kind"]},
        "projection": "sum over the two trailing prefill tokens of <h, mu/|mu|>; proj_sd divides by the model's x106 Hard-1K sd unit",
        "sd_unit": unit,
        "rendering": {"renderer": "x87_common.format_user_chat via x106_common.render_chat", "date": date},
        "site_audit": audit,
        "o_vs_x106_stage0a": {"n": len(o_diff), "max_abs_diff": float(max(o_diff)) if o_diff else None, "max_abs_diff_sd": float(max(o_diff) / unit) if o_diff else None},
        "n_texts": len(jobs),
        "environment": {"torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
        "rows": [rows[u["id"]] for u in units],
    }
    atomic_json(paths(tag)["projections"], out)
    print(f"x113 project {tag}: {len(jobs)} texts -> {paths(tag)['projections']}; O vs x106 max |diff| {out['o_vs_x106_stage0a']['max_abs_diff']}")


if __name__ == "__main__":
    main()
