"""x116 Stage 3 (GPU, one model per job): the projection of O and R_rand on the model's direction
at x106's read site, through x113_project.py's path (site audit, BoundaryReader, sum over the two
trailing prefill positions, the model's x106 Hard-1K sd unit). O is re-read so that every R_rand
change is taken against an O read in the same job; it is checked against x113's O projection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x103_extract import audit_site  # noqa: E402
from x106_common import BoundaryReader, assert_boundary, directions_path, load_mu, load_tag_model, load_tag_tokenizer, hook_site, projection_of, sha256_file  # noqa: E402
from x113_common import render_chat, sd_unit, x106_inputs_path, x106_projections_path  # noqa: E402
from x113_common import paths as x113_paths  # noqa: E402
from x116_common import ALL_TAGS, atomic_json, file_record, load_texts, paths, read_json  # noqa: E402


@torch.inference_mode()
def forward(model, encoded):
    return model(**encoded, use_cache=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, choices=ALL_TAGS)
    args = parser.parse_args()
    tag = args.tag
    texts = load_texts(tag)
    x113_proj = {r["id"]: r for r in read_json(x113_paths(tag)["projections"])["rows"]}
    prompts = {u["id"]: u for u in read_json(x113_paths(tag)["texts"])["units"]}
    entry = read_json(x106_inputs_path())["models"][tag]
    date = read_json(x106_projections_path(tag))["rendering"]["date"]
    unit = sd_unit(tag)
    dir_path = directions_path(tag)
    if sha256_file(dir_path) != entry["directions"]["sha256"]:
        raise SystemExit(f"FATAL: {tag}: direction artifact differs from the frozen x106 inputs")
    mu = load_mu(tag, dir_path)
    minus1 = torch.from_numpy(mu[0].astype(np.float32))
    tbg = torch.from_numpy(mu[1].astype(np.float32))
    tokenizer = load_tag_tokenizer(entry["snapshot"], tag)
    model = load_tag_model(entry["snapshot"], tag)
    model.eval()
    block = int(entry["hook_block"])
    hidden_dim = int(model.config.get_text_config().hidden_size)
    site = hook_site(model, block)
    boundary_ids = [int(v) for v in entry["boundary_ids"]]

    def render(text: str) -> str:
        chat = render_chat(tokenizer, text, date)
        assert_boundary(tokenizer, chat, boundary_ids, tag)
        return chat

    units = [u for u in texts["units"] if u["text"] is not None]
    audit = audit_site(model, tokenizer, site, block, minus1, tbg, [render(prompts[u["id"]]["prompt"]) for u in units[:2]])
    if audit["status"] != "pass":
        raise SystemExit(f"FATAL: {tag}: hook-site audit failed")
    reader = BoundaryReader()
    handle = site.register_forward_hook(reader)
    jobs = []
    for u in units:
        jobs.append((u["id"], "O", prompts[u["id"]]["prompt"]))
        jobs.append((u["id"], "R_rand", u["text"]))
    states = np.empty((len(jobs), 2, hidden_dim), dtype=np.float32)
    for index, (_, _, text) in enumerate(jobs):
        encoded = tokenizer(render(text), return_tensors="pt", add_special_tokens=False).to(model.device)
        calls = reader.calls
        forward(model, encoded)
        if reader.calls != calls + 1 or reader.last.shape != (1, 2, hidden_dim):
            raise SystemExit("FATAL: reader misfired")
        states[index] = reader.last[0].numpy()
    handle.remove()
    proj = projection_of(states, mu)
    rows: dict[str, dict] = {}
    for index, (pid, label, _) in enumerate(jobs):
        row = rows.setdefault(pid, {"id": pid, "proj": {}, "proj_sd": {}})
        row["proj"][label] = float(proj[index])
        row["proj_sd"][label] = float(proj[index] / unit)
    o_diff = [abs(rows[i]["proj_sd"]["O"] - x113_proj[i]["proj_sd"]["O"]) for i in rows if i in x113_proj]
    out = {
        "experiment": "x116",
        "stage": 3,
        "model_tag": tag,
        "texts": file_record(paths(tag)["texts"]),
        "texts_sha256": texts["texts_sha256"],
        "site": {"hook_block": block, "boundary_ids": boundary_ids, "read": "x113_project.py's read"},
        "sd_unit": unit,
        "site_audit": audit,
        "o_vs_x113_max_abs_diff_sd": float(max(o_diff)) if o_diff else None,
        "rows": [rows[u["id"]] for u in units],
    }
    atomic_json(paths(tag)["projections"], out)
    print(f"x116 project {tag}: {len(jobs)} texts; O vs x113 max |diff| {out['o_vs_x113_max_abs_diff_sd']} sd")


if __name__ == "__main__":
    main()
