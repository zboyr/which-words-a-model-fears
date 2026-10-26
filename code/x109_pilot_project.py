"""Benign insertion pilot, projection pass (GPU, Qwen3.5-4B).

For each of the 20 prompts: the original text (O) and the text with one word appended at the
end, for every x83 lexicon word, every x85 neutral and every twin candidate. Each text is
rendered with x106's renderer (Qwen3.5-4B: no date), read at the two trailing prefill
positions of block 29 through x106's BoundaryReader, one prompt per forward (x106 Stage 0a),
after x106's read-site audit, and projected on the x87 consensus direction. Two Hard-1K
prompts are re-read and compared with x106 Stage 0a to fix the scale.
shift_w = proj(E_w) - proj(O), per boundary token and summed.
"""

from __future__ import annotations

import importlib.metadata
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x109_pilot_common import (  # noqa: E402
    PROJECTIONS,
    PROMPTS,
    TAG,
    TWIN_CANDIDATES,
    X106_HARD1K_SD,
    append_end,
    atomic_json,
    read_json,
    sha256_text,
)
from x103_extract import audit_site  # noqa: E402
from x106_common import (  # noqa: E402
    BOUNDARY_POSITIONS,
    BoundaryReader,
    assert_boundary,
    directions_path,
    file_record,
    hard1k_rows,
    hook_site,
    inputs_path,
    load_mu,
    load_tag_model,
    load_tag_tokenizer,
    n_blocks_of,
    projection_of,
    projections_path,
    render_chat,
    sha256_file,
    unit_directions,
)
from x83_common import lexicon_words  # noqa: E402
from x85_common import WORD_PAIRS  # noqa: E402


@torch.inference_mode()
def read_state(model, tokenizer, reader, chat: str, hidden: int) -> tuple[np.ndarray, int]:
    encoded = tokenizer(chat, return_tensors="pt", add_special_tokens=False).to(model.device)
    calls = reader.calls
    model(**encoded, use_cache=False)
    if reader.calls != calls + 1 or reader.last.shape != (1, 2, hidden):
        raise SystemExit("FATAL: reader did not fire exactly once")
    return reader.last[0].numpy(), int(encoded["input_ids"].shape[1])


def main() -> None:
    inputs = read_json(inputs_path())
    entry = inputs["models"][TAG]
    prompts = read_json(PROMPTS)
    lex = [w for w, _ in lexicon_words()]
    neutrals = [n for _, _, n in WORD_PAIRS]
    words = list(dict.fromkeys(lex + neutrals + list(TWIN_CANDIDATES)))
    if sha256_file(directions_path(TAG)) != entry["directions"]["sha256"]:
        raise SystemExit("FATAL: direction artifact differs from the x106 inputs")
    mu = load_mu(TAG)
    units, norms = unit_directions(mu)
    tokenizer = load_tag_tokenizer(entry["snapshot"], TAG)
    model = load_tag_model(entry["snapshot"], TAG)
    model.eval()
    block = int(entry["hook_block"])
    hidden = int(model.config.get_text_config().hidden_size)
    site = hook_site(model, block)
    boundary_ids = [int(v) for v in entry["boundary_ids"]]
    stage0a = read_json(projections_path(TAG))
    date = stage0a["rendering"]["date"]

    def render(text: str) -> str:
        chat = render_chat(tokenizer, text, date)
        assert_boundary(tokenizer, chat, boundary_ids, TAG)
        return chat

    audit = audit_site(model, tokenizer, site, block, torch.from_numpy(mu[0].astype(np.float32)), torch.from_numpy(mu[1].astype(np.float32)), [render(p["text"]) for p in prompts["prompts"][:2]])
    print(f"site audit {audit['status']}", flush=True)
    if audit["status"] != "pass":
        raise SystemExit("FATAL: read-site audit failed")
    reader = BoundaryReader()
    handle = site.register_forward_hook(reader)

    # Scale check against x106 Stage 0a.
    s0 = {r["id"]: r["proj"] for r in stage0a["rows"]}
    scale = []
    for row in hard1k_rows(TAG)[:2]:
        chat = render(row["prompt"])
        if sha256_text(chat) != row["rendered_chat_sha256"]:
            raise SystemExit("FATAL: Hard-1K rendering does not reproduce the base row")
        state, _ = read_state(model, tokenizer, reader, chat, hidden)
        value = float(projection_of(state[None], mu)[0])
        scale.append({"id": row["id"], "proj_here": value, "proj_x106_stage0a": s0[row["id"]], "abs_diff": abs(value - s0[row["id"]])})
    print("scale check", scale, flush=True)

    t0 = time.time()
    n_p, n_w = len(prompts["prompts"]), len(words)
    o_tok = np.empty((n_p, 2))
    e_tok = np.empty((n_p, n_w, 2))
    o_len = []
    e_len = np.empty((n_p, n_w), dtype=int)
    for i, p in enumerate(prompts["prompts"]):
        state, length = read_state(model, tokenizer, reader, render(p["text"]), hidden)
        o_tok[i] = np.einsum("td,td->t", state.astype(np.float64), units)
        o_len.append(length)
        for j, w in enumerate(words):
            state, length = read_state(model, tokenizer, reader, render(append_end(p["text"], w)), hidden)
            e_tok[i, j] = np.einsum("td,td->t", state.astype(np.float64), units)
            e_len[i, j] = length
        print(f"prompt {i + 1}/{n_p}, {(time.time() - t0) / 60:.1f} min", flush=True)
    handle.remove()

    shift_tok = e_tok - o_tok[:, None, :]
    shift = shift_tok.sum(axis=2)
    o_proj = o_tok.sum(axis=1)
    pids = [p["id"] for p in prompts["prompts"]]
    summary = {}
    for j, w in enumerate(words):
        summary[w] = {
            "set": "lexicon" if w in lex else ("x85_neutral" if w in neutrals else "twin_candidate"),
            "mean_shift": float(shift[:, j].mean()),
            "mean_shift_sd_units": float(shift[:, j].mean() / X106_HARD1K_SD),
            "mean_shift_tbg_minus1": float(shift_tok[:, j, 0].mean()),
            "mean_shift_tbg": float(shift_tok[:, j, 1].mean()),
            "mean_shift_token_mean": float(shift_tok[:, j, :].mean()),
            "n_added_tokens_mean": float((e_len[:, j] - np.asarray(o_len)).mean()),
        }
    lex_idx = [words.index(w) for w in lex]
    argmax = {pid: lex[int(np.argmax(shift[i, lex_idx]))] for i, pid in enumerate(pids)}
    out = {
        "pilot": "x109/pilot",
        "model_tag": TAG,
        "model_id": entry["model_id"],
        "model_source": entry["snapshot"],
        "prompts_sha256": prompts["prompts_sha256"],
        "site": {"hook_block": block, "state_index": block + 1, "n_blocks": n_blocks_of(model), "hidden_dim": hidden, "boundary_ids": boundary_ids, "batch_size": 1, "read": "x106 Stage 0a: block output at the last two prefill positions"},
        "directions": {**file_record(directions_path(TAG)), "kind": entry["directions_kind"], "norms": {p: float(v) for p, v in zip(BOUNDARY_POSITIONS, norms)}},
        "rendering": {"renderer": "x106_common.render_chat", "date": date},
        "site_audit": audit,
        "scale_check_vs_x106_stage0a": scale,
        "sd_unit": {"value": X106_HARD1K_SD, "source": "population sd of the x106 Stage 0a projections over the 1,319 Hard-1K prompts, Qwen3.5-4B"},
        "projection": "sum over the two boundary tokens of <h, mu/|mu|>; shift = proj(text + ' ' + word) - proj(text)",
        "boundary_positions": list(BOUNDARY_POSITIONS),
        "prompt_ids": pids,
        "words": words,
        "o_proj": o_proj.tolist(),
        "o_proj_by_token": o_tok.tolist(),
        "o_n_tokens": o_len,
        "shift": shift.tolist(),
        "shift_by_token": shift_tok.tolist(),
        "e_n_tokens": e_len.tolist(),
        "word_summary": summary,
        "argmax_lexicon_word_by_prompt": argmax,
        "environment": {"torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
        "seconds": time.time() - t0,
    }
    atomic_json(PROJECTIONS, out)
    print(f"-> {PROJECTIONS}")


if __name__ == "__main__":
    main()
