"""x102 stage 2: fetch XSTest's 2,250 human-labelled completions at a pinned commit.

Five completion files (one per model, 450 prompts each) with the human
`final_label`, joined with the repository's released GPT-4 and string-match
labels for the same rows.  Every raw file's sha256 is recorded.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import urllib.request

from x102_common import (
    MANIFEST_PATH,
    N_XSTEST,
    XSTEST_COMMIT,
    XSTEST_HUMAN_LABELS,
    XSTEST_MODELS,
    XSTEST_PATH,
    XSTEST_REPO,
    atomic_json,
    file_record,
    keys_sha256,
    read_json,
    sha256_bytes,
    xstest_key,
)

RAW = "https://raw.githubusercontent.com/paul-rottger/xstest/" + XSTEST_COMMIT
COMPLETIONS = RAW + "/model_completions/xstest_v2_completions_{model}.csv"
AUTOMATED = RAW + "/evaluation/automated_evaluation_labels/xstest_v2_completions_{model}_{kind}.csv"
PUBLISHED_LABELS = ("1_full_compliance", "2_full_refusal", "3_partial_refusal")
# Eleven of the released GPT-4 verdicts (all on mistralinstruct) are free text
# instead of a label; they are kept as `malformed` with the raw text next to them.


def fetch(url: str) -> tuple[list[dict[str, str]], str]:
    with urllib.request.urlopen(url, timeout=120) as handle:
        payload = handle.read()
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
    return rows, sha256_bytes(payload)


def main() -> None:
    rows = []
    files = {}
    for model in XSTEST_MODELS:
        completions, sha = fetch(COMPLETIONS.format(model=model))
        files[f"completions_{model}"] = {"url": COMPLETIONS.format(model=model), "sha256": sha, "rows": len(completions)}
        automated = {}
        for kind, column in (("gpteval", "gpt4_label"), ("streval", "strmatch_label")):
            labelled, sha_auto = fetch(AUTOMATED.format(model=model, kind=kind))
            files[f"{kind}_{model}"] = {"url": AUTOMATED.format(model=model, kind=kind), "sha256": sha_auto, "rows": len(labelled)}
            automated[column] = {row["id"]: row for row in labelled}
            if len(automated[column]) != len(labelled):
                raise SystemExit(f"FATAL: duplicate ids in {kind}/{model}")
        if len(completions) != 450:
            raise SystemExit(f"FATAL: {model} has {len(completions)} completions, expected 450")
        for row in completions:
            if row["final_label"] not in XSTEST_HUMAN_LABELS:
                raise SystemExit(f"FATAL: unknown human label {row['final_label']!r} in {model}")
            published = {}
            for column, table in automated.items():
                auto = table.get(row["id"])
                if auto is None or auto["prompt"] != row["prompt"] or auto["completion"] != row["completion"]:
                    raise SystemExit(f"FATAL: {column} row for {model}/{row['id']} does not match the completion file")
                if auto["final_label"] != row["final_label"]:
                    raise SystemExit(f"FATAL: {column} human label for {model}/{row['id']} is inconsistent")
                if auto[column] in PUBLISHED_LABELS:
                    published[column] = XSTEST_HUMAN_LABELS[auto[column]]
                elif column == "gpt4_label":
                    published[column] = "malformed"
                    published["gpt4_raw"] = auto[column]
                else:
                    raise SystemExit(f"FATAL: unknown {column} {auto[column]!r} for {model}/{row['id']}")
            rows.append(
                {
                    "key": xstest_key(model, row["id"]),
                    "model": model,
                    "id": row["id"],
                    "type": row["type"],
                    "prompt": row["prompt"],
                    "response": row["completion"],
                    "annotation_1": XSTEST_HUMAN_LABELS[row["annotation_1"]],
                    "annotation_2": XSTEST_HUMAN_LABELS[row["annotation_2"]],
                    "agreement": row["agreement"].strip().lower() == "true",
                    "human_label": XSTEST_HUMAN_LABELS[row["final_label"]],
                    "gpt4_label": published["gpt4_label"],
                    "gpt4_raw": published.get("gpt4_raw"),
                    "strmatch_label": published["strmatch_label"],
                }
            )
        print(f"fetched {model}: {len(completions)} rows", flush=True)
    if len(rows) != N_XSTEST:
        raise SystemExit(f"FATAL: {len(rows)} XSTest rows, expected {N_XSTEST}")
    keys = [row["key"] for row in rows]
    xstest_sha = keys_sha256(keys)
    atomic_json(
        XSTEST_PATH,
        {
            "schema_version": 1,
            "experiment": "x102",
            "fetched": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "repo": XSTEST_REPO,
            "commit": XSTEST_COMMIT,
            "files": files,
            "xstest_sha256": xstest_sha,
            "rows": rows,
        },
    )
    manifest = read_json(MANIFEST_PATH) if MANIFEST_PATH.exists() else {}
    manifest["xstest_sha256"] = xstest_sha
    manifest["xstest"] = {"repo": XSTEST_REPO, "commit": XSTEST_COMMIT, "files": files} | file_record(XSTEST_PATH)
    atomic_json(MANIFEST_PATH, manifest)
    counts = {}
    for row in rows:
        counts[row["human_label"]] = counts.get(row["human_label"], 0) + 1
    print(f"x102 xstest: {len(rows)} rows, sha256 {xstest_sha[:12]}, human labels {counts}")


if __name__ == "__main__":
    main()
