"""Vocabulary-scan benchmark: dump the wordfreq top-50,000 English list (local, CPU)."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import wordfreq

OUT = Path(__file__).resolve().parents[1] / "data" / "runs" / "x109" / "bench" / "wordfreq_en_top50000.json"


def main() -> None:
    words = wordfreq.top_n_list("en", 50000)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"source": "wordfreq.top_n_list('en', 50000)", "wordfreq": importlib.metadata.version("wordfreq"), "words": words}, ensure_ascii=False))
    print(len(words), "->", OUT)


if __name__ == "__main__":
    main()
