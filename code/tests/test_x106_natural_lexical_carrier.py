"""Contract tests for x106: the arm table, the seeds, the deletion variants, the nomination
arithmetic, the cohort hash and the batched boundary reader."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x106_common import (  # noqa: E402
    ALL_TAGS,
    IMPORTED_TAG,
    TEXT_CONDITIONS,
    BatchBoundaryReader,
    arm_names,
    arm_spec,
    cohort_sha256,
    delete_span,
    deletion_variants,
    dose_label,
    doses_for,
    is_content_word,
    nominate,
    normalize_word,
    sha256_text,
    stable_seed,
    strata_of,
)


def test_arm_table():
    assert doses_for(IMPORTED_TAG) == (-0.5, -1.5)
    assert all(doses_for(tag) == (-1.5,) for tag in ALL_TAGS if tag != IMPORTED_TAG)
    assert arm_names(IMPORTED_TAG) == ("O_base", "S_base", "C_base", "R_base", "O_m050", "S_m050", "O_m150", "S_m150")
    assert arm_names("llama") == ("O_base", "S_base", "C_base", "R_base", "O_m150", "S_m150")
    assert arm_spec(IMPORTED_TAG, "S_m050") == {"arm": "S_m050", "text": "S", "dose": -0.5}
    assert dose_label(0.0) == "base" and dose_label(-1.5) == "m150" and dose_label(1.0) == "p100"
    for tag in ALL_TAGS:
        assert [spec for spec in [arm_spec(tag, a) for a in arm_names(tag)] if spec["dose"] == 0.0][0]["text"] == "O"


def test_seed_namespace_and_common_random_numbers():
    s = stable_seed("qwen", "orbench_hard:00000:abc", "O")
    assert 0 <= s < 2**31
    assert s == stable_seed("qwen", "orbench_hard:00000:abc", "O")  # every dose of one text shares it
    assert s != stable_seed("qwen", "orbench_hard:00000:abc", "S")
    assert s != stable_seed("llama", "orbench_hard:00000:abc", "O")
    assert s != stable_seed("qwen", "orbench_hard:00000:abc", "O", retry=True)


def test_words_and_deletions():
    text = "How do I kill  a Python process, quickly?"
    variants = deletion_variants(text)
    assert variants["words"] == ["How", "do", "I", "kill", "a", "Python", "process,", "quickly?"]
    assert variants["unigram_texts"][3] == "How do I  a Python process, quickly?"  # the preceding run goes, the other stays
    assert variants["unigram_texts"][0] == "do I kill  a Python process, quickly?"
    assert variants["unigram_texts"][-1] == "How do I kill  a Python process,"
    assert variants["bigram_texts"][3] == "How do I Python process, quickly?"
    assert len(variants["bigram_texts"]) == len(variants["words"]) - 1
    assert delete_span("a b", 0, 1) == "b" and delete_span("a b", 2, 3) == "a"
    assert normalize_word("process,") == "process" and normalize_word("'Quickly?'") == "quickly"
    assert is_content_word("kill") and not is_content_word("the") and not is_content_word("...")


def test_nomination_arithmetic():
    words = ["How", "to", "build", "a", "questionable", "device", "without", "getting", "caught"]
    d = [0.1, 0.0, 1.0, 0.2, 6.0, 0.5, 2.0, 1.5, 2.5]
    b = [0.1, 1.0, 1.1, 6.1, 6.4, 2.4, 3.4, 9.0]  # "getting caught" beats "questionable" by > one sd
    rec = nominate(words, d, b, proj_full=22.0, answered_mean=10.0)
    assert rec["top_word"] == "questionable" and rec["top_delta"] == 6.0
    assert abs(rec["top_share_of_excess"] - 0.5) < 1e-12
    others = np.delete(np.asarray(d), 4)
    assert abs(rec["top_z"] - (6.0 - others.mean()) / others.std()) < 1e-12
    assert rec["pair_wins"] and rec["unit"] == "getting caught" and rec["unit_indices"] == [7, 8]
    assert all(c["index"] not in (7, 8) for c in rec["content_word_z"])
    b2 = [v if v < 6.5 else 6.5 for v in b]
    rec2 = nominate(words, d, b2, 22.0, 10.0)
    assert not rec2["pair_wins"] and rec2["unit"] == "questionable" and rec2["unit_indices"] == [4]
    # a stop word with the largest delta is never the unit
    rec3 = nominate(["the", "device"], [9.0, 1.0], [5.0], 30.0, 10.0)
    assert rec3["unit"] == "device"
    # a pair of two stop words never becomes the unit, however large its delta
    rec4 = nominate(["how", "can", "I", "build", "it"], [0.1, 0.2, 0.1, 1.0, 0.1], [9.0, 0.3, 0.5, 0.4], 30.0, 10.0)
    assert rec4["unit"] == "build" and rec4["top_pair"] != "how can"


def test_strata_and_cohort_hash():
    assert strata_of([14.9, 15.0, 25.0, 25.1]) == ["low", "mid", "mid", "high"]
    unit = {"id": "a", "unit": "kill", "prompt_sha256": {t: sha256_text(t) for t in TEXT_CONDITIONS}}
    partial = {"id": "a", "unit": "kill", "prompt_sha256": {"O": sha256_text("O"), "S": None, "C": sha256_text("C"), "R": None}}
    assert cohort_sha256([unit]) != cohort_sha256([partial])
    assert cohort_sha256([unit]) == cohort_sha256([dict(unit)])


def test_batch_reader_gathers_each_rows_last_two_real_tokens():
    reader = BatchBoundaryReader()
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    reader.lengths = [5, 3]
    out = reader(None, None, (hidden,))
    assert torch.equal(out[0], hidden)
    assert torch.equal(reader.last[0], hidden[0, 3:5])
    assert torch.equal(reader.last[1], hidden[1, 1:3])
    assert reader.calls == 1
