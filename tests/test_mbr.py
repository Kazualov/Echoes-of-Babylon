"""MBR selection rule used by tools/mbr_from_predictions.py."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from mbr_from_predictions import mbr_select  # noqa: E402


def test_prefers_the_consensus_candidate():
    assert mbr_select(["completely different words here", "he pays ten shekels", "he pays ten shekels of silver"]) in (1, 2)


def test_skips_empty_candidates():
    assert mbr_select(["", "he pays silver", "  "]) == 1
    assert mbr_select(["", " "]) is None


def test_ties_keep_argument_order():
    assert mbr_select(["same text", "same text", "same text"]) == 0


def test_notebook_uses_the_same_rule():
    import json
    import random

    nb = json.load(open(os.path.join(os.path.dirname(__file__), "..", "ensemble-pipeline.ipynb"), encoding="utf-8"))
    src = next("".join(c["source"]) for c in nb["cells"] if "def mbr_orchestrate" in "".join(c["source"]))
    src = src[:src.index("def orchestrate_all")]
    from common.metrics import sentence_chrfpp
    ns = {"sentence_chrfpp": sentence_chrfpp}
    exec(src, ns)
    rng = random.Random(0)
    vocab = "he will pay ten shekels of silver to the son of Aššur textiles tin".split()
    for _ in range(200):
        cands = [" ".join(rng.choices(vocab, k=rng.choice([0, 1, 5, 12]))) for _ in range(3)]
        winner, _ = ns["mbr_orchestrate"]([{"model": str(i), "text": t, "conf": 0.0} for i, t in enumerate(cands)])
        k = mbr_select(cands)
        assert winner["text"].strip() == ("" if k is None else cands[k].strip())
