"""Checks that common/metrics.py gives exactly the same scores as sacreBLEU.

Run:  python -m pytest tests/  (needs sacrebleu; uses train.csv if it is present)
"""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.metrics import corpus_scores, sentence_chrfpp  # noqa: E402

sacrebleu = pytest.importorskip("sacrebleu")
from sacrebleu.metrics import BLEU, CHRF  # noqa: E402

TRAIN_CSV = os.path.join(os.path.dirname(__file__), "..", "train.csv")


def _perturb(text, rng):
    words = text.split()
    ops = rng.choice(["drop", "swap", "dup", "punct", "same", "empty_word"])
    if ops == "drop" and len(words) > 1:
        del words[rng.randrange(len(words))]
    elif ops == "swap" and len(words) > 2:
        i = rng.randrange(len(words) - 1)
        words[i], words[i + 1] = words[i + 1], words[i]
    elif ops == "dup" and words:
        words.insert(rng.randrange(len(words)), rng.choice(words))
    elif ops == "punct" and words:
        words[rng.randrange(len(words))] += rng.choice([",", ".", ")", "!"])
    return " ".join(words)


def _cases():
    rng = random.Random(0)
    base = [
        "Seal of Aššur-idī son of Šu-Ištar, 1.5 minas of silver (tin).",
        "He will pay 10 shekels of silver; if he does not pay, interest accrues.",
        "To Puzur-Aššur from Imdī-ilum: the textiles have arrived.",
        "x x x broken",
    ]
    if os.path.exists(TRAIN_CSV):
        import pandas as pd
        base += pd.read_csv(TRAIN_CSV)["translation"].dropna().sample(300, random_state=0).tolist()
    refs = base
    hyps = [_perturb(r, rng) for r in refs]
    return hyps, refs


def test_corpus_matches_sacrebleu():
    hyps, refs = _cases()
    ours = corpus_scores(hyps, refs)
    assert ours["bleu"] == pytest.approx(BLEU().corpus_score(hyps, [refs]).score, abs=1e-9)
    assert ours["chrfpp"] == pytest.approx(CHRF(word_order=2).corpus_score(hyps, [refs]).score, abs=1e-9)


def test_edge_cases_match_sacrebleu():
    for hyps, refs in [(["", "a"], ["a b", ""]), (["abc"], ["xyz"]), (["a b c d"], ["a b c d"]), (["1-2 3.5"], ["1-2 3,5"])]:
        ours = corpus_scores(hyps, refs)
        assert ours["bleu"] == pytest.approx(BLEU().corpus_score(hyps, [refs]).score, abs=1e-9)
        assert ours["chrfpp"] == pytest.approx(CHRF(word_order=2).corpus_score(hyps, [refs]).score, abs=1e-9)


def test_sentence_chrfpp_matches_sacrebleu():
    hyps, refs = _cases()
    for h, r in list(zip(hyps, refs))[:50]:
        assert sentence_chrfpp(h, r) == pytest.approx(CHRF(word_order=2).sentence_score(h, [r]).score, abs=1e-9)


def test_word_boundary_changes_chrfpp():
    # The old notebook metric ignored word boundaries; chrF++ must not.
    assert corpus_scores(["thesilver"], ["the silver"])["chrfpp"] < 100.0
