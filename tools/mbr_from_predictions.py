"""MBR ensemble computed from saved validation predictions (no GPU, no weights).

Usage:
    python tools/mbr_from_predictions.py \\
        results/preds_val_byt5__seed42.csv \\
        results/preds_val_mbart50__seed42.csv \\
        results/preds_val_qwen2.5-7b-lora__seed42.csv

Applies the rule of ensemble-pipeline.ipynb: for every validation document the
candidate with the highest mean sentence chrF++ against the other candidates is
kept; all models have an equal vote and ties keep the order of the arguments.
Each member contributes the prediction recorded by its own training notebook,
so members keep their own decoding (mBART and Qwen greedy, ByT5 beam 8).
References are used only to score the result, never to select a candidate.

Writes results/preds_val_mbr3-ensemble__seed<k>.csv and the run record, and
prints how often each member was selected.
"""
import argparse
import os
import re
import sys
from collections import Counter

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from common.metrics import sentence_chrfpp  # noqa: E402
from common.runlog import finalize_run  # noqa: E402
from common.split import EXPECTED_VAL_SHA256, val_ids_sha256  # noqa: E402

NAME_RE = re.compile(r"^preds_val_(?P<system>.+?)__seed(?P<seed>\d+)\.csv$")


def mbr_select(candidates):
    """Index of the candidate with the highest mean chrF++ against the others (empty ones skipped)."""
    valid = [i for i, c in enumerate(candidates) if c.strip()]
    if not valid:
        return None
    best, best_utility = valid[0], -1.0
    for i in valid:
        others = [candidates[j] for j in valid if j != i]
        utility = sum(sentence_chrfpp(candidates[i], o) for o in others) / len(others) if others else 0.0
        if utility > best_utility:
            best, best_utility = i, utility
    return best


def load(path):
    df = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
    if val_ids_sha256(df["id"]) != EXPECTED_VAL_SHA256 or df["id"].duplicated().any():
        raise SystemExit(f"{path}: not the shared validation split")
    return df.set_index("id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="one preds_val_<system>__seed<k>.csv per member")
    ap.add_argument("--system", default="mbr3-ensemble")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    parsed = []
    for p in args.files:
        m = NAME_RE.match(os.path.basename(p))
        if not m:
            raise SystemExit(f"{p}: expected preds_val_<system>__seed<k>.csv")
        parsed.append(m)
    members = [m["system"] for m in parsed]
    seeds = {int(m["seed"]) for m in parsed}
    if len(members) < 3 or len(set(members)) != len(members):
        raise SystemExit("Need at least three different members: with two candidates every choice ties.")
    if len(seeds) != 1:
        raise SystemExit(f"Members must share one seed, got {sorted(seeds)}")

    frames = [load(p) for p in args.files]
    ids = list(frames[0].index)
    for p, f in zip(args.files, frames):
        refs = f.loc[ids, "target_text"].str.strip().tolist()
        if refs != frames[0]["target_text"].str.strip().tolist():
            raise SystemExit(f"{p}: references differ from {args.files[0]}")

    preds, wins = [], Counter()
    for doc in ids:
        cands = [f.at[doc, "pred_translation"] for f in frames]
        k = mbr_select(cands)
        preds.append("" if k is None else cands[k])
        wins[members[k] if k is not None else "none"] += 1
    finalize_run(
        system=args.system, seed=seeds.pop(), ids=ids, refs=frames[0]["target_text"].tolist(), preds=preds,
        hyperparameters={"members": members, "utility": "mean sentence chrF++ against the other candidates",
                         "votes": "equal", "selected_per_member": dict(wins)},
        out_dir=args.out,
        notes="computed offline by tools/mbr_from_predictions.py from the recorded member predictions",
    )
    print("selected per member:", dict(wins))


if __name__ == "__main__":
    main()
