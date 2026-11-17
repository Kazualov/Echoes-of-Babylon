"""Score every validation prediction file with one metric on one split.

Usage:
    python tools/evaluate_all.py results/preds/*.csv --out results/

Each input CSV is a `preds_val_*.csv` written by a notebook, with columns
`id` (or `oare_id`), `pred_translation` and, optionally, `target_text`.
File names follow `preds_val_<system>__seed<k>.csv`; files without a seed
suffix count as a single run of <system>.

The script
  1. checks that every file covers exactly the shared validation documents
     (data/val_ids.csv), so no system is scored on a different set;
  2. scores each file with common/metrics.py (sacreBLEU-identical BLEU, chrF++);
  3. reports mean +- std over seeds, a 95% bootstrap interval over documents,
     and a paired bootstrap test of every system against the best one.
Outputs: <out>/per_run.csv, <out>/summary.csv, <out>/paired_tests.csv, <out>/summary.md
"""
import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from common.metrics import METRIC_SIGNATURE, scores_from_segment_stats, segment_stats  # noqa: E402

SEED_RE = re.compile(r"^preds_val_(?P<system>.+?)(?:__seed(?P<seed>\d+))?\.csv$")


def load_predictions(path, val_ids, refs_by_id):
    df = pd.read_csv(path)
    id_col = "oare_id" if "oare_id" in df.columns else "id"
    df[id_col] = df[id_col].astype(str)
    ids = set(df[id_col])
    if ids != val_ids or len(df) != len(val_ids):
        missing, extra = len(val_ids - ids), len(ids - val_ids)
        raise SystemExit(f"{path}: not the shared validation split "
                         f"({len(df)} rows, {missing} shared ids missing, {extra} foreign ids). "
                         "Re-generate it with the shared split cell.")
    df = df.set_index(id_col).loc[sorted(val_ids)]
    preds = df["pred_translation"].fillna("").astype(str).tolist()
    refs = [refs_by_id[i] for i in sorted(val_ids)]
    if "target_text" in df.columns:
        mismatched = sum(a.strip() != str(b).strip() for a, b in zip(refs, df["target_text"].fillna("")))
        if mismatched:
            print(f"WARNING {path}: {mismatched} target_text values differ from train.csv; using train.csv")
    return preds, refs


def bootstrap(bleu_rows, chrf_rows, idx_samples):
    b, c = np.asarray(bleu_rows), np.asarray(chrf_rows)
    out = []
    for idx in idx_samples:
        s = scores_from_segment_stats(b[idx].tolist(), c[idx].tolist())
        out.append([s["bleu"], s["chrfpp"], s["geo_mean"]])
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--train-csv", default=os.path.join(ROOT, "train.csv"))
    ap.add_argument("--val-ids", default=os.path.join(ROOT, "data", "val_ids.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    val_ids = set(pd.read_csv(args.val_ids)["oare_id"].astype(str))
    train = pd.read_csv(args.train_csv)
    refs_by_id = dict(zip(train["oare_id"].astype(str), train["translation"].astype(str)))
    rng = np.random.default_rng(args.seed)
    n = len(val_ids)
    idx_samples = [rng.integers(0, n, n) for _ in range(args.n_boot)]  # same resamples for every system

    runs, boots = [], defaultdict(list)
    for path in sorted(args.files):
        m = SEED_RE.match(os.path.basename(path))
        if not m:
            raise SystemExit(f"{path}: expected name preds_val_<system>[__seed<k>].csv")
        system, seed = m["system"], int(m["seed"]) if m["seed"] else None
        preds, refs = load_predictions(path, val_ids, refs_by_id)
        b_rows, c_rows = segment_stats(preds, refs)
        s = scores_from_segment_stats(b_rows, c_rows)
        runs.append({"system": system, "seed": seed, "file": os.path.basename(path), **s})
        boots[system].append(bootstrap(b_rows, c_rows, idx_samples))
        print(f"{system:28s} seed={seed}  BLEU={s['bleu']:6.2f}  chrF++={s['chrfpp']:6.2f}  GM={s['geo_mean']:6.2f}")

    per_run = pd.DataFrame(runs)
    rows = []
    for system, g in per_run.groupby("system", sort=False):
        # Bootstrap distribution of the seed-averaged score: averages over seeds, resamples documents.
        dist = np.mean(boots[system], axis=0)
        lo, hi = np.percentile(dist, [2.5, 97.5], axis=0)
        rows.append({
            "system": system, "n_seeds": len(g),
            "bleu_mean": g.bleu.mean(), "bleu_std": g.bleu.std(ddof=1) if len(g) > 1 else np.nan,
            "chrfpp_mean": g.chrfpp.mean(), "chrfpp_std": g.chrfpp.std(ddof=1) if len(g) > 1 else np.nan,
            "gm_mean": g.geo_mean.mean(), "gm_std": g.geo_mean.std(ddof=1) if len(g) > 1 else np.nan,
            "gm_ci95_low": lo[2], "gm_ci95_high": hi[2],
        })
    summary = pd.DataFrame(rows).sort_values("gm_mean", ascending=False).reset_index(drop=True)

    best = summary.system.iloc[0]
    best_dist = np.mean(boots[best], axis=0)[:, 2]
    tests = []
    for system in summary.system.iloc[1:]:
        diff = best_dist - np.mean(boots[system], axis=0)[:, 2]
        tests.append({"best": best, "other": system, "gm_diff_mean": diff.mean(),
                      "diff_ci95_low": np.percentile(diff, 2.5), "diff_ci95_high": np.percentile(diff, 97.5),
                      "p_value_best_not_better": float((diff <= 0).mean())})
    tests = pd.DataFrame(tests)

    os.makedirs(args.out, exist_ok=True)
    per_run.to_csv(os.path.join(args.out, "per_run.csv"), index=False)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    tests.to_csv(os.path.join(args.out, "paired_tests.csv"), index=False)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(f"Validation: {n} shared documents (data/val_ids.csv). Metric: {METRIC_SIGNATURE}. "
                f"Bootstrap: {args.n_boot} document resamples, seed {args.seed}.\n\n")
        f.write("| System | Seeds | BLEU | chrF++ | GM (mean ± std) | GM 95% CI |\n|---|---:|---:|---:|---:|---:|\n")
        for r in summary.itertuples():
            pm = "" if np.isnan(r.gm_std) else f" ± {r.gm_std:.2f}"
            f.write(f"| {r.system} | {r.n_seeds} | {r.bleu_mean:.2f} | {r.chrfpp_mean:.2f} | "
                    f"{r.gm_mean:.2f}{pm} | [{r.gm_ci95_low:.2f}, {r.gm_ci95_high:.2f}] |\n")
        if len(tests):
            f.write(f"\nPaired bootstrap vs. best system ({best}):\n\n"
                    "| Other | GM diff | 95% CI | p (best not better) |\n|---|---:|---:|---:|\n")
            for t in tests.itertuples():
                f.write(f"| {t.other} | {t.gm_diff_mean:.2f} | [{t.diff_ci95_low:.2f}, {t.diff_ci95_high:.2f}] | "
                        f"{t.p_value_best_not_better:.3f} |\n")
    print(f"\nWrote {args.out}/summary.md")


if __name__ == "__main__":
    main()
