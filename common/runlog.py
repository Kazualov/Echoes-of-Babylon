"""Shared run record: one prediction file and one JSON record per trained model.

Every notebook calls `finalize_run(...)` once per model after validation. It
scores the predictions against the full (untruncated) references with the shared
metric and writes, into `results/`:
  * preds_val_<system>__seed<k>.csv  - input to tools/evaluate_all.py
  * run_<system>__seed<k>.json       - hyperparameters, split checksum, metric
                                       signature, final scores and the
                                       per-epoch training log
These files link each row of the report tables to the exact configuration that
produced it.

The same source is pasted into each notebook by `tools/sync_shared_cells.py`;
edit this file, then re-run that script. Do not edit the notebook copies.
"""
import datetime
import json
import os

import pandas as pd

from common.metrics import METRIC_SIGNATURE, corpus_scores
from common.split import val_ids_sha256

RESULTS_DIR = "results"


def finalize_run(system, seed, ids, refs, preds, hyperparameters, log_history=None,
                 out_dir=RESULTS_DIR, notes=None):
    os.makedirs(out_dir, exist_ok=True)
    ids = [str(i) for i in ids]
    refs = ["" if r is None else str(r) for r in refs]
    preds = ["" if p is None else str(p) for p in preds]
    scores = corpus_scores(preds, refs)
    tag = f"{system}__seed{seed}"
    pred_path = os.path.join(out_dir, f"preds_val_{tag}.csv")
    pd.DataFrame({"id": ids, "target_text": refs, "pred_translation": preds}).to_csv(pred_path, index=False)
    record = {
        "system": system,
        "seed": seed,
        "created_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "metric": METRIC_SIGNATURE,
        "n_val_docs": len(ids),
        "val_ids_sha256": val_ids_sha256(ids),
        "final_val": scores,
        "hyperparameters": hyperparameters,
        "notes": notes,
        "log_history": log_history or [],
    }
    with open(os.path.join(out_dir, f"run_{tag}.json"), "w") as f:
        json.dump(record, f, indent=2, default=str, ensure_ascii=False)
    print(f"[run record] {tag}: BLEU={scores['bleu']:.2f} chrF++={scores['chrfpp']:.2f} "
          f"GM={scores['geo_mean']:.2f} -> {pred_path}")
    return scores
