"""Build the report material from the run records in results/.

Usage:
    python tools/analyze_runs.py --results results/

Reads results/run_*.json and results/preds_val_*.csv (written by finalize_run in
the notebooks) and writes:
  results/experiment_registry.md / .csv  - one row per run: notebook, seed, key
                                           hyperparameters, best checkpoint, scores
  results/figures/curves_<system>.png    - training loss and validation score per epoch
  results/error_analysis.md              - scores by document length and damage,
                                           numbers/names recall, length ratio,
                                           empty and repetitive outputs, worst examples
"""
import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from common.metrics import corpus_scores, sentence_chrfpp  # noqa: E402
from common.split import EXPECTED_VAL_SHA256  # noqa: E402

NOTEBOOK_OF = [("ensemble-member-", "ensemble-pipeline.ipynb"), ("mbr3-ensemble", "tools/mbr_from_predictions.py"),
               ("mbart", "mbart50.ipynb"), ("byt5", "byte-level-seq2seq.ipynb"),
               ("qwen", "pipeline_QwenQLoRa.ipynb")]
KEY_HPARAMS = ["learning_rate", "epochs", "num_train_epochs", "batch_size", "per_device_train_batch_size",
               "grad_accum", "gradient_accumulation_steps", "effective_batch_size", "lora_r", "max_length",
               "max_source_length", "max_source_len", "max_target_length", "max_target_len", "max_new_tokens",
               "train_rows", "members"]
ENV_OF = {"mbart50.ipynb": "mbart50", "byte-level-seq2seq.ipynb": "byt5", "pipeline_QwenQLoRa.ipynb": "qwen"}
DAMAGE_RE = re.compile(r"<gap>|\bx\b|\[|…")
NUMBER_RE = re.compile(r"\d+(?:[.,/]\d+)?")
NAME_RE = re.compile(r"\b[A-ZŠṢṬḪ][\w\-ʾ’]+")


def notebook_for(system):
    for prefix, nb in NOTEBOOK_OF:
        if system.startswith(prefix):
            return nb
    return "?"


def load_runs(results):
    runs = []
    for path in sorted(glob.glob(os.path.join(results, "run_*.json"))):
        r = json.load(open(path))
        r["_file"] = os.path.basename(path)
        runs.append(r)
    if not runs:
        raise SystemExit(f"No run_*.json in {results}. Copy the notebooks' results/ folders here first.")
    return runs


def best_epoch(log_history):
    evals = [e for e in log_history if any(k.startswith("eval_") for k in e) and "epoch" in e]
    key = next((k for k in ("eval_geo_mean", "eval_geom_mean") if any(k in e for e in evals)), None)
    if key:
        best = max((e for e in evals if key in e), key=lambda e: e[key])
        return best.get("epoch"), f"{key}={best[key]:.2f}"
    if any("eval_loss" in e for e in evals):
        best = min((e for e in evals if "eval_loss" in e), key=lambda e: e["eval_loss"])
        return best.get("epoch"), f"eval_loss={best['eval_loss']:.4f}"
    return None, ""


def batch_info(r, results):
    """GPUs (env_*.json), steps per epoch (log) and effective batch, cross-checked against each other."""
    env = os.path.join(results, f"env_{ENV_OF.get(notebook_for(r['system']), '-')}.json")
    steps = next((e["step"] for e in r.get("log_history") or [] if e.get("epoch") == 1.0 and "step" in e), None)
    if steps is None or not os.path.exists(env):
        return None, None, None
    hp = r["hyperparameters"]
    gpus = len(json.load(open(env))["gpu"] or [None])
    batch = hp.get("per_device_train_batch_size", hp.get("batch_size"))
    accum = hp.get("gradient_accumulation_steps", hp.get("grad_accum"))
    rows = hp.get("train_rows", 1405)
    expected = math.ceil(math.ceil(rows / (batch * gpus)) / accum)
    if expected != steps:
        raise SystemExit(f"{r['system']}: {steps} steps per epoch in the log, {expected} expected for {gpus} GPUs")
    return gpus, steps, batch * accum * gpus


def registry(runs, results):
    rows = []
    for r in runs:
        hp = r.get("hyperparameters") or {}
        ep, crit = best_epoch(r.get("log_history") or [])
        gpus, steps, eff = batch_info(r, results)
        rows.append({
            "system": r["system"], "seed": r["seed"], "notebook": notebook_for(r["system"]),
            "BLEU": round(r["final_val"]["bleu"], 2), "chrF++": round(r["final_val"]["chrfpp"], 2),
            "GM": round(r["final_val"]["geo_mean"], 2),
            "best_epoch": ep, "selection": crit, "gpus": gpus, "steps_per_epoch": steps, "effective_batch": eff,
            "hyperparameters": "; ".join(f"{k}={hp[k]}" for k in KEY_HPARAMS if k in hp),
            "shared_split": "yes" if r.get("val_ids_sha256") == EXPECTED_VAL_SHA256 else "NO",
            "run_record": r["_file"], "predictions": f"preds_val_{r['system']}__seed{r['seed']}.csv",
        })
    df = pd.DataFrame(rows).sort_values(["notebook", "system", "seed"])
    df.to_csv(os.path.join(results, "experiment_registry.csv"), index=False)
    with open(os.path.join(results, "experiment_registry.md"), "w") as f:
        f.write("| System | Seed | Notebook | BLEU | chrF++ | GM | Best epoch | Selected by | GPUs | Steps/epoch | Effective batch | Key hyperparameters | Shared split | Run record |\n")
        f.write("|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|---|\n")
        show = lambda v: "-" if v is None or v != v or v == "" else (f"{v:.0f}" if isinstance(v, float) else v)
        for r in df.itertuples():
            f.write(f"| {r.system} | {r.seed} | {r.notebook} | {r.BLEU} | {r._5} | {r.GM} | {show(r.best_epoch)} | "
                    f"{show(r.selection)} | {show(r.gpus)} | {show(r.steps_per_epoch)} | {show(r.effective_batch)} | {r.hyperparameters} | {r.shared_split} | {r.run_record} |\n")
    bad = df[df.shared_split != "yes"]
    if len(bad):
        print("WARNING: runs not on the shared split:\n", bad[["system", "seed"]].to_string(index=False))
    return df


def curves(runs, results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping curves")
        return
    by_system = defaultdict(list)
    for r in runs:
        if r.get("log_history"):
            by_system[r["system"]].append(r)
    out = os.path.join(results, "figures")
    os.makedirs(out, exist_ok=True)
    for system, rs in by_system.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
        for r in rs:
            lh = r["log_history"]
            tr = [(e["epoch"], e["loss"]) for e in lh if "loss" in e and "epoch" in e]
            ev_loss = [(e["epoch"], e["eval_loss"]) for e in lh if "eval_loss" in e]
            gm_key = next((k for k in ("eval_geo_mean", "eval_geom_mean") if any(k in e for e in lh)), None)
            if tr:
                ax1.plot(*zip(*tr), alpha=0.6, label=f"train, seed {r['seed']}")
            if ev_loss:
                ax1.plot(*zip(*ev_loss), marker="o", label=f"val, seed {r['seed']}")
            if gm_key:
                ax2.plot(*zip(*[(e["epoch"], e[gm_key]) for e in lh if gm_key in e]), marker="o",
                         label=f"seed {r['seed']}")
        ax1.set(title=f"{system}: loss (train: as logged, not normalised)", xlabel="epoch")
        ax2.set(title=f"{system}: validation GM (greedy, during training)", xlabel="epoch")
        ax1.legend(fontsize=7)
        if ax2.lines:
            ax2.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"curves_{system}.png"), dpi=150)
        plt.close(fig)
    print(f"Wrote {out}/curves_*.png")


def _recall(pattern, hyp, ref):
    want = pattern.findall(ref)
    if not want:
        return None
    have = set(pattern.findall(hyp))
    return sum(w in have for w in want) / len(want)


def _repetitive(text, n=4):
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(grams) > 0 and len(set(grams)) < 0.7 * len(grams)


def _cut(text, n=256):
    """First n UTF-8 bytes: the part of a document that Phase 1 read and scored for ByT5."""
    return text.encode("utf-8")[:n].decode("utf-8", errors="ignore")


def error_analysis(runs, results, train_csv, top_k_examples):
    train = pd.read_csv(train_csv)
    src_by_id = dict(zip(train["oare_id"].astype(str), train["transliteration"].astype(str)))
    # One prediction file per system: the first seed, so systems are compared on single runs.
    chosen = {}
    for r in sorted(runs, key=lambda r: r["seed"]):
        chosen.setdefault(r["system"], r)
    lines = ["# Error analysis (validation, shared split)\n",
             "Each system is analysed on its first seed. Length buckets are terciles of the source length in bytes; "
             "'damaged' documents contain `<gap>`/`x` markers in the transliteration.\n"]
    table, examples = [], []
    src_len = {i: len(s.encode("utf-8")) for i, s in src_by_id.items()}
    for system, r in chosen.items():
        df = pd.read_csv(os.path.join(results, f"preds_val_{system}__seed{r['seed']}.csv"))
        df["id"] = df["id"].astype(str)
        df["pred_translation"] = df["pred_translation"].fillna("").astype(str)
        df["target_text"] = df["target_text"].fillna("").astype(str)
        df["src"] = df["id"].map(src_by_id)
        df["src_bytes"] = df["id"].map(src_len)
        df["bucket"] = pd.qcut(df["src_bytes"], 3, labels=["short", "medium", "long"])
        df["damaged"] = df["src"].str.contains(DAMAGE_RE)
        row = {"system": system}
        for name, mask in [("all", df.index == df.index), ("short", df.bucket == "short"),
                           ("medium", df.bucket == "medium"), ("long", df.bucket == "long"),
                           ("damaged", df.damaged), ("intact", ~df.damaged)]:
            sub = df[mask]
            row[f"GM {name} (n={len(sub)})"] = round(corpus_scores(sub.pred_translation.tolist(),
                                                                   sub.target_text.tolist())["geo_mean"], 2)
        row["GM first 256 bytes"] = round(corpus_scores([_cut(t) for t in df.pred_translation],
                                                        [_cut(t) for t in df.target_text])["geo_mean"], 2)
        nums = [x for x in (_recall(NUMBER_RE, h, t) for h, t in zip(df.pred_translation, df.target_text)) if x is not None]
        names = [x for x in (_recall(NAME_RE, h, t) for h, t in zip(df.pred_translation, df.target_text)) if x is not None]
        ratio = df.pred_translation.str.split().str.len() / df.target_text.str.split().str.len().clip(lower=1)
        row.update({
            "number recall": round(float(np.mean(nums)), 3) if nums else None,
            "name recall": round(float(np.mean(names)), 3) if names else None,
            "median length ratio": round(float(ratio.median()), 2),
            "too short (<0.5)": int((ratio < 0.5).sum()),
            "empty": int((df.pred_translation.str.strip() == "").sum()),
            "repetitive": int(df.pred_translation.map(_repetitive).sum()),
        })
        table.append(row)
        df["chrfpp"] = [sentence_chrfpp(h, t) for h, t in zip(df.pred_translation, df.target_text)]
        for x in df.nsmallest(top_k_examples, "chrfpp").itertuples():
            examples.append((system, x.id, x.chrfpp, x.src[:300], x.target_text[:300], x.pred_translation[:300]))
    t = pd.DataFrame(table)
    lines.append("\n## Scores by subset and error statistics\n")
    lines.append("| " + " | ".join(t.columns) + " |")
    lines.append("|" + "---|" * len(t.columns))
    for rr in t.itertuples(index=False):
        lines.append("| " + " | ".join("" if v is None else str(v) for v in rr) + " |")
    lines.append("\n- **GM first 256 bytes**: outputs and references both cut to 256 bytes before scoring (diagnostic only)."
                 "\n- **number recall**: share of numbers in the reference (amounts, weights) that appear in the output."
                 "\n- **name recall**: share of capitalised words (mostly personal and place names) reproduced."
                 "\n- **length ratio**: output words / reference words; < 0.5 usually means a truncated or stopped translation.\n")
    lines.append(f"\n## {top_k_examples} worst documents per system (sentence chrF++)\n")
    for system, doc, score, s, ref, hyp in examples:
        lines.append(f"**{system}**, `{doc}`, chrF++ {score:.1f}\n\n- source: {s}\n- reference: {ref}\n- output: {hyp}\n")
    with open(os.path.join(results, "error_analysis.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {results}/error_analysis.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--train-csv", default=os.path.join(ROOT, "train.csv"))
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()
    runs = load_runs(args.results)
    registry(runs, args.results)
    print(f"Wrote {args.results}/experiment_registry.md")
    curves(runs, args.results)
    error_analysis(runs, args.results, args.train_csv, args.examples)


if __name__ == "__main__":
    main()
