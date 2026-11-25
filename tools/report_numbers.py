"""Write report/numbers.tex (LaTeX macros) from the files in results/.

Usage (after evaluate_all.py and analyze_runs.py):
    python tools/report_numbers.py

Every validation number of report/project.tex and report/fixes.tex is one of
these macros, so the report cannot differ from the run records.
"""
import json
import math
import os
import re

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
SYSTEMS = {"Mb": "mbart50", "Qw": "qwen2.5-7b-lora", "Bt": "byt5", "Bm": "byt5-member2",
           "Bx": "byt5-mbr", "En": "mbr3-ensemble"}


def run(system):
    return json.load(open(os.path.join(RES, f"run_{system}__seed42.json"), encoding="utf-8"))


def error_table():
    lines = [l for l in open(os.path.join(RES, "error_analysis.md"), encoding="utf-8") if l.startswith("|")]
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]
    return {r[0]: dict(zip(rows[0], r)) for r in rows[2:]}


def hm(seconds):
    m = round(seconds / 60)
    return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{m} min"


def main():
    macros = {}
    summ = pd.read_csv(os.path.join(RES, "summary.csv")).set_index("system")
    errs = error_table()
    for k, system in SYSTEMS.items():
        r, s = run(system), summ.loc[system]
        assert math.isclose(r["final_val"]["geo_mean"], s["gm_mean"], abs_tol=1e-6), f"{system}: record != re-scored"
        macros.update({k + "BLEU": f"{s.bleu_mean:.2f}", k + "CHR": f"{s.chrfpp_mean:.2f}", k + "GM": f"{s.gm_mean:.2f}",
                       k + "LO": f"{s.gm_ci95_low:.2f}", k + "HI": f"{s.gm_ci95_high:.2f}"})
        e = errs[system]
        for name, prefix in [("Short", "GM short"), ("Medium", "GM medium"), ("Long", "GM long"),
                             ("Damaged", "GM damaged"), ("Intact", "GM intact")]:
            macros[k + name] = f"{float(next(v for c, v in e.items() if c.startswith(prefix))):.2f}"
        macros[k + "Prefix"] = f"{float(e['GM first 256 bytes']):.2f}"
        for name, col in [("Num", "number recall"), ("Name", "name recall"), ("Ratio", "median length ratio"),
                          ("TooShort", "too short (<0.5)"), ("Rep", "repetitive"), ("Empty", "empty")]:
            macros[k + name] = f"{float(e[col]):.2f}" if name == "Ratio" else e[col]

    tests = pd.read_csv(os.path.join(RES, "paired_tests.csv"))
    best = tests["best"].iloc[0]
    tests = tests.set_index("other")
    # Differences are reported as mBART minus other; re-derived from the paired file only if mBART is the best system.
    if best == "mbart50":
        for k in ("Qw", "Bt", "En"):
            t = tests.loc[SYSTEMS[k]]
            macros.update({f"Diff{k}": f"{t.gm_diff_mean:.2f}", f"Diff{k}LO": f"{t.diff_ci95_low:.2f}",
                           f"Diff{k}HI": f"{t.diff_ci95_high:.2f}"})
    else:
        raise SystemExit(f"Best system is {best}: update the report text before generating numbers.")

    for k in ("Mb", "Bt"):
        r = run(SYSTEMS[k])
        lh, hp = r["log_history"], r["hyperparameters"]
        key = next(x for x in ("eval_geo_mean", "eval_geom_mean") if any(x in e for e in lh))
        evals = {}
        for e in lh:
            if key in e:
                evals.setdefault(round(e["epoch"]), (e[key], e["step"]))
        best_epoch = max(evals, key=lambda x: evals[x][0])
        steps = evals[min(evals)][1]
        train = next(e for e in lh if "train_runtime" in e)
        rows, micro = 1405, 2 * 4
        gpus = {math.ceil(math.ceil(rows / (2 * g)) / 4): g for g in (1, 2)}[steps]
        macros.update({k + "BestEpoch": str(best_epoch), k + "Steps": str(steps), k + "Gpus": str(gpus),
                       k + "EffBatch": str(micro * gpus), k + "Time": hm(train["train_runtime"]),
                       k + "Curve": " ".join(f"({ep},{v:.2f})" for ep, (v, _) in sorted(evals.items()))})
        if "effective_batch_size" in hp:
            assert hp["effective_batch_size"] == micro * gpus, f"{k}: effective batch mismatch"

    sel = run("mbr3-ensemble")["hyperparameters"]["selected_per_member"]
    picks = f"mBART {sel.get('mbart50', 0)}, Qwen {sel.get('qwen2.5-7b-lora', 0)}, ByT5 {sel.get('byt5', 0)}"
    lo, hi = float(macros["DiffEnLO"]), float(macros["DiffEnHI"])
    verdict = ("This difference is larger than expected from document sampling alone." if lo > 0 or hi < 0
               else "The interval includes zero, so the two systems cannot be separated on this validation set.")
    macros["EnsembleSentence"] = (
        f"The three-model ensemble scores {macros['DiffEn']} GM below mBART-50 alone "
        f"(paired bootstrap, 95\\% CI {macros['DiffEnLO']} to {macros['DiffEnHI']}). {verdict} "
        f"The documents were taken from: {picks}. Equal votes give two weaker models a majority over the strongest "
        f"one, so the proposed single model is kept.")
    macros["EnsembleSummary"] = (
        f"An equal-vote MBR ensemble of the three models scored {macros['EnGM']}, against {macros['MbGM']} for mBART-50 "
        f"alone. It did not improve on the best single model.")
    macros["WeightsSentence"] = os.environ.get(
        "WEIGHTS_SENTENCE", "The Phase 2 weights are kept by the team but are not part of the archive because of their size (mBART 2.4 GB, "
        "ByT5 1.2 GB per model, Qwen adapter 0.16 GB) and are not published. Retraining instructions are in the README.")

    assert all(re.fullmatch(r"[A-Za-z]+", m) for m in macros)
    out = os.path.join(ROOT, "report", "numbers.tex")
    with open(out, "w", encoding="utf-8") as f:
        f.write("% Generated by tools/report_numbers.py from results/. Do not edit.\n")
        f.writelines(f"\\newcommand{{\\{m}}}{{{v}}}\n" for m, v in macros.items())
    print(f"Wrote {out} ({len(macros)} macros)")

    # Self-contained versions (numbers filled in) for Overleaf: one file per PDF.
    flat = os.path.join(ROOT, "report", "overleaf")
    os.makedirs(flat, exist_ok=True)
    for name in ("project.tex", "fixes.tex"):
        tex = open(os.path.join(ROOT, "report", name), encoding="utf-8").read().replace("\\input{numbers}\n", "")
        tex = re.sub(r"\\([A-Z][A-Za-z]*)(?![A-Za-z])(\{\})?",
                     lambda m: macros[m.group(1)] if m.group(1) in macros else m.group(0), tex)
        with open(os.path.join(flat, name), "w", encoding="utf-8") as f:
            f.write(tex)
    print(f"Wrote {flat}/project.tex and fixes.tex")


if __name__ == "__main__":
    main()
