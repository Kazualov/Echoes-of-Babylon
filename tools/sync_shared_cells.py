"""Copy common/*.py into the notebooks' shared cells.

Kaggle notebooks cannot import from this repository, so each notebook carries a
verbatim copy of the shared modules in cells that start with
    # >>> SHARED CELL: common/<module>.py
Run this script after editing anything in common/ (it rewrites those cells),
or with --check to fail if any notebook copy is out of date.
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NOTEBOOKS = ["mbart50.ipynb", "byte-level-seq2seq.ipynb", "pipeline_QwenQLoRa.ipynb", "ensemble-pipeline.ipynb"]
MARKER = re.compile(r"^# >>> SHARED CELL: (common/[\w]+\.py)")


def cell_body(module_path):
    src = open(os.path.join(ROOT, module_path), encoding="utf-8").read()
    src = re.sub(r'^""".*?"""\n', "", src, count=1, flags=re.S)  # docstring -> header below
    src = "\n".join(l for l in src.splitlines() if not l.startswith("from common."))
    header = (f"# >>> SHARED CELL: {module_path}\n"
              f"# Auto-copied from the repository by tools/sync_shared_cells.py. Do not edit here;\n"
              f"# identical in mbart50, byte-level-seq2seq, pipeline_QwenQLoRa and ensemble-pipeline.\n")
    return header + src.strip("\n") + "\n"


def to_source(text):
    lines = text.splitlines(keepends=True)
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    stale = []
    for name in NOTEBOOKS:
        path = os.path.join(ROOT, name)
        nb = json.load(open(path, encoding="utf-8"))
        changed = False
        for cell in nb["cells"]:
            src = "".join(cell["source"])
            m = MARKER.match(src)
            if not m:
                continue
            want = to_source(cell_body(m.group(1)))
            if cell["source"] != want:
                stale.append(f"{name}: {m.group(1)}")
                cell["source"] = want
                changed = True
        if changed and not args.check:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
                f.write("\n")
    if stale:
        print(("STALE:\n  " if args.check else "Updated:\n  ") + "\n  ".join(stale))
        if args.check:
            sys.exit(1)
    else:
        print("All shared cells up to date.")


if __name__ == "__main__":
    main()
