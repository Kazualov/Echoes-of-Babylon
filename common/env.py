"""Shared environment report: records the exact package versions and GPU of a run.

`report_environment("<notebook>")` prints the versions and writes
results/env_<notebook>.json, so every reported number can be traced to the
software it was produced with (Kaggle images change over time).

The same source is pasted into each notebook by `tools/sync_shared_cells.py`;
edit this file, then re-run that script. Do not edit the notebook copies.
"""
import importlib
import json
import os
import platform
import sys

TRACKED_PACKAGES = ["torch", "transformers", "datasets", "accelerate", "peft", "bitsandbytes",
                    "sentencepiece", "sacrebleu", "pandas", "numpy", "scikit-learn"]


def report_environment(notebook, out_dir="results"):
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            module = importlib.import_module("sklearn" if name == "scikit-learn" else name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = None
    env = {"notebook": notebook, "python": sys.version.split()[0], "platform": platform.platform(),
           "packages": versions, "gpu": None, "cuda": None}
    try:
        import torch
        if torch.cuda.is_available():
            env["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            env["cuda"] = torch.version.cuda
    except Exception:
        pass
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"env_{notebook}.json"), "w") as f:
        json.dump(env, f, indent=2)
    print(json.dumps(env, indent=2))
    return env
