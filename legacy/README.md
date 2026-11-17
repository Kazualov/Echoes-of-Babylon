# Legacy code (not used for any reported result)

`qwen_cli/` is an early command-line version of the Qwen2.5-7B QLoRA training
(`prepare_data.py` → `train.py` → `infer.py`). It uses the cleaned sentence-level
data from `preprocessing/` and its own hash-based split, so its numbers are not
comparable with the notebooks. All results in the report come from the four
notebooks in the repository root, which share one split and one metric
(`common/`). The code is kept only for history.
