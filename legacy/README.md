# Legacy code (not used for any reported result)

`qwen_cli/` is an early command-line version of the Qwen2.5-7B QLoRA training
(`prepare_data.py` → `train.py` → `infer.py`). It uses the cleaned sentence-level
data from `preprocessing/` and its own hash-based split, so its numbers are not
comparable with the notebooks. All results in the report come from the four
notebooks in the repository root, which share one split and one metric
(`common/`). The code is kept only for history.

## `phase1/`

`ensemble-pipeline_phase1.ipynb` is the ensemble notebook exactly as submitted in
Phase 1. It loads the Phase 1 weights, which are still published on Kaggle
(`eeee13/akkadian-mbart` `subword-seq2seq-v1`, `eeee13/akkadian-byt5`
`byt5-akkadian-v4`, `eeee13/akkadika` `adapter-v1`, `eeee13/qwen2.5-7b-instruct`
`base-v1`, dataset `eeee13/bitsandbytes`), and reproduces the public leaderboard
score 27.3855. Its validation scores must not be used: ByT5 and Qwen were trained
on other splits, so part of its validation set was in their training data.

## `executed/`

`pipeline_QwenQLoRa_executed_seed42.ipynb` is the Kaggle notebook, with outputs,
that produced the reported Qwen result. It stops at the first generation call
(autocast missing) and a recovery cell finishes the evaluation, so it is kept as
a record only. Run `pipeline_QwenQLoRa.ipynb` in the repository root instead.
