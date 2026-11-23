# Deep Past Initiative: Machine Translation (Old Assyrian → English)

Solution and research code for the [Deep Past Initiative Machine Translation](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/overview) competition: three fine-tuned translators (mBART-50, ByT5-small, Qwen2.5-7B with QLoRA) and a Minimum Bayes Risk (MBR) ensemble of the three.

All notebooks run in **Kaggle Notebooks** with a GPU. The helper scripts in `tools/` run on any machine with Python ≥ 3.10.

## Task, data and metric

- **Task:** translate transliterated Old Assyrian cuneiform documents (mostly merchant letters and contracts) into English. The language is low-resource and morphologically rich. The texts contain scribal notation, determinatives and damaged passages (`<gap>`).
- **Data:** competition `train.csv` with 1,561 documents (`oare_id`, `transliteration`, `translation`) and `test.csv`.
- **Metric:** the competition score is the geometric mean (GM) of corpus BLEU and corpus chrF++. Every number in this repository is computed by `common/metrics.py`. It is a dependency-free port of sacreBLEU 2.x (`BLEU()` with the 13a tokenizer and `CHRF(word_order=2)`), and `tests/test_metrics.py` checks that it returns exactly the same scores as sacreBLEU.

## Evaluation protocol (shared by all models)

| What | How | Where |
|---|---|---|
| Validation split | 10% of **documents**: sorted `oare_id`s shuffled with `random.Random(42)`, giving 156 documents. Every notebook checks the SHA-256 of the sorted ids and stops if it differs. | `common/split.py`, `data/val_ids.csv` |
| Metric | sacreBLEU-identical BLEU + chrF++ (char 6-grams + word 2-grams), GM = √(BLEU·chrF++) | `common/metrics.py` |
| Reported score | Computed **after** training, on the best checkpoint, against the full (untruncated) reference translations | `finalize_run` in `common/runlog.py` |
| Checkpoint selection | Validation GM after every epoch, greedy decoding (mBART, ByT5). Qwen uses validation loss, because generating 156 documents with a 7B model every epoch is too slow. | training notebooks |
| Seeds | mBART and ByT5: seeds 42, 43, 44. Qwen: seed 42 (one 7B run). Results are reported as mean ± std. | `SEED` in each notebook |
| Uncertainty | 95% bootstrap CI over validation documents (1,000 resamples) and a paired bootstrap test against the best system | `tools/evaluate_all.py` |
| Run record | One `preds_val_<system>__seed<k>.csv` and one `run_<system>__seed<k>.json` per trained model (hyperparameters, split checksum, metric, per-epoch log), plus `env_<notebook>.json` with package versions and GPU | `results/` |

The shared code lives in `common/`. Kaggle notebooks cannot import from the repository, so each notebook contains verbatim copies of these modules in cells marked `# >>> SHARED CELL`. `python tools/sync_shared_cells.py --check` verifies that the copies are identical to `common/`.

### Length limits

Each model's limits are set so that at least 95% of training documents fit without truncation. The limits are measured with that model's own tokenizer:

| Model | Unit | Source (p95) | Target (p95) | Limits used |
|---|---|---:|---:|---|
| mBART-50 | sentencepiece tokens | 568 | 413 | source 576, target 448 |
| ByT5-small | UTF-8 bytes | 1,024 | 1,384 | source 1,024, target 1,408 |
| Qwen2.5-7B | Qwen tokens | 543 | 419 | prompt + source + target 1,024, new tokens 448 |

## Repository layout

```
mbart50.ipynb               mBART-50 fine-tuning, validation, test inference
byte-level-seq2seq.ipynb    ByT5-small fine-tuning (+ ByT5-only MBR ensemble), validation, test inference
pipeline_QwenQLoRa.ipynb    Qwen2.5-7B-Instruct QLoRA fine-tuning and validation
ensemble-pipeline.ipynb     3-model MBR ensemble (offline), validation and submission.csv
common/                     shared split, metric, run record, environment report
tools/evaluate_all.py       one table for all runs: mean ± std over seeds, bootstrap CIs, paired tests
tools/analyze_runs.py       experiment registry, training curves, error analysis
tools/sync_shared_cells.py  copies common/*.py into the notebooks (or checks them with --check)
tests/test_metrics.py       metric parity with sacreBLEU
data/val_ids.csv            the 156 shared validation documents
results/                    run records produced by the notebooks (input for the tools)
preprocessing/              sentence-level dataset pipeline (not used for the reported models)
legacy/                     early Qwen CLI scripts (not used for any reported result)
```

## Reproducing the results

### Kaggle inputs

| Resource | Kaggle link | Needed by |
|---|---|---|
| Competition data | [deep-past-initiative-machine-translation](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation) | all notebooks |
| mBART-50 base weights (third party) | [facebookmbart-large-50-many-to-many-mmt](https://www.kaggle.com/models/nensipansuriya1311/facebookmbart-large-50-many-to-many-mmt) | `mbart50.ipynb` (offline mode) |
| ByT5-small pretrained base weights | Attach the pretrained `google/byt5-small` checkpoint as a Kaggle Model or Dataset input; set `BASE_MODEL_OFFLINE_PATH` if its mount differs | `byte-level-seq2seq.ipynb` (offline mode) |
| Qwen2.5-7B-Instruct base | [eeee13/qwen2.5-7b-instruct](https://www.kaggle.com/models/eeee13/qwen2.5-7b-instruct), variation `base-v1` | `ensemble-pipeline.ipynb` |
| Fine-tuned mBART | [eeee13/akkadian-mbart](https://www.kaggle.com/models/eeee13/akkadian-mbart), variation `mbart50-seed42` | `ensemble-pipeline.ipynb` |
| Fine-tuned ByT5 | [eeee13/akkadian-byt5](https://www.kaggle.com/models/eeee13/akkadian-byt5), variation `byt5-seed42` | `ensemble-pipeline.ipynb` |
| Qwen LoRA adapter | [eeee13/akkadika](https://www.kaggle.com/models/eeee13/akkadika), variation `qwen-lora-seed42` | `ensemble-pipeline.ipynb` |
| bitsandbytes wheel | [eeee13/bitsandbytes](https://www.kaggle.com/datasets/eeee13/bitsandbytes) | `ensemble-pipeline.ipynb` (offline install) |

The fine-tuned weights are published. The ensemble can therefore be reproduced without retraining anything, and every training notebook can be rerun from scratch.

### Settings common to all notebooks

- Accelerator: **GPU T4 ×2** or **P100** (≥ 16 GB VRAM).
- Environment: Kaggle's default Python image. The versions actually used are recorded in `results/env_<notebook>.json`. For an exact rerun, select *Settings → Environment → Pin to original environment*.
- Default flags never read Kaggle Secrets and never upload anything: `USE_TRACKING = False`, `PUBLISH_MODELS = False`. mBART, ByT5 and the ensemble default to `EXECUTION_MODE = "offline"`; attach their required model inputs before running offline.
- Outputs: each notebook writes `results/` into `/kaggle/working/`. Download it from the *Output* tab.

### 1. `mbart50.ipynb` (about 1–1.5 h per seed on a T4)

1. Attach the competition data and the third-party mBART-50 base weights. Set Internet **OFF** (offline mode loads the base model from the attached input).
2. Keep `EXECUTION_MODE = "offline"` and set `SEED` (42, then 43, then 44).
3. *Run All*. Outputs: `results/preds_val_mbart50__seed<k>.csv`, `results/run_mbart50__seed<k>.json`, `results/env_mbart50.json`, the model in `mbart50-model/` and `submission.csv`.

### 2. `byte-level-seq2seq.ipynb` (about 3–4 h for seed 42, about 2 h for each of seeds 43 and 44)

1. Attach the competition data and the pretrained `google/byt5-small` checkpoint as a Kaggle Model or Dataset input. Keep Internet **OFF** and `EXECUTION_MODE = "offline"`; set `BASE_MODEL_OFFLINE_PATH` only if the mounted path differs from the default (the notebook can auto-detect one attached ByT5 checkpoint). The offline run skips pip, secrets, tracking, Hub calls and downloads.
2. Keep `USE_TRACKING = False`, `PUBLISH_MODELS = False`, and set `SEED`. Seed 42 trains the baseline (`byt5`, full training split) and one extra member (`byt5-member2`, 85% bootstrap subset) for the ByT5-only MBR ensemble. Seeds 43 and 44 train only the baseline.
3. *Run All*. If you later need to publish weights, enable Internet and set `EXECUTION_MODE = "online"`, `PUBLISH_MODELS = True`, then run the final publishing cell. Outputs: `results/preds_val_byt5__seed<k>.csv` (plus `byt5-member2` and `byt5-mbr` for seed 42), the run records, the weights in `outputs/final_model_*` and `submission.csv`.

### 3. `pipeline_QwenQLoRa.ipynb` (several hours on a T4)

1. Attach the competition data. Set Internet **ON** (downloads `Qwen/Qwen2.5-7B-Instruct`).
2. Keep `USE_TRACKING = False`. *Run All*. Outputs: `results/preds_val_qwen2.5-7b-lora__seed42.csv`, the run record, and the adapter in `outputs/qwen2.5-7b-lora/`.

### 4. `ensemble-pipeline.ipynb` (offline, about 1 h)

1. Attach every resource in the table above (the competition data, the Qwen base model, the three fine-tuned members and the bitsandbytes wheel). Set Internet **OFF**.
2. Check that the paths in cell *“Local Kaggle input paths”* match the *Input* panel (the version number may differ).
3. *Run All*. Outputs: `submission.csv`, `ensemble_candidates_debug.csv`, and run records for every member and for `mbr3-ensemble`.

### 5. Results table, figures and error analysis (local, a few minutes)

Collect every notebook's `results/` folder into the repository's `results/` folder, then run:

```bash
pip install -r requirements.txt
python tools/evaluate_all.py results/preds_val_*.csv --out results
python tools/analyze_runs.py --results results
python -m pytest tests/
python tools/sync_shared_cells.py --check
```

`train.csv` (the competition file) must be in the repository root.

| Output | Content |
|---|---|
| `results/summary.md` | Main comparison: BLEU, chrF++, GM mean ± std over seeds, 95% bootstrap CI |
| `results/paired_tests.csv` | Paired bootstrap of every system against the best one |
| `results/experiment_registry.md` | One row per run: notebook, seed, hyperparameters, best epoch, scores, run record file |
| `results/figures/curves_*.png` | Training loss and validation GM per epoch |
| `results/error_analysis.md` | GM by document length and damage, number and name recall, length ratio, worst examples |

### Optional: tracking and publishing

- `USE_TRACKING = True` logs runs to MLflow/DagsHub and pushes to the Hugging Face Hub. It reads `HF_TOKEN`, `MLFLOW_USERNAME`, `MLFLOW_PASSWORD`, `MLFLOW_TRACKING_URI` (and `HF_REPO_NAME` for mBART) from Kaggle Secrets.
- `PUBLISH_MODELS = True` uploads the trained weights to Kaggle Models under `KAGGLE_USERNAME` (Internet ON). The ensemble expects the variations `mbart50-seed42`, `byt5-seed42` and `qwen-lora-seed42`.

## Data preparation pipeline (not used by the reported models)

`preprocessing/` normalizes transliterations, extracts sentence-level pairs from the published documents and writes a sentence-level dataset (see `preprocessing/README_dataset.md`). The reported models are trained on the raw document-level `train.csv`. The sentence-level data is future work.
