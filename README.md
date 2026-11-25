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
| Seeds | One training run per model (seed 42). Seed-to-seed variance was **not** measured. | `SEED` in each notebook |
| Decoding of the reported scores | mBART: greedy. Qwen: greedy. ByT5: beam search (8 beams). Each prediction file comes from the model's own training notebook. | training notebooks |
| Uncertainty | 95% bootstrap CI over validation documents and a paired bootstrap test against the best system (5,000 resamples). They cover document sampling only, not training variance | `tools/evaluate_all.py` |
| Run record | One `preds_val_<system>__seed<k>.csv` and one `run_<system>__seed<k>.json` per trained model (hyperparameters, split checksum, metric, per-epoch log), plus `env_<notebook>.json` with package versions and GPU | `results/` |

The shared code lives in `common/`. Kaggle notebooks cannot import from the repository, so each notebook contains verbatim copies of these modules in cells marked `# >>> SHARED CELL`. `python tools/sync_shared_cells.py --check` verifies that the copies are identical to `common/`.

### Length limits

Each model's limits follow the 95th percentile (p95) of the training documents, measured with that model's own tokenizer. One exception: the ByT5 limit includes the end token, so 93.5% of the sources fit completely and the others lose at most 33 bytes (the longest source has 1,056 bytes).

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
ensemble-pipeline.ipynb     3-model MBR ensemble (offline); not executed in Phase 2, see below
common/                     shared split, metric, run record, environment report
tools/evaluate_all.py       one table for all runs: scores, bootstrap CIs, paired tests (and mean ± std if several seeds exist)
tools/analyze_runs.py       experiment registry, training curves, error analysis
tools/mbr_from_predictions.py  MBR ensemble from the saved validation predictions (no GPU)
tools/sync_shared_cells.py  copies common/*.py into the notebooks (or checks them with --check)
tests/                      metric parity with sacreBLEU, MBR selection rule
data/val_ids.csv            the 156 shared validation documents
results/                    run records and predictions of the reported runs, plus the generated tables
results/phase1/             exported MLflow records of the Phase 1 runs
preprocessing/              sentence-level dataset pipeline (not used for the reported models)
legacy/                     Phase 1 ensemble, executed Qwen notebook, early Qwen CLI scripts
```

## Reproducing the results

### What is provided and what must be retrained

| Item | Status |
|---|---|
| Validation predictions, run records, environment of every reported Phase 2 run | in `results/` |
| Phase 2 weights (mBART 2.4 GB, ByT5 1.2 GB per model, Qwen adapter 162 MB) | kept by the team, **not part of the repository or archive and not published**. Retrain with the notebooks below |
| Phase 1 weights of all three models (trained on the old, different splits) | published on Kaggle under `eeee13`; used only by `legacy/phase1/` |

All reported validation scores, tables, figures and the ensemble score can be recomputed **without a GPU and without weights** from `results/` (step 5). Retraining reproduces the procedure; scores can differ slightly between runs because GPU training is not fully deterministic.

### Kaggle inputs

| Resource | Kaggle link | Needed by |
|---|---|---|
| Competition data | [deep-past-initiative-machine-translation](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation) | all notebooks |
| mBART-50 base weights (third party) | [facebookmbart-large-50-many-to-many-mmt](https://www.kaggle.com/models/nensipansuriya1311/facebookmbart-large-50-many-to-many-mmt) | `mbart50.ipynb` (offline mode) |
| Qwen2.5-7B-Instruct base | [eeee13/qwen2.5-7b-instruct](https://www.kaggle.com/models/eeee13/qwen2.5-7b-instruct), variation `base-v1` | `ensemble-pipeline.ipynb` |
| bitsandbytes wheel | [eeee13/bitsandbytes](https://www.kaggle.com/datasets/eeee13/bitsandbytes) | `ensemble-pipeline.ipynb` (offline install) |

ByT5 and Qwen download their pretrained base models (`google/byt5-small`, `Qwen/Qwen2.5-7B-Instruct`) from Hugging Face, so they need Internet ON.

### Settings common to all notebooks

- Accelerator: **GPU T4 ×2** (used for the recorded runs). On T4 ×2 the Trainer uses both GPUs, so the effective batch is per-device batch × accumulation × 2. On a single GPU it is half as large.
- Environment: Kaggle's default Python image. The versions actually used are recorded in `results/env_<notebook>.json`. For an exact rerun, select *Settings → Environment → Pin to original environment*.
- Default flags never read Kaggle Secrets and never upload anything: `USE_TRACKING = False`, `PUBLISH_MODELS = False`. mBART and the ensemble default to `EXECUTION_MODE = "offline"`, ByT5 to `"online"`.
- Outputs: each notebook writes `results/` into `/kaggle/working/`. Download it from the *Output* tab.

### 1. `mbart50.ipynb` (recorded training time: 61 min)

1. Attach the competition data and the third-party mBART-50 base weights. Internet **OFF** (offline mode loads the base model from the attached input).
2. Keep `EXECUTION_MODE = "offline"` and `SEED = 42`. *Run All*.
3. Outputs: `results/preds_val_mbart50__seed42.csv`, `results/run_mbart50__seed42.json`, `results/env_mbart50.json`, the model in `mbart50-model/` and `submission.csv`.

It runs on the current Kaggle image (the Phase 1 argument `group_by_length`, removed in transformers 5.2, is gone). Validation is decoded greedily; `submission.csv` uses the base model's generation defaults (5 beams) and was not evaluated in Phase 2: one of its four sample-test outputs is a repetition loop.

### 2. `byte-level-seq2seq.ipynb` (recorded training time: 6 h 12 min for `byt5`, 4 h 19 min for `byt5-member2`)

1. Attach the competition data. Internet **ON**; keep `EXECUTION_MODE = "online"` (downloads `google/byt5-small`; no account or token needed).
2. Keep `USE_TRACKING = False`, `PUBLISH_MODELS = False`, `SEED = 42`. *Run All* trains the baseline (`byt5`, full training split) and `byt5-member2` (85% subset) and evaluates the ByT5-only MBR ensemble. The whole recorded notebook took 11 h 24 min, close to Kaggle's 12 h session limit. Set `N_MODELS = 1` to reproduce only the baseline.
3. Outputs: `results/preds_val_byt5__seed42.csv` (plus `byt5-member2` and `byt5-mbr`), the run records, the weights in `outputs/final_model_*` and `submission.csv`.

### 3. `pipeline_QwenQLoRa.ipynb` (recorded training time: 1 h 53 min, plus validation generation)

1. Attach the competition data. Accelerator T4 ×2, Internet **ON** (downloads `Qwen/Qwen2.5-7B-Instruct`).
2. Keep `USE_TRACKING = False` and `PUBLISH_MODELS = False`. *Run All*.
3. Outputs: `results/preds_val_qwen2.5-7b-lora__seed42.csv`, the run record, and the adapter in `outputs/qwen2.5-7b-lora/`.

In the recorded run generation first failed because it ran outside autocast; the same process then generated the validation predictions inside autocast. The notebook now contains that fix in `generate_predictions`; the training code is unchanged and the fixed notebook completed a short smoke test on Kaggle: it ran to the end without errors with heavily shortened training and generation. **The full training was not repeated**, so the reported scores come from the recorded run. The executed original is kept in `legacy/executed/`. The recorded run used two GPUs (44 steps per epoch, effective batch 32).

### 4. `ensemble-pipeline.ipynb` (not executed in Phase 2)

Needs the three fine-tuned members, whose Phase 2 weights are not published. To run it, train the members (steps 1–3), upload the saved folders as Kaggle inputs, set the four paths in *Local Kaggle input paths*, attach the remaining inputs from the table above, set Internet **OFF** and *Run All*. It decodes every member with 4 beams, so its scores will not equal the reported ones.

The ensemble score in the report is computed from the recorded predictions with the same MBR rule (step 5). To reproduce the Phase 1 leaderboard submission (27.3855), use `legacy/phase1/`.

### 5. Results table, figures and error analysis (local, a few minutes)

`results/` already contains the recorded predictions and run records. If you retrained a model, copy the notebook's `results/` folder over it. Then run:

```bash
pip install -r requirements.txt
python tools/mbr_from_predictions.py results/preds_val_byt5__seed42.csv results/preds_val_mbart50__seed42.csv results/preds_val_qwen2.5-7b-lora__seed42.csv
python tools/evaluate_all.py results/preds_val_*.csv --out results --n-boot 5000
python tools/analyze_runs.py --results results
python -m pytest tests/
python tools/sync_shared_cells.py --check
```

`train.csv` (the competition file) must be in the repository root.

| Output | Content |
|---|---|
| `results/summary.md` | Main comparison: BLEU, chrF++, GM and 95% bootstrap CI |
| `results/paired_tests.csv` | Paired bootstrap of every system against the best one |
| `results/experiment_registry.md` | One row per run: notebook, seed, hyperparameters, best epoch, scores, run record file |
| `results/figures/curves_*.png` | Training loss and validation GM per epoch |
| `results/error_analysis.md` | GM by document length and damage, number and name recall, length ratio, worst examples |

### Optional: tracking and publishing

- `USE_TRACKING = True` logs runs to MLflow/DagsHub and pushes to the Hugging Face Hub. It reads `HF_TOKEN`, `MLFLOW_USERNAME`, `MLFLOW_PASSWORD`, `MLFLOW_TRACKING_URI` (and `HF_REPO_NAME` for mBART) from Kaggle Secrets.
- `PUBLISH_MODELS = True` uploads the trained weights to Kaggle Models under `KAGGLE_USERNAME` (Internet ON). The ensemble expects the variations `mbart50-seed42`, `byt5-seed42` and `qwen-lora-seed42`.

## Data preparation pipeline (not used by the reported models)

`preprocessing/` normalizes transliterations, extracts sentence-level pairs from the published documents and writes a sentence-level dataset (see `preprocessing/README_dataset.md`). The reported models are trained on the raw document-level `train.csv`. The sentence-level data is future work.
