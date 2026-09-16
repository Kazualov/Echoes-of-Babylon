# Echoes of Babylon

Neural machine translation of transliterated Old Assyrian cuneiform into modern English, developed for the Deep Past Initiative machine translation competition.

## Overview

This repository contains the full pipeline for training, evaluating, and ensembling several neural machine translation models on the Old Assyrian–English parallel corpus. The project explores three architectures:

- **mBART-large-50** – multilingual sequence-to-sequence model (best single model).
- **ByT5-small** – byte-level sequence-to-sequence model.
- **Qwen2.5-7B-Instruct** – large language model fine-tuned with QLoRA.

The final submission is an ensemble of these models using Minimum Bayes Risk (MBR) decoding.

## Repository Structure

```
.
├── preprocessing/               # Data preparation utilities
│   ├── README_dataset.md        # Detailed data description
│   ├── extract_sentences.py     # Extract sentence pairs from documents
│   ├── normalize.py             # Transliteration normalisation
│   └── pipeline.py              # End-to-end data preparation
├── src/                         # Core training and inference scripts
│   ├── infer.py                 # Run inference with a trained model
│   ├── prepare_data.py          # Prepare data for training
│   └── train.py                 # Fine-tune a model
├── byte-level-seq2seq.ipynb     # ByT5 training & inference notebook
├── ensemble-pipeline.ipynb      # MBR ensemble of all models
├── mbart50.ipynb                # mBART-50 training & evaluation
├── pipeline_QwenQLoRa.ipynb     # Qwen2.5 QLoRA fine-tuning
├── .env.example                 # Environment variables template
├── .gitignore
└── README.md
```

## Requirements

- Python 3.12+
- CUDA-enabled PyTorch (for GPU training)
- Kaggle environment (recommended) or a machine with an NVIDIA GPU (≥16 GB VRAM)

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the packages used in the notebooks:

```bash
pip install transformers datasets accelerate peft sentencepiece \
            evaluate sacrebleu mlflow huggingface_hub bitsandbytes
```

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Kazualov/Echoes-of-Babylon.git
   cd Echoes-of-Babylon
   ```

2. **Configure environment variables**

   Copy `.env.example` to `.env` and fill in your credentials (Kaggle, Hugging Face, DagsHub, etc.):

   ```bash
   cp .env.example .env
   # edit .env with your values
   ```

3. **Prepare the data**

   The competition data should be placed under `data/` or attached as a Kaggle input.
   Run the preprocessing pipeline:

   ```bash
   python preprocessing/pipeline.py
   ```

   This normalises transliterations, extracts sentence-level pairs, and creates train/validation splits.

## Training & Inference

All experiments are implemented as Jupyter notebooks. You can run them locally or on Kaggle.

| Notebook | Description |
|----------|-------------|
| `mbart50.ipynb` | Fine-tune mBART-large-50 (baseline & raw-data experiments) |
| `pipeline_QwenQLoRa.ipynb` | QLoRA fine-tuning of Qwen2.5-7B-Instruct |
| `byte-level-seq2seq.ipynb` | Train ByT5-small and generate predictions |
| `ensemble-pipeline.ipynb` | Combine predictions from all models via MBR decoding |

Each notebook contains detailed instructions and hyperparameters.
For offline Kaggle evaluation, set `EXECUTION_MODE = "offline"` in the relevant notebooks.

## Reproducibility

To reproduce the reported results:

1. Attach the required models and datasets as inputs in your Kaggle notebook.
   The exact input names and paths are listed inside each notebook.
2. Run the notebooks in the order: data preparation → mBART-50 → ByT5 → Qwen → ensemble.
3. Experiments are logged to MLflow on a private DagsHub instance. Replace the tracking URI
   in your environment configuration with your own DagsHub project to enable logging.

Detailed data preparation notes are in `preprocessing/README_dataset.md`.

## Results

| Model | Validation BLEU | Validation chrF++ | Geometric Mean |
|-------|----------------|-------------------|----------------|
| mBART-large-50 | 20.12 | 40.80 | 28.64 |
| ByT5-small | 12.34 | 45.67 | 23.74 |
| Qwen2.5-7B-Instruct (QLoRA) | 11.44 | 29.46 | 18.36 |

**Public leaderboard:**

| Submission | Geometric Mean (public) |
|------------|------------------------|
| mBART-50 (single model) | **28.0895** |
| Ensemble pipeline | 27.3855 |

The ensemble did not improve over the single mBART-50 model on the public leaderboard.
A detailed analysis of this negative result is provided in the project report.

## License

This project is released for research and educational purposes.
Please refer to the competition rules for data usage restrictions.

## Acknowledgements

We thank the organisers of the Deep Past Initiative for providing the dataset and evaluation framework.
