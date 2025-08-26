

# Comprehensive Evaluation of Transformer Models for Hate Speech Detection

This repository contains the code and experiments for the research project **“Comprehensive Evaluation of Various Transformer Models in Detecting Normal, Hate, and Offensive Text.”**
It benchmarks several transformer architectures on a curated Kaggle dataset to classify tweets into **Normal**, **Hate**, and **Offensive** categories.

---

## 📌 Objective

Evaluate and compare transformer models for offensive/hate speech detection on social media. Models are compared on **accuracy, F1 (macro/micro/weighted), precision, recall, evaluation loss, runtime,** and **training efficiency**.

---

## 📂 Repository Structure

```
├── config/
│   └── config.yaml                 # Default hyperparameters & paths
├── data/
│   └── README.md                   # Dataset source & expected schema (put merged CSV here)
├── notebooks/                      # Exploratory notebooks (import from src/)
├── results/                        # Plots, confusion matrices, aggregated metrics
├── scripts/
│   ├── prepare_dataset.py          # Merge the 12 Kaggle CSVs → data/merged_dataset.csv
│   ├── make_figures.py             # Label distribution & length-by-label plots
│   └── make_curves_from_state.py   # Plot curves from trainer_state.json (optional)
├── src/
│   └── hate_speech_eval/
│       ├── __init__.py
│       ├── config.py               # Label maps & max lengths
│       ├── data.py                 # Dataset loading & tokenization
│       ├── metrics.py              # Accuracy / Macro & Micro F1 / Precision / Recall
│       ├── trainer.py              # CustomTrainer with class-weighted loss
│       ├── train.py                # Training entrypoint (supports --config)
│       ├── evaluate.py             # Simple evaluation/prediction demo
│       └── viz_curves.py           # Plot training/eval curves from log_history
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 🔍 Dataset

* **Source:** [https://www.kaggle.com/datasets/subhajournal/normal-hate-and-offensive-speeches](https://www.kaggle.com/datasets/subhajournal/normal-hate-and-offensive-speeches)
* **Classes:** `Normal`, `Hate`, `Offensive`
* **Preparation:** The original source provides **12 CSVs** (4 per class). Merge them into a single file:

  ```bash
  python scripts/prepare_dataset.py --input_dir /path/to/kaggle_csvs --out data/merged_dataset.csv
  ```
* **Preprocessing choices used in the study:**

  * Keep case (no lowercasing) and **retain special characters** (important for obfuscation/intent).
  * Label-specific length filtering (as described in the article).
  * Tokenizer max length ≈ **114** (1.3× longest sequence).

---

## ⚙️ Experimental Setup

* **Hardware:** NVIDIA A100 GPU
* **Defaults:** `epochs=5`, `batch_size=16`, `learning_rate=2e-5`, `max_length=114`
* **Class imbalance:** handled via **class-weighted CrossEntropyLoss** in a `CustomTrainer`.

> You can edit these in `config/config.yaml` or override via CLI flags.

---

## 🤖 Models Evaluated

* `bert-base-uncased`
* `bert-large-uncased`
* `distilbert-base-uncased`
* `diptanu/fBERT`
* `GroNLP/hateBERT`
* `roberta-large`

---

## 📊 Results (Actual)

### A) Training Metrics

| Model                       | Accuracy | Weighted F1 | Weighted Recall | Weighted Precision | Micro F1 | Micro Recall | Micro Precision | Macro F1 | Macro Recall | Macro Precision | Training Time | Evaluation Time |
| --------------------------- | -------: | ----------: | --------------: | -----------------: | -------: | -----------: | --------------: | -------: | -----------: | --------------: | ------------: | --------------: |
| **bert-base-uncased**       |   0.9935 |      0.9935 |          0.9935 |             0.9935 |   0.9935 |       0.9935 |          0.9935 |   0.9933 |       0.9931 |          0.9936 |        83.87s |           650ms |
| **bert-large-uncased**      |   0.9935 |      0.9935 |          0.9935 |             0.9936 |   0.9935 |       0.9935 |          0.9935 |   0.9932 |       0.9938 |          0.9927 |       275.92s |           1.74s |
| **distilbert-base-uncased** |   0.9935 |      0.9935 |          0.9935 |             0.9936 |   0.9935 |       0.9935 |          0.9935 |   0.9932 |       0.9938 |          0.9927 |    **46.27s** |       **365ms** |
| **diptanu/fBERT**           |   0.9935 |      0.9935 |          0.9935 |             0.9936 |   0.9935 |       0.9935 |          0.9935 |   0.9932 |       0.9938 |          0.9927 |        81.92s |           634ms |
| **GroNLP/hateBERT**         |   0.9902 |      0.9902 |          0.9902 |             0.9904 |   0.9902 |       0.9902 |          0.9902 |   0.9901 |       0.9903 |          0.9899 |        81.93s |           637ms |
| **roberta-large**           |   0.9837 |      0.9837 |          0.9837 |             0.9839 |   0.9837 |       0.9837 |          0.9837 |   0.9832 |       0.9821 |          0.9845 |       281.49s |           1.74s |

### B) Evaluation Metrics

| Model                       | Macro F1 |  Eval Loss | Eval Runtime |
| --------------------------- | -------: | ---------: | -----------: |
| **bert-base-uncased**       |   0.9933 |     0.0164 |        650ms |
| **bert-large-uncased**      |   0.9932 | **0.0097** |        1.74s |
| **distilbert-base-uncased** |   0.9932 |     0.0178 |    **365ms** |
| **diptanu/fBERT**           |   0.9932 |     0.0152 |        634ms |
| **GroNLP/hateBERT**         |   0.9901 |     0.0207 |        637ms |
| **roberta-large**           |   0.9832 |     0.0293 |        1.74s |

**Highlights**

* All BERT-family models (base/large/distil/fBERT) score \~**0.993 macro F1**.
* **DistilBERT** is the **fastest** (training and evaluation).
* **BERT-large** has the **lowest eval loss**, but is much slower.
* **RoBERTa-large** lags in both accuracy/F1 and speed on this task.
* **hateBERT** performs well but slightly below the top group.

> Times were measured on A100; absolute numbers may vary by hardware.

---

## Recommendations

* **Best overall (accuracy + efficiency):** `bert-base-uncased`
* **Best efficiency (speed first):** `distilbert-base-uncased`
* **Best if time isn’t a constraint:** `bert-large-uncased`
* **Less suitable here:** `roberta-large` (slower & lower F1), `GroNLP/hateBERT` (slightly lower than top group)

---

## 🚀 Getting Started

### Install

```bash
git clone https://github.com/DrishtiShrrrma/multiclass-classification-transformer-model-comparison.git
cd multiclass-classification-transformer-model-comparison
pip install -r requirements.txt
# optional for imports in notebooks:
pip install -e .
```

### Prepare data

```bash
python scripts/prepare_dataset.py --input_dir /path/to/kaggle_csvs --out data/merged_dataset.csv
```

### Train (via YAML config)

```bash
python -m hate_speech_eval.train --config config/config.yaml
```

or override on the CLI:

```bash
python -m hate_speech_eval.train \
  --model_name bert-base-uncased --epochs 5 --batch_size 16 \
  --lr 2e-5 --max_length 114 --data data/merged_dataset.csv \
  --out runs/bert-base-uncased
```

### Evaluate / quick predictions

```bash
python -m hate_speech_eval.evaluate --model_name bert-base-uncased --ckpt runs/bert-base-uncased --csv data/merged_dataset.csv
```

### Generate figures

```bash
# From CSV (no training needed)
python scripts/make_figures.py --csv data/merged_dataset.csv

# From a finished run
python scripts/make_curves_from_state.py --state runs/bert-base-uncased/trainer_state.json --prefix bert_base
```

---

if you want, I can also add a tiny `results/metrics_table.csv` generator so your README table stays in sync with saved `metrics_*.json`.
