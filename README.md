

# Comprehensive Evaluation of Transformer Models for Hate Speech Detection

This repository contains the code and experiments for the research project **“Comprehensive Evaluation of Various Transformer Models in Detecting Normal, Hate, and Offensive Text.”**
It benchmarks several transformer architectures on a curated Kaggle dataset to classify tweets into **Normal**, **Hate**, and **Offensive** categories.

---

## 📌 Objective

Evaluate and compare transformer models for offensive/hate speech detection on social media. Models are compared on **accuracy, F1 (macro/micro), precision, recall, evaluation loss, runtime,** and **training efficiency**.

---

## 📂 Repository Structure

```
├── config/
│   └── config.yaml                 # Default hyperparameters & paths
├── data/
│   └── README.md                   # Dataset source & expected schema (put merged CSV here)
├── notebooks/                      # Your exploratory notebooks
├── results/                        # Plots, confusion matrices, aggregated metrics
├── scripts/
│   ├── prepare_dataset.py          # Merge the 12 Kaggle CSVs → data/merged_dataset.csv
│   ├── make_figures.py             # Label distribution & length-by-label plots
│   └── make_curves_from_state.py   # (optional) Plot curves from trainer_state.json
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

  * Keep case (no lowercasing) and **retain special characters** (important for obfuscation patterns).
  * Label-specific length filtering (as explored in the article).
  * Tokenizer max length ≈ **114** (1.3× longest sequence in Normal).

---

## ⚙️ Experimental Setup

* **Hardware:** NVIDIA A100 GPU (experiments run on A100 in the study; any CUDA GPU works)
* **Key hyperparameters (defaults):**

  * Epochs: `5`
  * Batch size: `16`
  * Learning rate: `2e-5`
  * Max length: `114`
* **Class imbalance:** weighted `CrossEntropyLoss` via **CustomTrainer**.

> These can be edited in `config/config.yaml` or overridden via CLI flags.

---

## 🤖 Models Evaluated

* `bert-base-uncased`
* `bert-large-uncased`
* `distilbert-base-uncased`
* `diptanu/fBERT`
* `GroNLP/hateBERT`
* `roberta-large`

---

## 📊 Results

### Training Metrics (illustrative summary)

| Model                  | Accuracy | Micro F1 | Macro F1 | Training Time |
| ---------------------- | -------: | -------: | -------: | ------------- |
| **bert-base-uncased**  |   \~0.99 |   \~0.99 |   \~0.99 | Moderate      |
| **bert-large-uncased** |   \~0.99 |   \~0.99 |   \~0.99 | Slow          |
| **distilbert-base**    |   \~0.98 |   \~0.98 |   \~0.98 | **Fastest**   |
| **diptanu/fBERT**      |   \~0.99 |   \~0.99 |   \~0.99 | Moderate      |
| **GroNLP/hateBERT**    |   \~0.98 |   \~0.98 |   \~0.98 | Moderate      |
| **roberta-large**      |   \~0.97 |   \~0.97 |   \~0.96 | **Slowest**   |

### Evaluation Metrics (illustrative summary)

| Model                  |  Macro F1 | Precision |    Recall | Eval Loss        | Eval Runtime |
| ---------------------- | --------: | --------: | --------: | ---------------- | -----------: |
| **bert-base-uncased**  | **0.99+** |     0.99+ |     0.99+ | Low              |     Moderate |
| **bert-large-uncased** | **0.99+** |     0.99+ |     0.99+ | **Lowest**       |         Slow |
| **distilbert-base**    | 0.98–0.99 | 0.98–0.99 | 0.98–0.99 | Higher than base |  **Fastest** |
| **diptanu/fBERT**      |      0.99 |      0.99 |      0.99 | Slightly higher  |     Moderate |
| **GroNLP/hateBERT**    | 0.98–0.99 |      0.98 |      0.98 | Moderate         |     Moderate |
| **roberta-large**      |    \~0.97 |    \~0.97 |    \~0.97 | Higher           |      Slowest |

> Replace these with your exact run outputs when you log or export metrics to `results/`.

---

## ✅ Recommendations

* **Best overall:** `bert-base-uncased`
* **Best efficiency:** `distilbert-base-uncased`
* **Best if time is not a constraint:** `bert-large-uncased`
* **Less suitable here:** `roberta-large`, `GroNLP/hateBERT`

---

## 🚀 Getting Started

### 1) Install dependencies

```bash
pip install -r requirements.txt
# (optional) developer mode
pip install -e .
```

### 2) Prepare data

```bash
python scripts/prepare_dataset.py --input_dir /path/to/kaggle_csvs --out data/merged_dataset.csv
```

### 3) Train (via YAML config)

```bash
python -m hate_speech_eval.train --config config/config.yaml
```

> Or override on the CLI:

```bash
python -m hate_speech_eval.train \
  --model_name bert-base-uncased --epochs 5 --batch_size 16 \
  --lr 2e-5 --max_length 114 --data data/merged_dataset.csv \
  --out runs/bert-base-uncased
```

### 4) Evaluate / quick predictions

```bash
python -m hate_speech_eval.evaluate --model_name bert-base-uncased --ckpt runs/bert-base-uncased --csv data/merged_dataset.csv
```

### 5) Generate figures (no training needed)

```bash
python scripts/make_figures.py --csv data/merged_dataset.csv
# outputs: results/label_distribution.png, results/length_by_label.png
```

### 6) Plot training curves (after a run)

```bash
python scripts/make_curves_from_state.py \
  --state runs/bert-base-uncased/trainer_state.json \
  --prefix bert_base
# outputs under results/: bert_base_eval_loss.png, bert_base_eval_macro_f1.png, etc.
```

---


If you want, I can now slot in a **Results** section that auto-updates from `results/metrics_*.json` (via a tiny aggregator script) and add image references for your plots.
