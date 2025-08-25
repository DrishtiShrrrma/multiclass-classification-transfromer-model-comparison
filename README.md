

# Comprehensive Evaluation of Transformer Models for Hate Speech Detection

This repository contains the code and experiments for the research project **“Comprehensive Evaluation of Various Transformer Models in Detecting Normal, Hate, and Offensive Text”**.

The project benchmarks multiple transformer-based architectures on a curated Kaggle dataset to evaluate their performance in classifying tweets into **Normal**, **Hate**, and **Offensive** categories.

---

## 📌 Objective

The aim of this project is to assess how well different transformer models perform in detecting offensive and hateful speech on social media. The models are compared based on **accuracy, F1 scores, recall, precision, evaluation loss, and training efficiency**.

---

## 📂 Repository Structure

```
├── data/              # Preprocessed dataset
├── notebooks/         # Jupyter notebooks with training/evaluation experiments
├── src/               # Core training and evaluation scripts
│   ├── dataset.py     # Dataset preprocessing
│   ├── trainer.py     # CustomTrainer implementation
│   ├── train.py       # Training pipeline
│   └── eval.py        # Evaluation metrics
├── results/           # Model metrics, logs, and plots
└── README.md          # Project overview
```

---

## 🔍 Dataset

* **Source:** Combination of 12 Kaggle datasets.
* **Classes:** `Normal`, `Hate`, `Offensive`.
* **Merging & Filtering:** Datasets were merged and filtered by text length per label.
* **Preprocessing Choices:**

  * No lowercasing (case carries semantic meaning).
  * No removal of special characters (important for obfuscated offensive text).
  * Applied label-specific filtering.

---

## ⚙️ Experimental Setup

* **Hardware:** NVIDIA A100 GPU.

* **Training Parameters:**

  * Epochs: `5`
  * Batch Size: `16`
  * Learning Rate: `2e-5`
  * Max Sequence Length: `~114` (1.3 × longest sequence)

* **Class Imbalance Handling:**
  Used a **CustomTrainer** with normalized class weights in the `CrossEntropyLoss` function.

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

### 🔹 Training Metrics

| Model                  | Accuracy | Micro F1 | Macro F1 | Training Time |
| ---------------------- | -------- | -------- | -------- | ------------- |
| **bert-base-uncased**  | \~0.99   | \~0.99   | \~0.99   | Moderate      |
| **bert-large-uncased** | \~0.99   | \~0.99   | \~0.99   | Slow          |
| **distilbert-base**    | \~0.98   | \~0.98   | \~0.98   | **Fastest**   |
| **diptanu/fBERT**      | \~0.99   | \~0.99   | \~0.99   | Moderate      |
| **GroNLP/hateBERT**    | \~0.98   | \~0.98   | \~0.98   | Moderate      |
| **roberta-large**      | \~0.97   | \~0.97   | \~0.96   | **Slowest**   |

---

### 🔹 Evaluation Metrics

| Model                  | Macro F1  | Precision | Recall    | Eval Loss        | Eval Runtime |
| ---------------------- | --------- | --------- | --------- | ---------------- | ------------ |
| **bert-base-uncased**  | **0.99+** | 0.99+     | 0.99+     | Low              | Moderate     |
| **bert-large-uncased** | **0.99+** | 0.99+     | 0.99+     | **Lowest**       | Slow         |
| **distilbert-base**    | 0.98–0.99 | 0.98–0.99 | 0.98–0.99 | Higher than base | **Fastest**  |
| **diptanu/fBERT**      | 0.99      | 0.99      | 0.99      | Slightly higher  | Moderate     |
| **GroNLP/hateBERT**    | 0.98–0.99 | 0.98      | 0.98      | Moderate         | Moderate     |
| **roberta-large**      | \~0.97    | \~0.97    | \~0.97    | Higher           | Slowest      |

---

## ✅ Recommendations

* **Best Overall Model:** `bert-base-uncased`
* **Best for Efficiency:** `distilbert-base-uncased`
* **Best for Maximum Performance (time-insensitive):** `bert-large-uncased`
* **Models to Avoid:** `roberta-large`, `GroNLP/hateBERT`

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/DrishtiShrrrma/multiclass-classification-transfromer-model-comparison.git
cd multiclass-classification-transfromer-model-comparison
pip install -r requirements.txt
```

### Training a Model

```bash
python src/train.py --model_name bert-base-uncased --epochs 5 --batch_size 16
```

### Evaluating a Model

```bash
python src/eval.py --model_name bert-base-uncased
```

---

