

# Comprehensive Evaluation of Transformer Models for Hate Speech Detection

This repository contains the code and experiments for the research project **“Comprehensive Evaluation of Various Transformer Models in Detecting Normal, Hate, and Offensive Text”**.

The project benchmarks multiple transformer-based architectures on a curated Kaggle dataset to evaluate their performance in classifying tweets into **Normal**, **Hate**, and **Offensive** categories.

---

## 📂 Repository Structure

```
├── data/              # Preprocessed dataset (not included, due to size)
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

## 📌 Objective

The aim of this project is to assess how well different transformer models perform in detecting offensive and hateful speech on social media. The models are compared based on **accuracy, F1 scores, recall, precision, evaluation loss, and training efficiency**.

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

### Training Observations

* **Accuracy & Micro Metrics:** Most models performed similarly.
* **Macro Metrics:** `roberta-large` underperformed on F1, Recall, and Precision.
* **Training Time:**

  * Fastest → `distilbert-base-uncased`
  * Slowest → `bert-large-uncased`, `roberta-large`

### Evaluation Observations

* **Macro F1 Score:** All models scored **>0.98**.
* **Best Balance:** `bert-base-uncased` (top performance with efficiency).
* **Efficiency Winner:** `distilbert-base-uncased`.
* **Highest Performance (time-insensitive):** `bert-large-uncased`.
* **Less Suitable Models:** `roberta-large`, `hateBERT`.

---

## ✅ Recommendations

* **Best Overall Model:** `bert-base-uncased`
* **Best for Efficiency:** `distilbert-base-uncased`
* **Best for Maximum Performance:** `bert-large-uncased`
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

Do you want me to also **insert a results table** (instead of just text summaries) so readers can quickly compare models’ F1/Precision/Recall in the README? That usually makes research repos look more professional.
