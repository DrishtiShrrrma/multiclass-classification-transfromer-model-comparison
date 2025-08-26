# ===== Makefile =====
SHELL := /bin/bash
PY    ?= python

# paths
CSV    ?= data/merged_dataset.csv
CONFIG ?= config/config.yaml

# hyperparams (override on CLI if you like)
EPOCHS ?= 5
BATCH  ?= 16
LR     ?= 2e-5
MAXLEN ?= 114

# list of models to benchmark
MODELS := bert-base-uncased \
          bert-large-uncased \
          distilbert-base-uncased \
          diptanu/fBERT \
          GroNLP/hateBERT \
          roberta-large

.PHONY: help data figures train train-all eval curves curves-all aggregate clean

help:
	@echo "Targets:"
	@echo "  make data IN=/path/to/kaggle_csvs   # merge 12 CSVs -> $(CSV)"
	@echo "  make figures                        # label distribution & length-by-label plots"
	@echo "  make train MODEL=bert-base-uncased  # train a single model"
	@echo "  make train-all                      # train all models in MODELS"
	@echo "  make eval  MODEL=bert-base-uncased  # quick predictions using ckpt"
	@echo "  make curves MODEL=bert-base-uncased # plots from trainer_state.json"
	@echo "  make curves-all                     # curves for all models that have runs/"
	@echo "  make aggregate                      # builds results/metrics_table.csv & .md tables"
	@echo "  make clean                          # remove generated plots/metrics (keeps runs/)"

data:
	@test -n "$(IN)" || (echo "Usage: make data IN=/path/to/kaggle_csvs"; exit 1)
	$(PY) scripts/prepare_dataset.py --input_dir "$(IN)" --out "$(CSV)"

figures:
	@test -f "$(CSV)" || (echo "Missing $(CSV). Run: make data IN=..."; exit 1)
	$(PY) scripts/make_figures.py --csv "$(CSV)"

# Train one model; OUT dir slugifies the model name
train:
	@test -f "$(CSV)" || (echo "Missing $(CSV). Run: make data IN=..."; exit 1)
	@test -n "$(MODEL)" || (echo "Usage: make train MODEL=bert-base-uncased"; exit 1)
	OUT=$$(echo "$(MODEL)" | sed 's/[^A-Za-z0-9._-]/_/g'); \
	$(PY) -m hate_speech_eval.train \
	  --model_name "$(MODEL)" --epochs $(EPOCHS) --batch_size $(BATCH) \
	  --lr $(LR) --max_length $(MAXLEN) --data "$(CSV)" --out runs/$$OUT

# Train all models defined above
train-all:
	@test -f "$(CSV)" || (echo "Missing $(CSV). Run: make data IN=..."; exit 1)
	for m in $(MODELS); do \
	  out=$$(echo $$m | sed 's/[^A-Za-z0-9._-]/_/g'); \
	  echo "=== Training $$m -> runs/$$out ==="; \
	  $(PY) -m hate_speech_eval.train \
	    --model_name "$$m" --epochs $(EPOCHS) --batch_size $(BATCH) \
	    --lr $(LR) --max_length $(MAXLEN) --data "$(CSV)" --out runs/$$out; \
	done

# Quick predictions using a trained checkpoint
eval:
	@test -n "$(MODEL)" || (echo "Usage: make eval MODEL=bert-base-uncased"; exit 1)
	OUT=$$(echo "$(MODEL)" | sed 's/[^A-Za-z0-9._-]/_/g'); \
	$(PY) -m hate_speech_eval.evaluate --model_name "$(MODEL)" --ckpt runs/$$OUT --csv "$(CSV)"

# Curves for a single model from its trainer_state.json
curves:
	@test -n "$(MODEL)" || (echo "Usage: make curves MODEL=bert-base-uncased"; exit 1)
	OUT=$$(echo "$(MODEL)" | sed 's/[^A-Za-z0-9._-]/_/g'); \
	$(PY) scripts/make_curves_from_state.py --state runs/$$OUT/trainer_state.json --prefix $$OUT

# Curves for all models that have trainer_state.json
curves-all:
	for m in $(MODELS); do \
	  out=$$(echo $$m | sed 's/[^A-Za-z0-9._-]/_/g'); \
	  st="runs/$$out/trainer_state.json"; \
	  if [ -f "$$st" ]; then \
	    echo "=== Curves for $$m ==="; \
	    $(PY) scripts/make_curves_from_state.py --state "$$st" --prefix $$out; \
	  else \
	    echo "Skipping $$m (no $$st)"; \
	  fi; \
	done

aggregate:
	$(PY) scripts/aggregate_metrics.py

clean:
	rm -f results/*.png results/metrics_*.json results/metrics_*.md results/metrics_table.csv
	@echo "Cleaned generated results (kept runs/)."
