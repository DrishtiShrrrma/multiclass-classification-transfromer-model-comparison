import argparse, os, yaml, json, time, re
from transformers import AutoModelForSequenceClassification, TrainingArguments
from .data import DataConfig, load_dataset, tokenize_dataset
from .metrics import compute_metrics
from .trainer import CustomTrainer

def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Path to config/config.yaml")
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=114)
    ap.add_argument("--data", default="data/merged_dataset.csv")
    ap.add_argument("--out", default="runs/bert-base-uncased")
    args = ap.parse_args()
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
    return args

def main():
    args = parse_args()

    # Load + split + tokenize
    dcfg = DataConfig(path=args.data, max_length=args.max_length)
    ds = load_dataset(dcfg.path)
    dsd = ds.train_test_split(test_size=0.2, seed=42)
    dsd, tok = tokenize_dataset(dsd, args.model_name, dcfg.max_length)

    # Class weights (from train split)
    y = dsd["train"]["labels"]
    n0 = sum(int(v == 0) for v in y)
    n1 = sum(int(v == 1) for v in y)
    n2 = sum(int(v == 2) for v in y)
    total = n0 + n1 + n2
    import torch
    cw = torch.tensor([total/n0, total/n1, total/n2], dtype=torch.float32)
    cw = cw / cw.sum()

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=3
    )

    os.makedirs(args.out, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    targs = TrainingArguments(
        output_dir=args.out,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to="none",
        logging_steps=50,
    )

    trainer = CustomTrainer(
        model=model,
        args=targs,
        train_dataset=dsd["train"],
        eval_dataset=dsd["test"],
        tokenizer=tok,
        compute_metrics=compute_metrics,
        class_weights=cw,
    )

    # ---- timing
    t0 = time.perf_counter()
    trainer.train()
    train_time_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    eval_out = trainer.evaluate()
    eval_time_sec = time.perf_counter() - t1

    # collect + save
    model_slug = _slug(args.model_name)
    metrics = {
        "model": args.model_name,
        "accuracy": eval_out.get("eval_accuracy") or eval_out.get("accuracy"),
        "weighted_f1": eval_out.get("eval_weighted_f1") or eval_out.get("weighted_f1"),
        "weighted_recall": eval_out.get("eval_weighted_recall") or eval_out.get("weighted_recall"),
        "weighted_precision": eval_out.get("eval_weighted_precision") or eval_out.get("weighted_precision"),
        "micro_f1": eval_out.get("eval_micro_f1") or eval_out.get("micro_f1"),
        "micro_recall": eval_out.get("eval_micro_recall") or eval_out.get("micro_recall"),
        "micro_precision": eval_out.get("eval_micro_precision") or eval_out.get("micro_precision"),
        "macro_f1": eval_out.get("eval_macro_f1") or eval_out.get("macro_f1"),
        "macro_recall": eval_out.get("eval_macro_recall") or eval_out.get("macro_recall"),
        "macro_precision": eval_out.get("eval_macro_precision") or eval_out.get("macro_precision"),
        "eval_loss": eval_out.get("eval_loss"),
        "eval_runtime": eval_out.get("eval_runtime"),
        "training_time_sec": round(train_time_sec, 2),
        "evaluation_time_sec": round(eval_time_sec, 2),
    }
    out_path = f"results/metrics_{model_slug}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[saved] {out_path}\n", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
