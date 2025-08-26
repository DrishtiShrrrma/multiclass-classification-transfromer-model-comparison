import argparse, os
from transformers import AutoModelForSequenceClassification, TrainingArguments
from .data import DataConfig, load_dataset, tokenize_dataset
from .metrics import compute_metrics
from .trainer import CustomTrainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=114)
    ap.add_argument("--data", default="data/merged_dataset.csv")
    ap.add_argument("--out", default="runs/bert-base-uncased")
    args = ap.parse_args()

    # Load + split + tokenize
    dcfg = DataConfig(path=args.data, max_length=args.max_length)
    ds = load_dataset(dcfg.path)
    dsd = ds.train_test_split(test_size=0.2, seed=42)
    dsd, tok = tokenize_dataset(dsd, args.model_name, dcfg.max_length)

    # Compute class weights from train labels
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

    trainer.train()
    print(trainer.evaluate())

if __name__ == "__main__":
    main()
