#!/usr/bin/env python
"""
Aggregate results/metrics_*.json into:
- results/metrics_table.csv
- results/metrics_training.md   (training-style table)
- results/metrics_eval.md       (eval table like in the README)
"""
import argparse, glob, json, pandas as pd
from pathlib import Path

def fmt_time(sec):
    if sec is None:
        return ""
    if sec < 1:
        return f"{int(sec*1000)}ms"
    return f"{sec:.2f}s"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="results/metrics_*.json")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    rows = []
    for path in glob.glob(args.pattern):
        with open(path) as f:
            m = json.load(f)
        rows.append(m)

    if not rows:
        raise SystemExit("No metrics_*.json files found in results/. Run training first.")

    df = pd.DataFrame(rows)

    # CSV (all columns)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = f"{args.out_dir}/metrics_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    # Training-style table (matches your big table)
    train_cols = [
        "model","accuracy",
        "weighted_f1","weighted_recall","weighted_precision",
        "micro_f1","micro_recall","micro_precision",
        "macro_f1","macro_recall","macro_precision",
        "training_time_sec","evaluation_time_sec"
    ]
    tdf = df[train_cols].copy()
    tdf["training_time"] = tdf["training_time_sec"].apply(fmt_time)
    tdf["evaluation_time"] = tdf["evaluation_time_sec"].apply(fmt_time)
    tdf = tdf.drop(columns=["training_time_sec","evaluation_time_sec"])

    md_train = "| " + " | ".join([
        "Model","Accuracy","Weighted F1","Weighted Recall","Weighted Precision",
        "Micro F1","Micro Recall","Micro Precision",
        "Macro F1","Macro Recall","Macro Precision",
        "Training Time","Evaluation Time"
    ]) + " |\n" + \
    "|" + "----|"*13 + "\n"

    for _, r in tdf.sort_values("macro_f1", ascending=False).iterrows():
        md_train += f"| {r['model']} | {r['accuracy']:.4f} | {r['weighted_f1']:.4f} | {r['weighted_recall']:.4f} | {r['weighted_precision']:.4f} | {r['micro_f1']:.4f} | {r['micro_recall']:.4f} | {r['micro_precision']:.4f} | {r['macro_f1']:.4f} | {r['macro_recall']:.4f} | {r['macro_precision']:.4f} | {r['training_time']} | {r['evaluation_time']} |\n"

    train_md_path = f"{args.out_dir}/metrics_training.md"
    with open(train_md_path, "w") as f:
        f.write(md_train)
    print(f"[saved] {train_md_path}")

    # Eval table (matches your smaller table)
    edf = df[["model","macro_f1","eval_loss","eval_runtime"]].copy()
    edf["eval_runtime"] = edf["eval_runtime"].apply(lambda x: f"{x:.2f}s" if isinstance(x,(int,float)) else str(x))

    md_eval = "| Model | Macro F1 Score | Eval Loss | Eval Runtime |\n|------|------:|------:|------:|\n"
    for _, r in edf.sort_values("macro_f1", ascending=False).iterrows():
        md_eval += f"| {r['model']} | {r['macro_f1']:.4f} | {r['eval_loss'] if r['eval_loss'] is not None else ''} | {r['eval_runtime']} |\n"

    eval_md_path = f"{args.out_dir}/metrics_eval.md"
    with open(eval_md_path, "w") as f:
        f.write(md_eval)
    print(f"[saved] {eval_md_path}")

if __name__ == "__main__":
    main()
