#!/usr/bin/env python
"""
Generate exploratory figures from the merged dataset:
- results/label_distribution.png
- results/length_by_label.png
"""
import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/merged_dataset.csv")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv).dropna(subset=["text","label"]).copy()

    # Label distribution
    ax = df["label"].value_counts().sort_index().plot(kind="bar")
    ax.set_xlabel("Label"); ax.set_ylabel("Count"); ax.set_title("Label Distribution")
    plt.tight_layout(); plt.savefig(f"{args.outdir}/label_distribution.png", dpi=160)
    plt.clf()

    # Length by label (word count)
    df["len_words"] = df["text"].str.split().str.len()
    df.boxplot(column="len_words", by="label", grid=False)
    plt.title("Text Length by Label (words)"); plt.suptitle("")
    plt.xlabel("Label"); plt.ylabel("Words")
    plt.tight_layout(); plt.savefig(f"{args.outdir}/length_by_label.png", dpi=160)
    plt.clf()

    print("Saved:",
          f"{args.outdir}/label_distribution.png",
          f"{args.outdir}/length_by_label.png", sep="\n- ")

if __name__ == "__main__":
    main()
