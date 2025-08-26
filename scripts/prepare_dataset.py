
"""
Utility script to merge the 12 Kaggle CSVs into a single dataset
with columns: text, label.

Usage:
    python scripts/prepare_dataset.py --input_dir /path/to/kaggle_csvs --out data/merged_dataset.csv
"""
import argparse, glob, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory containing the 12 Kaggle CSVs")
    ap.add_argument("--out", default="data/merged_dataset.csv")
    args = ap.parse_args()

    files = glob.glob(f"{args.input_dir}/*.csv")
    if not files:
        raise SystemExit("❌ No CSV files found in input_dir")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # find appropriate column names (some Kaggle CSVs differ)
        col_text = "text" if "text" in df.columns else ("tweet" if "tweet" in df.columns else None)
        col_label = "label" if "label" in df.columns else ("class" if "class" in df.columns else None)
        if not col_text or not col_label:
            print(f"⚠️ Skipping {f}: missing text/label columns")
            continue
        dfs.append(df[[col_text, col_label]].rename(columns={col_text: "text", col_label: "label"}))

    if not dfs:
        raise SystemExit("❌ No compatible CSVs found.")

    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(args.out, index=False)
    print(f"✅ Written {args.out} with {len(merged)} rows.")

if __name__ == "__main__":
    main()
