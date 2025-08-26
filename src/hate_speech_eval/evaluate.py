import argparse
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--ckpt", default=None, help="Path to trained checkpoint (e.g., runs/bert-base-uncased)")
    ap.add_argument("--csv", default="data/merged_dataset.csv")
    ap.add_argument("--max_length", type=int, default=114)
    args = ap.parse_args()

    # Read just text to demo predictions; labels are optional here
    df = pd.read_csv(args.csv).dropna(subset=["text"])
    ds = Dataset.from_pandas(df[["text"]], preserve_index=False)

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    def _t(batch): return tok(batch["text"], truncation=True, max_length=args.max_length)
    ds = ds.map(_t, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt or args.model_name)
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(ds), 32):
            b = ds[i:i+32]
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
            preds.append(out.logits.argmax(-1))
    preds = torch.cat(preds).cpu().tolist()
    print("First 10 predictions:", preds[:10])

if __name__ == "__main__":
    main()
