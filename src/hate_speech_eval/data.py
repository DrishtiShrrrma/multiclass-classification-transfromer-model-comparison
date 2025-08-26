from dataclasses import dataclass
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from .config import LABEL2ID, DEFAULT_MAX_LENGTH

@dataclass
class DataConfig:
    path: str = "data/merged_dataset.csv"
    max_length: int = DEFAULT_MAX_LENGTH

def load_dataset(df_or_path):
    """
    Loads a DataFrame or CSV into a Hugging Face Dataset with 'text' and 'labels' columns.
    """
    if isinstance(df_or_path, str):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path
    assert {"text", "label"} <= set(df.columns), "CSV must have 'text' and 'label' columns."
    df = df.dropna(subset=["text", "label"]).copy()
    df["labels"] = df["label"].map(LABEL2ID)
    return Dataset.from_pandas(df[["text", "labels"]], preserve_index=False)

def tokenize_dataset(dset, model_name: str, max_length: int = DEFAULT_MAX_LENGTH):
    """
    Tokenizes a (possibly split) DatasetDict or Dataset and returns (tokenized_dataset, tokenizer).
    """
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def _tok(batch):
        return tok(batch["text"], truncation=True, max_length=max_length)

    # Handle both Dataset and DatasetDict
    if hasattr(dset, "keys") and callable(dset.keys):
        # DatasetDict: map on each split
        dset = dset.map(_tok, batched=True)
        for split in dset.keys():
            dset[split].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    else:
        # Single Dataset
        dset = dset.map(_tok, batched=True)
        dset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return dset, tok
