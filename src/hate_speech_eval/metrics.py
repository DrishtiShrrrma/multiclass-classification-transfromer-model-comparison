from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    micro = f1_score(labels, preds, average="micro")
    return {
        "accuracy": acc,
        "macro_f1": f1,
        "macro_precision": p,
        "macro_recall": r,
        "micro_f1": micro,
    }
