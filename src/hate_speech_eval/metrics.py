from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    accuracy_score,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)

    # accuracy
    acc = accuracy_score(labels, preds)

    # macro (unweighted)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    # weighted (by support)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    # micro
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )

    return {
        "accuracy": acc,
        # macro
        "macro_f1": f1_macro,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        # weighted
        "weighted_f1": f1_w,
        "weighted_precision": p_w,
        "weighted_recall": r_w,
        # micro
        "micro_f1": f1_micro,
        "micro_precision": p_micro,
        "micro_recall": r_micro,
    }
