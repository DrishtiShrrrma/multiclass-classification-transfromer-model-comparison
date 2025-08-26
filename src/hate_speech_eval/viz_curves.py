import os
import matplotlib.pyplot as plt

def _plot(xs, ys, xlabel, ylabel, title, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

def save_trainer_curves(log_history, out_dir="results", prefix="run"):
    """
    Saves common curves derived from Hugging Face Trainer's log_history:
      - training loss (per logging step)
      - eval loss (per epoch)
      - eval macro_f1 (per epoch)  if present
      - eval accuracy (per epoch)  if present

    Args:
        log_history: trainer.state.log_history (list of dicts)
        out_dir: where to save PNGs
        prefix: filename prefix (e.g., model name)
    """
    # training loss (has "loss" and "step")
    train_steps, train_loss = [], []
    # eval metrics (have "epoch" and keys like "eval_loss", "eval_macro_f1")
    eval_epochs, eval_loss, eval_macro_f1, eval_accuracy = [], [], [], []

    for rec in log_history:
        if "loss" in rec and "step" in rec:
            train_steps.append(rec["step"])
            train_loss.append(rec["loss"])
        if "epoch" in rec and any(k.startswith("eval_") for k in rec.keys()):
            eval_epochs.append(rec["epoch"])
            if "eval_loss" in rec:       eval_loss.append(rec["eval_loss"])
            if "eval_macro_f1" in rec:   eval_macro_f1.append(rec["eval_macro_f1"])
            if "eval_accuracy" in rec:   eval_accuracy.append(rec["eval_accuracy"])

    if train_steps:
        _plot(train_steps, train_loss, "Step", "Training loss",
              f"{prefix}: Training loss", f"{out_dir}/{prefix}_train_loss.png")

    if eval_epochs and eval_loss:
        _plot(eval_epochs, eval_loss, "Epoch", "Eval loss",
              f"{prefix}: Eval loss", f"{out_dir}/{prefix}_eval_loss.png")

    if eval_epochs and eval_macro_f1:
        _plot(eval_epochs, eval_macro_f1, "Epoch", "Eval Macro F1",
              f"{prefix}: Eval Macro F1", f"{out_dir}/{prefix}_eval_macro_f1.png")

    if eval_epochs and eval_accuracy:
        _plot(eval_epochs, eval_accuracy, "Epoch", "Eval Accuracy",
              f"{prefix}: Eval Accuracy", f"{out_dir}/{prefix}_eval_accuracy.png")
