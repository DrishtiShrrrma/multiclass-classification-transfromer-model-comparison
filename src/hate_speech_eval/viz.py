import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def save_confusion_matrix(y_true, y_pred, labels, out_path):
    """
    y_true, y_pred: 1D arrays of label ids (0..n-1)
    labels: list of label names (e.g., ["Normal","Hate","Offensive"])
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(values_format="d", xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
