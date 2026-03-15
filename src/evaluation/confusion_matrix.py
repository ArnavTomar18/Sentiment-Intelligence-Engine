"""
Confusion matrix plotting utilities.
"""

from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    class_names: List[str] | None = None,
    normalize: str | None = None,
    title: str = "Confusion Matrix",
):
    y_true = np.array(list(y_true))
    y_pred = np.array(list(y_pred))
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format=".2f" if normalize else "d")
    plt.title(title)
    plt.tight_layout()
    plt.show()

