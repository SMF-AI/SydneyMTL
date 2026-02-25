from __future__ import annotations

import warnings
from itertools import cycle
from typing import Tuple, List, Optional

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
import seaborn as sns
from scipy.interpolate import interp1d


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
    """Plot ROC curve for binary classification."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, lw=2, label=f"AUROC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")

    return fig, ax


def plot_roc_multiclass(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot ROC curves for multi-class classification."""
    unique_classes = np.unique(y_true)
    n_classes = y_prob.shape[1]

    if len(unique_classes) < 2:
        fig, ax = plt.subplots()
        ax.text(
            0.5, 0.5, "Insufficient classes for ROC curve", ha="center", va="center"
        )
        return fig, ax

    y_true_onehot = np.eye(n_classes)[y_true]

    fpr, tpr, roc_auc = {}, {}, {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_onehot.ravel(), y_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        for i in unique_classes:
            fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(
        fpr["micro"],
        tpr["micro"],
        linestyle=":",
        linewidth=4,
        label=f'micro-average (AUC = {roc_auc["micro"]:.2f})',
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(unique_classes, colors):
        ax.plot(
            fpr[i],
            tpr[i],
            color=color,
            label=f"Class {i} (AUC = {roc_auc[i]:.2f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=2)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")

    return fig, ax


def plot_prc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
    """Plot Precision-Recall curve for binary classification."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, lw=2, label=f"PRAUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")

    return fig, ax


def plot_prc_multiclass(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot Precision-Recall curves for multi-class classification."""
    unique_classes = np.unique(y_true)
    n_classes = y_prob.shape[1]
    y_true_onehot = np.eye(n_classes)[y_true]

    precision, recall, average_precision = {}, {}, {}

    fig, ax = plt.subplots(figsize=(10, 8))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if len(unique_classes) >= 2:
            precision["micro"], recall["micro"], _ = precision_recall_curve(
                y_true_onehot.ravel(), y_prob.ravel()
            )
            average_precision["micro"] = average_precision_score(
                y_true_onehot, y_prob, average="micro"
            )

            ax.plot(
                recall["micro"],
                precision["micro"],
                linestyle=":",
                linewidth=4,
                label=f'micro-average (AP = {average_precision["micro"]:.2f})',
            )

        colors = cycle(["aqua", "darkorange", "cornflowerblue"])
        for i in unique_classes:
            precision[i], recall[i], _ = precision_recall_curve(
                y_true_onehot[:, i], y_prob[:, i]
            )
            average_precision[i] = average_precision_score(
                y_true_onehot[:, i], y_prob[:, i]
            )

            ax.plot(
                recall[i],
                precision[i],
                color=next(colors),
                label=f"Class {i} (AP = {average_precision[i]:.2f})",
            )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left")

    return fig, ax


def plot_confusion_matrix(
    y_true: NDArray[int],
    y_pred: NDArray[int],
    labels: Optional[List[str]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    return fig, ax


def plot_confusion_matrix_multiclass(
    y_true: NDArray[int],
    y_pred: NDArray[int],
    normalize: bool = False,
    title: Optional[str] = None,
    labels: Optional[List[str]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot multi-class confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=(10, 8))

    if labels is None:
        labels = [f"Class {i}" for i in range(cm.shape[0])]

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title if title else "Confusion Matrix")

    return fig, ax


def plot_cv_auroc(
    fold_y_trues: List[np.ndarray],
    fold_y_probs: List[np.ndarray],
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot cross-validation ROC curves with mean curve."""
    fig, ax = plt.subplots()
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "purple"])

    all_fpr, all_tpr = [], []

    for i, (y_true, y_prob) in enumerate(zip(fold_y_trues, fold_y_probs)):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        all_fpr.append(fpr)
        all_tpr.append(tpr)

        ax.plot(fpr, tpr, color=next(colors), label=f"Fold {i+1} (AUC = {roc_auc:.2f})")

    max_length = max(len(arr) for arr in all_fpr)
    interp_fpr = [
        interp1d(np.arange(len(fpr)), fpr)(np.linspace(0, len(fpr) - 1, max_length))
        for fpr in all_fpr
    ]
    interp_tpr = [
        interp1d(np.arange(len(tpr)), tpr)(np.linspace(0, len(tpr) - 1, max_length))
        for tpr in all_tpr
    ]

    mean_fpr = np.mean(interp_fpr, axis=0)
    mean_tpr = np.mean(interp_tpr, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)

    ax.plot(mean_fpr, mean_tpr, linestyle="--", label=f"Mean (AUC = {mean_auc:.2f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Cross-Validation ROC")
    ax.legend(loc="lower right")

    return fig, ax


def plot_cv_prauc(
    fold_y_trues: List[np.ndarray],
    fold_y_probs: List[np.ndarray],
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot cross-validation Precision-Recall curves with mean curve."""
    fig, ax = plt.subplots()
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "purple"])

    all_precision, all_recall = [], []

    for i, (y_true, y_prob) in enumerate(zip(fold_y_trues, fold_y_probs)):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        all_precision.append(precision)
        all_recall.append(recall)

        ax.plot(
            recall,
            precision,
            color=next(colors),
            label=f"Fold {i+1} (AUC = {pr_auc:.2f})",
        )

    max_length = max(len(arr) for arr in all_recall)
    interp_precision = [
        interp1d(np.arange(len(p)), p)(np.linspace(0, len(p) - 1, max_length))
        for p in all_precision
    ]
    interp_recall = [
        interp1d(np.arange(len(r)), r)(np.linspace(0, len(r) - 1, max_length))
        for r in all_recall
    ]

    mean_precision = np.mean(interp_precision, axis=0)
    mean_recall = np.mean(interp_recall, axis=0)
    mean_auc = auc(mean_recall, mean_precision)

    ax.plot(
        mean_recall,
        mean_precision,
        linestyle="--",
        label=f"Mean (AUC = {mean_auc:.2f})",
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Cross-Validation Precision-Recall")
    ax.legend(loc="lower left")

    return fig, ax
