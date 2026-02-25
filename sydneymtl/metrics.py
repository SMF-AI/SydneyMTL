from __future__ import annotations

from typing import List, Dict, Set, Optional, Literal, Tuple, Sequence

import numpy as np
import seaborn as sns
from numpy.typing import NDArray
from matplotlib import pyplot as plt


class AverageMeter:
    """Track and compute running average of a scalar metric."""

    def __init__(self, name: str):
        self.name = name
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: float = 0.0

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def __repr__(self) -> str:
        return f"AverageMeter(name={self.name}, avg={self.avg:.4f}, count={self.count})"


class MultiTaskAverageMeter:
    """Collection of AverageMeter instances for multi-task training."""

    def __init__(self, phase: str, task: List[str]):
        self.phase = phase
        self.meters = {key: AverageMeter(f"{phase}_{key}") for key in task}
        self.meters["total_loss"] = AverageMeter(f"{phase}_total_loss")

    def update(self, updates: Dict[str, float], n: int = 1) -> None:
        total = 0.0
        for key, val in updates.items():
            if key in self.meters:
                self.meters[key].update(val, n)
                total += val
        self.meters["total_loss"].update(total, n)

    def reset(self) -> None:
        for meter in self.meters.values():
            meter.reset()

    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        return {
            f"{prefix}{self.phase}_{key}": round(meter.avg, 5)
            for key, meter in self.meters.items()
        }

    def __getitem__(self, key: str) -> AverageMeter:
        return self.meters[key]

    def __repr__(self) -> str:
        return " | ".join(str(meter) for meter in self.meters.values())


class MulticlassMetricsMeter:
    """
    Streaming confusion-matrix-based metric meter for multi-class classification.
    """

    def __init__(self, name: str, n_classes: int = 4) -> None:
        self.name = name
        self.total = 0
        self.correct = 0
        self.n_classes = n_classes
        self.conf_mat = np.zeros((n_classes, n_classes), dtype=int)

    def reset(self) -> None:
        self.total = 0
        self.correct = 0
        self.conf_mat.fill(0)

    def update(
        self,
        probs: Sequence[Sequence[float]],
        labels: Sequence[int] | NDArray[np.integer],
    ) -> None:
        preds = np.argmax(np.array(probs), axis=1)
        labels_np = np.asarray(labels, dtype=np.int64)

        self.total += labels_np.size
        self.correct += int((preds == labels_np).sum())

        for p, t in zip(preds, labels_np):
            self.conf_mat[t, p] += 1

    @property
    def accuracy(self) -> float:
        return 0.0 if self.total == 0 else self.correct / self.total

    @property
    def confusion_matrix(self) -> np.ndarray:
        return self.conf_mat.copy()

    def _weights_matrix(self, weights: Optional[str] = "quadratic") -> np.ndarray:
        K = self.n_classes
        I, J = np.ogrid[:K, :K]
        D = np.abs(I - J).astype(np.float64)

        if weights is None or weights == "none":
            return np.zeros((K, K), dtype=np.float64)
        if weights == "linear":
            return D / (K - 1)
        if weights == "quadratic":
            return (D**2) / ((K - 1) ** 2)

        raise ValueError(f"Unknown weight scheme: {weights}")

    def kappa(
        self, weights: Optional[Literal["linear", "quadratic"]] = "quadratic"
    ) -> float:
        N = self.conf_mat.sum()
        if N == 0:
            return 0.0

        O = self.conf_mat.astype(np.float64) / N
        r = O.sum(axis=1, keepdims=True)
        c = O.sum(axis=0, keepdims=True)
        E = r @ c

        W = self._weights_matrix(weights)
        weighted_observed = (W * O).sum()
        weighted_expected = (W * E).sum()

        if weighted_expected == 0:
            return 0.0

        return 1.0 - (weighted_observed / weighted_expected)

    @property
    def kappa_linear(self) -> float:
        return self.kappa(weights="linear")

    @property
    def kappa_quadratic(self) -> float:
        return self.kappa(weights="quadratic")

    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        return {f"{prefix}{self.name}_acc": round(self.accuracy, 4)}

    def plot_confusion_matrix(
        self, normalize: bool = False, **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        if self.total == 0:
            return plt.subplots()

        cm = self.confusion_matrix
        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

        labels = kwargs.pop("labels", None)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            ax=ax,
            xticklabels=labels if labels is not None else "auto",
            yticklabels=labels if labels is not None else "auto",
            **kwargs,
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"Confusion Matrix ({self.name})")

        return fig, ax


class LenientMetricsMeter(MulticlassMetricsMeter):
    """Extension supporting lenient accuracy computation."""

    def __init__(
        self,
        name: str,
        n_classes: int,
        lenient_map: Optional[Dict[int, Set[int]]] = None,
    ):
        super().__init__(name=name, n_classes=n_classes)
        self.lenient_map = lenient_map
        self.lenient_correct = 0

    def update(
        self,
        probs: Sequence[Sequence[float]],
        labels: Sequence[int] | NDArray[np.integer],
    ) -> None:
        preds = np.argmax(np.array(probs), axis=1)
        labels_np = np.asarray(labels, dtype=np.int64)

        self.total += labels_np.size
        self.correct += int((preds == labels_np).sum())

        if self.lenient_map is not None:
            self.lenient_correct += int(
                sum(p in self.lenient_map[t] for p, t in zip(preds, labels_np))
            )

        for p, t in zip(preds, labels_np):
            self.conf_mat[t, p] += 1

    @property
    def lenient_accuracy(self) -> float:
        return 0.0 if self.total == 0 else self.lenient_correct / self.total

    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        return {
            f"{prefix}{self.name}_acc": round(self.accuracy, 4),
            f"{prefix}{self.name}_lacc": round(self.lenient_accuracy, 4),
            f"{prefix}{self.name}_kappa_quadratic": round(self.kappa_quadratic, 4),
        }


class MultiTaskMulticlassMetricMeters:
    """Multi-task wrapper for MulticlassMetricsMeter."""

    def __init__(self, phase: str, n_classes: int, task_names: List[str]):
        self.phase = phase
        self.task_names = task_names
        self.meters = {
            task: MulticlassMetricsMeter(task, n_classes=n_classes)
            for task in task_names
        }

    def update(
        self, probs: Dict[str, List[List[float]]], labels: Dict[str, List[int]]
    ) -> None:
        for task in self.task_names:
            self.meters[task].update(probs[task], labels[task])

    def reset(self) -> None:
        for meter in self.meters.values():
            meter.reset()

    def __getitem__(self, task_name: str) -> MulticlassMetricsMeter:
        return self.meters[task_name]

    def __repr__(self) -> str:
        return " | ".join(str(meter) for meter in self.meters.values())

    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        results = {}
        for task, metrics in self.meters.items():
            for metric_name, value in metrics.to_dict(prefix=prefix).items():
                results[f"{prefix}{self.phase}_{metric_name}"] = value
        return results


class LenientMultiTaskMetricsMeter(MultiTaskMulticlassMetricMeters):
    """Multi-task wrapper supporting lenient metrics."""

    def __init__(
        self,
        phase: str,
        n_classes: int,
        task_names: List[str],
        lenient_map: Optional[Dict[int, Set[int]]] = None,
    ):
        super().__init__(phase=phase, n_classes=n_classes, task_names=task_names)
        self.lenient_map = lenient_map
        self.meters = {
            task: LenientMetricsMeter(
                task, n_classes=n_classes, lenient_map=lenient_map
            )
            for task in task_names
        }
