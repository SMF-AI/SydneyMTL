from __future__ import annotations

import os
from typing import (
    List,
    Tuple,
    Optional,
    Dict,
    Union,
    Iterable,
    Iterator,
    Generator,
)
from dataclasses import dataclass, field

import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold

from sydneymtl.data_models import Patches


@dataclass
class SydneyDataPoint:
    """
    Data structure representing a single slide-level bag.
    """

    bag_path: str
    bag_label: Dict[str, torch.Tensor]

    @property
    def label(self) -> Dict[str, torch.Tensor]:
        return self.bag_label

    @property
    def combined_label(
        self, tasknames: List[str] = ["hp", "neut", "mono", "atrophy", "im"]
    ) -> str:
        return "".join([str(self.bag_label[taskname].item()) for taskname in tasknames])

    @property
    def hp(self) -> torch.Tensor:
        return self.bag_label["hp"]

    @property
    def neut(self) -> torch.Tensor:
        return self.bag_label["neut"]

    @property
    def mono(self) -> torch.Tensor:
        return self.bag_label["mono"]

    @property
    def atrophy(self) -> torch.Tensor:
        return self.bag_label["atrophy"]

    @property
    def im(self) -> torch.Tensor:
        return self.bag_label["im"]


@dataclass
class SydneyBatch:
    """
    Collection of slide-level data points.
    """

    data: List[SydneyDataPoint]
    tasknames: List[str] = field(
        default_factory=lambda: ["hp", "neut", "mono", "atrophy", "im"]
    )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, idx: Union[int, Iterable]
    ) -> Union[SydneyDataPoint, SydneyBatch]:
        if isinstance(idx, int):
            return self.data[idx]
        elif isinstance(idx, Iterable):
            return SydneyBatch(data=[self.data[i] for i in idx])
        else:
            raise ValueError(f"Invalid index type: {type(idx)}")

    def __iter__(self) -> Iterator[SydneyDataPoint]:
        return iter(self.data)

    def __repr__(self) -> str:
        return f"SydneyBatch(length={len(self.data)})"

    @classmethod
    def from_csv(
        cls,
        feature_dir: str,
        csv_path: str,
        tasknames: List[str] = ["hp", "neut", "mono", "atrophy", "im"],
        mapping_label: Dict[int, int] = {9: 4},
        dry_run: bool = False,
    ) -> SydneyBatch:
        """
        Construct SydneyBatch from a CSV file.

        Args:
            feature_dir: Directory containing H5 feature files.
            csv_path: Path to label CSV file.
            tasknames: List of task names.
            mapping_label: Optional label remapping dictionary.
            dry_run: If True, limit the number of loaded samples.

        Returns:
            SydneyBatch instance.
        """
        df = pd.read_csv(csv_path, dtype={col: int for col in tasknames})
        df.set_index("slide_name", inplace=True)

        data = []

        for slide_name, row in tqdm(
            df.iterrows(), total=len(df), desc="Loading SydneyBatch"
        ):
            bag_path = os.path.join(feature_dir, slide_name + ".h5")
            if not os.path.exists(bag_path):
                continue

            bag_label = {}

            for taskname in tasknames:
                label = row[taskname]
                if label in mapping_label:
                    label = mapping_label[label]
                bag_label[taskname] = torch.tensor(label).long()

            data.append(SydneyDataPoint(bag_path, bag_label))

            if dry_run and len(data) > 200:
                break

        return cls(data=data, tasknames=tasknames)

    @property
    def labels(self) -> List[Dict[str, torch.Tensor]]:
        return [datapoint.bag_label for datapoint in self.data]

    @property
    def bag_paths(self) -> List[str]:
        return [datapoint.bag_path for datapoint in self.data]

    @property
    def combined_labels(self) -> List[str]:
        return [datapoint.combined_label for datapoint in self.data]

    @property
    def hp(self) -> torch.Tensor:
        return torch.stack([d.bag_label["hp"] for d in self.data])

    @property
    def neut(self) -> torch.Tensor:
        return torch.stack([d.bag_label["neut"] for d in self.data])

    @property
    def mono(self) -> torch.Tensor:
        return torch.stack([d.bag_label["mono"] for d in self.data])

    @property
    def atrophy(self) -> torch.Tensor:
        return torch.stack([d.bag_label["atrophy"] for d in self.data])

    @property
    def im(self) -> torch.Tensor:
        return torch.stack([d.bag_label["im"] for d in self.data])

    def train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> Tuple[SydneyBatch, SydneyBatch]:
        stratify_labels = self.combined_labels if stratify else None

        try:
            train_data, val_data = train_test_split(
                self.data,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_labels,
            )
        except ValueError:
            train_data, val_data = train_test_split(
                self.data,
                test_size=test_size,
                random_state=random_state,
                stratify=None,
            )

        return SydneyBatch(train_data), SydneyBatch(val_data)

    def kfold_generator(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        stratify: bool = True,
    ) -> Generator[Tuple[SydneyBatch, SydneyBatch, SydneyBatch], None, None]:
        """
        Generate train/validation/test splits using stratified K-fold.
        """
        kfold = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )

        for train_val_idx, test_idx in kfold.split(self.data, self.combined_labels):
            train_val_batch = self[train_val_idx]
            test_batch = self[test_idx]

            train_batch, val_batch = train_val_batch.train_test_split(
                test_size=0.125,
                random_state=random_state,
                stratify=stratify,
            )

            yield train_batch, val_batch, test_batch

    def compute_class_weight(self, task: str) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for CrossEntropyLoss.
        """
        labels = getattr(self, task)
        num_classes = 5 if task == "atrophy" else 4

        counts = torch.bincount(labels, minlength=num_classes).float()
        eps = 1e-6
        weights = 1.0 / (counts + eps)

        return weights / weights.mean()


class SydneyDataset(Dataset):
    """
    PyTorch Dataset wrapper for SydneyBatch.

    Returns:
        features: Tensor of shape (N, D)
        labels: Dict[str, torch.Tensor]
    """

    def __init__(self, batch: SydneyBatch):
        self.batch = batch
        self.tasknames = batch.tasknames

    def __len__(self) -> int:
        return len(self.batch)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h5_path = self.batch.bag_paths[idx]
        patches = Patches.from_feature_h5_sydney(h5_path)
        label = self.batch.labels[idx]

        return patches.features, label
