from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Iterable

import h5py
import numpy as np
from numpy.typing import NDArray
import torch
from PIL import Image


@dataclass
class Coordinates:
    x_min: int = None
    y_min: int = None
    x_max: int = None
    y_max: int = None

    def __repr__(self) -> str:
        return (
            f"Coordinates(x_min={self.x_min}, y_min={self.y_min}, "
            f"x_max={self.x_max}, y_max={self.y_max})"
        )

    def to_string(self) -> str:
        return f"{self.x_min}_{self.y_min}_{self.x_max}_{self.y_max}"

    def to_list(self) -> List[int]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]


@dataclass
class Patch:
    """
    Data structure representing a single image patch.

    Attributes:
        image_array (np.ndarray): Image data.
        confidences (np.ndarray): Prediction confidences.
        coordinates (Coordinates): Bounding box in level 0 slide coordinates.
        feature (torch.Tensor): Feature embedding.
        address (Tuple[int, int]): (column, row) index.
        label (str): Patch label.
        slide_name (str): Slide identifier.
        path (str): File path to the patch image.
        level (int): Resolution level.
    """

    image_array: np.ndarray = None
    confidences: np.ndarray = None
    coordinates: Coordinates = None
    feature: torch.Tensor = None
    address: Tuple[int, int] = None
    label: str = None
    slide_name: str = None
    path: str = None
    level: int = None

    @classmethod
    def from_file(cls, path: str, **kwargs: dict) -> Patch:
        image_array = np.array(Image.open(path))
        return cls(image_array=image_array, path=path, **kwargs)

    def __repr__(self) -> str:
        image_shape = (
            "None" if self.image_array is None else str(self.image_array.shape)
        )
        return (
            "Patch("
            f"image_shape={image_shape}, path={self.path}, label={self.label}, "
            f"coordinates={self.coordinates}, address={self.address}, "
            f"confidences={self.confidences})"
        )

    def load(self) -> None:
        if not self.path:
            raise ValueError("Path is not specified.")
        self.image_array = np.array(Image.open(self.path))

    def close(self) -> None:
        del self.image_array


@dataclass
class Patches:
    """
    Collection of Patch objects.
    """

    data: List[Patch] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    dimension: tuple = None

    def __getitem__(self, i: Union[int, tuple, Iterable[int]]) -> Patch:
        if isinstance(i, int):
            return self.data[i]

        if isinstance(i, tuple) and all(isinstance(xy, int) for xy in i):
            if hasattr(self, "_address2idx"):
                if i in self._address2idx:
                    return self.data[self._address2idx[i]]
                raise KeyError(f"Address {i} not found.")
            else:
                self._address2idx = {
                    patch.address: idx for idx, patch in enumerate(self.data)
                }
                return self[i]

        if isinstance(i, Iterable):
            if all(isinstance(idx, int) for idx in i):
                return Patches([self.data[idx] for idx in i])
            return Patches([patch for patch, mask in zip(self.data, i) if mask])

        raise TypeError("Invalid index type.")

    def __contains__(self, address: tuple) -> bool:
        if not hasattr(self, "_address2idx"):
            self._address2idx = {
                patch.address: idx for idx, patch in enumerate(self.data)
            }
        return address in self._address2idx

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"Patches(length={len(self.data)})"

    @property
    def features(self) -> torch.Tensor:
        return torch.stack([patch.feature for patch in self.data], dim=0)

    @features.setter
    def features(self, features: torch.Tensor) -> None:
        if not isinstance(features, torch.Tensor):
            raise TypeError(
                f"features must be a torch.Tensor, but got {type(features)}"
            )
        if len(self.data) != len(features):
            raise ValueError(
                f"Feature length ({len(features)}) does not match "
                f"number of patches ({len(self.data)})."
            )

        for idx, patch in enumerate(self.data):
            patch.feature = features[idx]

    @property
    def addresses(self) -> List[Tuple[int, int]]:
        return [patch.address for patch in self.data]

    @classmethod
    def from_feature_h5_sydney(cls, path: str) -> Patches:
        """
        Loads patch features from an HDF5 file.

        Handles inconsistent internal structures by converting
        half-precision (float16) features to float32.
        """

        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")

        data: List[Patch] = []

        with h5py.File(path, "r") as fh:
            addresses_np = fh["addresses"][:]
            features_ds: NDArray = fh["features"][:]

            for i, addr in enumerate(addresses_np):
                col, row = int(addr[0]), int(addr[1])

                feature_tensor = (
                    torch.from_numpy(features_ds[i].astype("float32"))
                    if features_ds is not None
                    else None
                )

                patch = Patch(
                    image_array=None,
                    feature=feature_tensor,
                    address=(col, row),
                )
                data.append(patch)

        return cls(data=data)
