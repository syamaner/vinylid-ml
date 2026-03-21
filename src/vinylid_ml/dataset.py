"""Dataset loading, manifest handling, and PyTorch Dataset for album cover images."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

import numpy as np
import pandas as pd
import structlog
import torch
from PIL import Image
from PIL import PngImagePlugin as _PngPlugin
from torch.utils.data import Dataset
from torchvision import transforms

#: PIL decompression bomb pixel limit for large gallery scans (~17K px).
#: Applied on first image load in __getitem__, not at import time.
_MAX_IMAGE_PIXELS = 300_000_000
#: PNG text chunk limit for images with large embedded XML metadata.
_MAX_TEXT_CHUNK = 10 * 1024 * 1024  # 10 MB


def _ensure_pil_limits() -> None:
    """Ensure PIL safety limits are at least the desired thresholds.

    Monotonic/idempotent: checks current globals and raises them if lower
    than desired.  Avoids issues where another module (e.g. ``exif.py``)
    might set a lower value after this module's first call.
    """
    current_pixels = getattr(Image, "MAX_IMAGE_PIXELS", None)
    if current_pixels is None or current_pixels < _MAX_IMAGE_PIXELS:
        Image.MAX_IMAGE_PIXELS = _MAX_IMAGE_PIXELS

    current_text = getattr(_PngPlugin, "MAX_TEXT_CHUNK", None)
    if current_text is None or current_text < _MAX_TEXT_CHUNK:
        _PngPlugin.MAX_TEXT_CHUNK = _MAX_TEXT_CHUNK


__all__ = [
    "AlbumCoverDataset",
    "DatasetConfig",
    "get_eval_transforms",
    "get_train_transforms",
    "load_manifest",
    "load_splits",
]

logger = structlog.get_logger()

# ImageNet normalization (used by DINOv2, MobileNetV3, SSCD)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for dataset loading.

    Attributes:
        manifest_path: Path to manifest.parquet.
        splits_path: Path to splits.json.
        gallery_root: Root directory of album-export images.
        input_size: Model input resolution (e.g., 224 or 518).
    """

    manifest_path: Path
    splits_path: Path
    gallery_root: Path
    input_size: int = 224


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load the dataset manifest from a Parquet file.

    Args:
        manifest_path: Path to manifest.parquet.

    Returns:
        DataFrame with columns: image_path, release_id, artist, album,
        artist_dir, album_dir, album_id, width, height, format, cover_type.

    Raises:
        FileNotFoundError: If manifest file does not exist.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_parquet(manifest_path)
    logger.info(
        "manifest_loaded",
        num_images=len(df),
        num_albums=df["album_id"].nunique(),
        num_artists=df["artist"].nunique(),
    )
    return df


def load_splits(splits_path: Path) -> dict[str, str]:
    """Load album-level train/val/test split assignments.

    Args:
        splits_path: Path to splits.json.

    Returns:
        Mapping of album_id (as string) -> split name ("train", "val", "test").

    Raises:
        FileNotFoundError: If splits file does not exist.
    """
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_path}")

    with splits_path.open() as f:
        splits = json.load(f)

    counts: dict[str, int] = {}
    for split_name in splits.values():
        counts[split_name] = counts.get(split_name, 0) + 1

    logger.info("splits_loaded", **counts)
    return splits


def get_eval_transforms(input_size: int = 224) -> transforms.Compose:
    """Get standard evaluation transforms (deterministic, no augmentation).

    Args:
        input_size: Target image size. Must be divisible by 14 for DINOv2.

    Returns:
        Composed transform: Resize → CenterCrop → ToTensor → Normalize.
    """
    return transforms.Compose(
        [
            transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_train_transforms(input_size: int = 224) -> transforms.Compose:
    """Get training transforms with augmentation for fine-tuning.

    Args:
        input_size: Target image size.

    Returns:
        Composed transform with random augmentations suitable for album covers.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            # No horizontal flip — album covers should not be flipped
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class AlbumCoverDataset(Dataset[tuple[torch.Tensor | Image.Image, int]]):
    """PyTorch Dataset for album cover images.

    Loads images from the manifest, filters by split, and applies transforms.

    Args:
        manifest: DataFrame from load_manifest().
        splits: Split mapping from load_splits().
        split_name: Which split to use ("train", "val", or "test").
        transform: Image transform to apply, or ``None`` to return raw
            PIL Images (useful for ``MultiViewTransform`` wrappers).
        gallery_root: Root directory to resolve relative image paths.
    """

    @overload
    def __init__(
        self,
        manifest: pd.DataFrame,
        splits: dict[str, str],
        split_name: Literal["train", "val", "test"],
        transform: transforms.Compose,
        gallery_root: Path,
    ) -> None: ...

    @overload
    def __init__(
        self,
        manifest: pd.DataFrame,
        splits: dict[str, str],
        split_name: Literal["train", "val", "test"],
        transform: None,
        gallery_root: Path,
    ) -> None: ...

    def __init__(
        self,
        manifest: pd.DataFrame,
        splits: dict[str, str],
        split_name: Literal["train", "val", "test"],
        transform: transforms.Compose | None,
        gallery_root: Path,
    ) -> None:
        # Filter manifest to images belonging to albums in the requested split
        album_ids_in_split = {album_id for album_id, s in splits.items() if s == split_name}
        mask = manifest["album_id"].astype(str).isin(album_ids_in_split)
        self._df = manifest[mask].reset_index(drop=True)
        self._transform = transform
        self._gallery_root = gallery_root

        # Build album_id -> integer label mapping (deterministic)
        unique_albums = sorted(self._df["album_id"].unique())
        self._album_to_label: dict[str, int] = {str(aid): i for i, aid in enumerate(unique_albums)}

        logger.info(
            "dataset_created",
            split=split_name,
            num_images=len(self._df),
            num_albums=len(unique_albums),
        )

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | Image.Image, int]:
        row = self._df.iloc[idx]
        image_path = Path(row["image_path"])

        # Resolve relative paths against gallery root
        if not image_path.is_absolute():
            image_path = self._gallery_root / image_path

        _ensure_pil_limits()
        with Image.open(image_path) as pil_img:
            img = pil_img.convert("RGB")

        label = self._album_to_label[str(row["album_id"])]

        if self._transform is None:
            return img, label

        tensor: torch.Tensor = self._transform(img)  # type: ignore[assignment]
        return tensor, label

    @property
    def num_classes(self) -> int:
        """Number of unique album classes in this split."""
        return len(self._album_to_label)

    @property
    def album_to_label(self) -> dict[str, int]:
        """Mapping from album_id string to integer label."""
        return dict(self._album_to_label)

    def get_album_image_counts(self) -> dict[int, int]:
        """Get the number of images per album label in this split.

        Returns:
            Mapping of integer label -> number of images.
        """
        counts: dict[int, int] = {}
        for album_id_str, label in self._album_to_label.items():
            n = int((self._df["album_id"].astype(str) == album_id_str).sum())
            counts[label] = n
        return counts

    def get_resolutions(self) -> np.ndarray:
        """Get the max dimension (width or height) for each image.

        Returns:
            Array of shape (num_images,) with max(width, height) per image.
        """
        widths: np.ndarray = self._df["width"].to_numpy()
        heights: np.ndarray = self._df["height"].to_numpy()
        return np.maximum(widths, heights)
