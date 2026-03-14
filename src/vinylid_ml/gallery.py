"""Gallery embedding pipeline — batch-embed images and persist results.

Provides reusable functions for embedding a gallery of images with any
EmbeddingModel wrapper and saving/loading the results as .npy + metadata JSON.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import structlog
import torch
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

if TYPE_CHECKING:
    from vinylid_ml.models import EmbeddingModel

__all__ = [
    "EmbeddingResult",
    "GalleryImageDataset",
    "embed_dataset",
    "load_embeddings",
    "save_embeddings",
]

logger = structlog.get_logger()


@dataclass
class EmbeddingResult:
    """Container for gallery embedding output.

    Attributes:
        embeddings: L2-normalized embedding matrix of shape (N, D), float16.
        image_paths: Relative image paths aligned with embedding rows.
        album_ids: Album ID strings aligned with embedding rows.
        model_id: ID of the model used to produce embeddings.
        embedding_dim: Dimensionality of each embedding vector.
    """

    embeddings: NDArray[np.floating]
    image_paths: list[str]
    album_ids: list[str]
    model_id: str
    embedding_dim: int


class GalleryImageDataset(Dataset[tuple[torch.Tensor, str, str]]):
    """Lightweight dataset for gallery embedding extraction.

    Returns (tensor, image_path, album_id) per item. Accepts a pre-filtered
    manifest DataFrame — no split filtering is done internally.

    Args:
        manifest: DataFrame with at least 'image_path' and 'album_id' columns.
        gallery_root: Root directory to resolve relative image paths.
        transform: Image preprocessing transform (from model.get_transforms()).
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        gallery_root: Path,
        transform: transforms.Compose,
    ) -> None:
        self._df = manifest.reset_index(drop=True)
        self._gallery_root = gallery_root
        self._transform = transform

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, str]:
        row = self._df.iloc[idx]
        image_path = Path(row["image_path"])

        if not image_path.is_absolute():
            image_path = self._gallery_root / image_path

        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
            tensor: torch.Tensor = self._transform(img_rgb)  # type: ignore[assignment]

        return tensor, str(row["image_path"]), str(row["album_id"])


def _collate_gallery(
    batch: list[tuple[torch.Tensor, str, str]],
) -> tuple[torch.Tensor, list[str], list[str]]:
    """Custom collate function that separates tensors from string metadata."""
    tensors, paths, album_ids = zip(*batch, strict=True)
    return torch.stack(tensors), list(paths), list(album_ids)


def embed_dataset(
    model: EmbeddingModel,
    dataset: GalleryImageDataset,
    *,
    batch_size: int = 64,
    num_workers: int = 4,
) -> EmbeddingResult:
    """Batch-embed all images in a gallery dataset.

    Args:
        model: An EmbeddingModel instance (e.g., DINOv2Embedder).
        dataset: GalleryImageDataset to embed.
        batch_size: Images per batch. Default 64.
        num_workers: DataLoader worker processes. Default 4.

    Returns:
        EmbeddingResult with float16 embeddings and aligned metadata.
    """
    # pin_memory only benefits CUDA; not supported on MPS
    use_pin_memory = torch.cuda.is_available()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_gallery,
        pin_memory=use_pin_memory,
    )

    total_images = len(dataset)

    if total_images == 0:
        logger.warning("embed_dataset_empty", model_id=model.model_id)
        return EmbeddingResult(
            embeddings=np.empty((0, model.embedding_dim), dtype=np.float16),
            image_paths=[],
            album_ids=[],
            model_id=model.model_id,
            embedding_dim=model.embedding_dim,
        )

    all_embeddings: list[np.ndarray] = []
    all_paths: list[str] = []
    all_album_ids: list[str] = []

    num_batches = (total_images + batch_size - 1) // batch_size

    logger.info(
        "embedding_start",
        model_id=model.model_id,
        total_images=total_images,
        batch_size=batch_size,
        num_batches=num_batches,
    )

    start_time = time.monotonic()

    for batch_idx, (images, paths, album_ids) in enumerate(loader):
        batch_start = time.monotonic()
        embeddings = model.embed(images)
        batch_elapsed = time.monotonic() - batch_start

        all_embeddings.append(embeddings.numpy())
        all_paths.extend(paths)
        all_album_ids.extend(album_ids)

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            elapsed = time.monotonic() - start_time
            images_done = min((batch_idx + 1) * batch_size, total_images)
            throughput = images_done / elapsed if elapsed > 0 else 0.0
            logger.info(
                "embedding_progress",
                batch=f"{batch_idx + 1}/{num_batches}",
                images=f"{images_done}/{total_images}",
                batch_time_ms=round(batch_elapsed * 1000, 1),
                throughput_ips=round(throughput, 1),
            )

    total_elapsed = time.monotonic() - start_time
    final_throughput = total_images / total_elapsed if total_elapsed > 0 else 0.0

    embeddings_matrix = np.concatenate(all_embeddings, axis=0).astype(np.float16)

    logger.info(
        "embedding_complete",
        model_id=model.model_id,
        total_images=total_images,
        embedding_shape=list(embeddings_matrix.shape),
        total_time_s=round(total_elapsed, 2),
        throughput_ips=round(final_throughput, 1),
    )

    return EmbeddingResult(
        embeddings=embeddings_matrix,
        image_paths=all_paths,
        album_ids=all_album_ids,
        model_id=model.model_id,
        embedding_dim=model.embedding_dim,
    )


def save_embeddings(result: EmbeddingResult, output_dir: Path) -> Path:
    """Save embedding result to disk.

    Creates ``{output_dir}/{model_id}/embeddings.npy`` and
    ``{output_dir}/{model_id}/metadata.json``.

    Args:
        result: EmbeddingResult to save.
        output_dir: Parent directory for output (e.g., ``data/``).

    Returns:
        Path to the model-specific output directory.
    """
    model_dir = output_dir / result.model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings as float16 numpy array
    npy_path = model_dir / "embeddings.npy"
    np.save(npy_path, result.embeddings.astype(np.float16))

    # Save metadata
    metadata = {
        "model_id": result.model_id,
        "embedding_dim": result.embedding_dim,
        "num_images": len(result.image_paths),
        "image_paths": result.image_paths,
        "album_ids": result.album_ids,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = model_dir / "metadata.json"
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "embeddings_saved",
        model_id=result.model_id,
        npy_path=str(npy_path),
        num_images=len(result.image_paths),
    )
    return model_dir


def load_embeddings(output_dir: Path, model_id: str) -> EmbeddingResult:
    """Load previously saved embeddings from disk.

    Args:
        output_dir: Parent directory containing model subdirectories.
        model_id: Model ID to load (e.g., "A1-dinov2-cls").

    Returns:
        EmbeddingResult with the saved embeddings and metadata.

    Raises:
        FileNotFoundError: If embeddings or metadata files are missing.
    """
    model_dir = output_dir / model_id

    npy_path = model_dir / "embeddings.npy"
    if not npy_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {npy_path}")

    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    embeddings: NDArray[np.floating] = np.load(npy_path)
    with meta_path.open() as f:
        metadata = json.load(f)  # Returns Any — validated below

    # metadata is Any from json.load — iterate directly to produce list[str]
    # (isinstance narrows to list[Unknown] which causes strict-mode issues)
    try:
        image_paths: list[str] = [str(p) for p in metadata["image_paths"]]
        album_ids: list[str] = [str(a) for a in metadata["album_ids"]]
    except (TypeError, KeyError) as e:
        msg = f"Invalid image_paths or album_ids in {meta_path}: {e}"
        raise TypeError(msg) from e

    if len(image_paths) != embeddings.shape[0]:
        msg = (
            f"Metadata/embedding mismatch: {len(image_paths)} paths "
            f"vs {embeddings.shape[0]} embeddings in {model_dir}"
        )
        raise ValueError(msg)

    result = EmbeddingResult(
        embeddings=embeddings,
        image_paths=image_paths,
        album_ids=album_ids,
        model_id=str(metadata["model_id"]),
        embedding_dim=int(metadata["embedding_dim"]),
    )

    logger.info(
        "embeddings_loaded",
        model_id=model_id,
        shape=list(embeddings.shape),
    )
    return result
