#!/usr/bin/env python3
"""Embed gallery images with a given model and save results to disk.

Usage:
    python scripts/embed_gallery.py --config configs/dataset.yaml --model A1-dinov2-cls
    python scripts/embed_gallery.py --config configs/dataset.yaml --model A2-openclip --batch-size 32
    python scripts/embed_gallery.py --config configs/dataset.yaml --model A4-sscd --split test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import structlog
import yaml

from vinylid_ml.dataset import load_manifest, load_splits
from vinylid_ml.gallery import GalleryImageDataset, embed_dataset, save_embeddings
from vinylid_ml.models import (
    DINOv2Embedder,
    EmbeddingModel,
    OpenCLIPEmbedder,
    SSCDEmbedder,
)

logger = structlog.get_logger()


def _create_model(model_id: str) -> EmbeddingModel:
    """Instantiate an EmbeddingModel from a model ID string.

    Args:
        model_id: Model identifier (e.g., "A1-dinov2-cls", "A2-openclip").

    Returns:
        Configured EmbeddingModel instance.

    Raises:
        ValueError: If model_id is not recognized.
    """
    match model_id:
        case "A1-dinov2-cls":
            return DINOv2Embedder(pooling="cls")
        case "A1-dinov2-gem":
            return DINOv2Embedder(pooling="gem")
        case "A1-dinov2-cls-518":
            return DINOv2Embedder(pooling="cls", input_size=518)
        case "A1-dinov2-gem-518":
            return DINOv2Embedder(pooling="gem", input_size=518)
        case "A2-openclip":
            return OpenCLIPEmbedder()
        case "A4-sscd":
            return SSCDEmbedder()
        case _:
            msg = (
                f"Unknown model_id: '{model_id}'. Available: "
                "A1-dinov2-cls, A1-dinov2-gem, A1-dinov2-cls-518, A1-dinov2-gem-518, "
                "A2-openclip, A4-sscd"
            )
            raise ValueError(msg)


def _filter_manifest_by_split(
    manifest: pd.DataFrame,
    splits: dict[str, str],
    split_name: str,
) -> pd.DataFrame:
    """Filter manifest to images in the specified split(s).

    Args:
        manifest: Full manifest DataFrame.
        splits: Album ID -> split name mapping.
        split_name: "train", "val", "test", or "all".

    Returns:
        Filtered DataFrame.
    """
    if split_name == "all":
        return manifest

    album_ids_in_split = {album_id for album_id, s in splits.items() if s == split_name}
    mask = manifest["album_id"].astype(str).isin(album_ids_in_split)
    return manifest[mask].reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    """Entry point for gallery embedding script."""
    parser = argparse.ArgumentParser(
        description="Embed gallery images with a given model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to dataset.yaml config file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model ID (e.g., A1-dinov2-cls, A2-openclip, A4-sscd).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="Which split to embed (default: all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding (default: 64).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: from config).",
    )

    args = parser.parse_args(argv)

    # Load config
    config_path: Path = args.config.resolve()
    if not config_path.exists():
        logger.error("config_not_found", path=str(config_path))
        sys.exit(1)

    with config_path.open() as f:
        config = yaml.safe_load(f)

    config_dir = config_path.parent
    gallery_root = Path(config["paths"]["gallery_root"])
    if not gallery_root.is_absolute():
        gallery_root = (config_dir / gallery_root).resolve()

    # Resolve output dir
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        output_dir = (config_dir / config["paths"]["output_dir"]).resolve()

    # Resolve manifest and splits paths
    manifest_path = output_dir / "manifest.parquet"
    splits_path = output_dir / "splits.json"

    logger.info(
        "config_loaded",
        gallery_root=str(gallery_root),
        output_dir=str(output_dir),
        model=args.model,
        split=args.split,
        batch_size=args.batch_size,
    )

    # Load manifest and splits
    manifest = load_manifest(manifest_path)
    splits = load_splits(splits_path)

    # Filter by split
    filtered = _filter_manifest_by_split(manifest, splits, args.split)
    logger.info(
        "manifest_filtered",
        split=args.split,
        num_images=len(filtered),
    )

    if len(filtered) == 0:
        logger.warning("no_images_in_split", split=args.split)
        sys.exit(0)

    # Create model
    model = _create_model(args.model)

    # Create dataset with model-specific transforms
    dataset = GalleryImageDataset(filtered, gallery_root, model.get_transforms())

    # Embed
    start = time.monotonic()
    result = embed_dataset(model, dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    elapsed = time.monotonic() - start

    # Save
    save_embeddings(result, output_dir)

    logger.info(
        "pipeline_complete",
        model_id=args.model,
        split=args.split,
        num_images=len(result.image_paths),
        embedding_shape=list(result.embeddings.shape),
        total_time_s=round(elapsed, 2),
        output_dir=str(output_dir / args.model),
    )


if __name__ == "__main__":
    main()
