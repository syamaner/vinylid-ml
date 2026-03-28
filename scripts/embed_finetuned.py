#!/usr/bin/env python3
"""Embed gallery images using a fine-tuned FineTuneModel checkpoint.

Loads a checkpoint from ``train.py``, embeds images from the requested split,
and saves results in the standard ``EmbeddingResult`` format (npy + metadata.json)
so that ``evaluate.py`` can run on them directly.

Usage:
    python scripts/embed_finetuned.py --config configs/dataset.yaml \
        --checkpoint results/B1-mobilenet_v3_small-supcon/2026-03-21T22-03-32/best_checkpoint.pt \
        --model-id B1-mobilenet_v3_small-supcon --split test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from vinylid_ml.dataset import get_eval_transforms, load_manifest, load_splits
from vinylid_ml.gallery import EmbeddingResult, save_embeddings
from vinylid_ml.training import FineTuneModel, TrainingConfig

logger = structlog.get_logger()


class _SplitImageDataset(Dataset[tuple[torch.Tensor, str, str]]):
    """Dataset that loads images from a manifest filtered by split.

    Returns ``(tensor, image_path, album_id)`` per item — same interface as
    ``GalleryImageDataset`` so that downstream collation is compatible.

    Args:
        manifest: Full manifest DataFrame.
        splits: Album ID → split name mapping.
        split_name: Split to filter to.
        transform: Eval transforms to apply.
        gallery_root: Root directory of album images.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        splits: dict[str, str],
        split_name: str,
        transform: torch.nn.Module,
        gallery_root: Path,
    ) -> None:
        album_ids_in_split = {aid for aid, s in splits.items() if s == split_name}
        self._df = manifest[manifest["album_id"].astype(str).isin(album_ids_in_split)].reset_index(
            drop=True
        )
        self._transform = transform
        self._gallery_root = gallery_root

        logger.info(
            "split_dataset_created",
            split=split_name,
            num_images=len(self._df),
            num_albums=len(album_ids_in_split),
        )

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


def _collate(
    batch: list[tuple[torch.Tensor, str, str]],
) -> tuple[torch.Tensor, list[str], list[str]]:
    """Custom collate separating tensors from string metadata."""
    tensors, paths, album_ids = zip(*batch, strict=True)
    return torch.stack(tensors), list(paths), list(album_ids)


def _get_device() -> torch.device:
    """Get best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def embed_with_finetuned(
    model: FineTuneModel,
    dataset: _SplitImageDataset,
    *,
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Embed all images in the dataset using the fine-tuned model.

    Args:
        model: FineTuneModel in eval mode.
        dataset: Dataset returning (tensor, path, album_id).
        batch_size: Batch size for embedding.
        num_workers: DataLoader workers.

    Returns:
        Tuple of (embeddings_np, image_paths, album_ids).
    """
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
    )

    all_embeddings: list[np.ndarray] = []
    all_paths: list[str] = []
    all_album_ids: list[str] = []

    total = len(dataset)
    num_batches = (total + batch_size - 1) // batch_size
    start = time.monotonic()

    for batch_idx, (images, paths, album_ids) in enumerate(loader):
        embeddings = model(images)
        all_embeddings.append(embeddings.cpu().numpy())
        all_paths.extend(paths)
        all_album_ids.extend(album_ids)

        if (batch_idx + 1) % 20 == 0 or batch_idx == num_batches - 1:
            elapsed = time.monotonic() - start
            done = min((batch_idx + 1) * batch_size, total)
            throughput = done / elapsed if elapsed > 0 else 0.0
            logger.info(
                "embed_progress",
                batch=f"{batch_idx + 1}/{num_batches}",
                images=f"{done}/{total}",
                throughput_ips=round(throughput, 1),
            )

    embeddings_np = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    total_time = time.monotonic() - start

    logger.info(
        "embedding_complete",
        total_images=total,
        shape=list(embeddings_np.shape),
        total_time_s=round(total_time, 2),
    )

    return embeddings_np, all_paths, all_album_ids


def main(argv: list[str] | None = None) -> None:
    """Entry point for fine-tuned embedding script."""
    parser = argparse.ArgumentParser(
        description="Embed gallery images with a fine-tuned checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to dataset.yaml.")
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to best_checkpoint.pt."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model ID for output directory naming (e.g., B1-mobilenet_v3_small-supcon).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Which split to embed (default: test).",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for embeddings (default: from config).",
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

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        output_dir = (config_dir / config["paths"]["output_dir"]).resolve()

    data_dir = (config_dir / config["paths"]["output_dir"]).resolve()
    manifest_path = data_dir / "manifest.parquet"
    splits_path = data_dir / "splits.json"

    # Load training config from checkpoint directory
    checkpoint_path: Path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        logger.error("checkpoint_not_found", path=str(checkpoint_path))
        sys.exit(1)

    run_dir = checkpoint_path.parent
    train_config_path = run_dir / "config.json"
    if train_config_path.exists():
        train_config = TrainingConfig.load(train_config_path)
        backbone_name = train_config.backbone
        projection_dim = train_config.projection_dim
    else:
        logger.error("train_config_not_found", path=str(train_config_path))
        sys.exit(1)

    device = _get_device()

    logger.info(
        "embed_finetuned_start",
        model_id=args.model_id,
        checkpoint=str(checkpoint_path),
        backbone=backbone_name,
        projection_dim=projection_dim,
        split=args.split,
        device=str(device),
    )

    # Load model and checkpoint
    model = FineTuneModel(
        backbone_name=backbone_name,  # type: ignore[arg-type]
        projection_dim=projection_dim,
        device=device,
        freeze_backbone=False,
    )

    # weights_only=False is required: checkpoints include optimizer_state_dict and
    # loss_state_dict, which contain arbitrary Python objects that weights_only=True
    # cannot deserialise. These checkpoints are always produced by our own train.py.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(
        "checkpoint_loaded",
        epoch=checkpoint.get("epoch"),
        val_r1=checkpoint.get("val_recall_at_1"),
    )

    # Load data
    manifest = load_manifest(manifest_path)
    splits = load_splits(splits_path)
    transform = get_eval_transforms(input_size=train_config.input_size)

    split_names = ["train", "val", "test"] if args.split == "all" else [args.split]

    dataset = _SplitImageDataset(
        manifest=manifest,
        splits=(
            splits
            if args.split != "all"
            else {aid: "combined" for aid in splits if splits[aid] in split_names}
        ),
        split_name=args.split if args.split != "all" else "combined",
        transform=transform,
        gallery_root=gallery_root,
    )

    if len(dataset) == 0:
        logger.warning("no_images", split=args.split)
        sys.exit(0)

    # Embed
    embeddings_np, image_paths, album_ids = embed_with_finetuned(
        model, dataset, batch_size=args.batch_size
    )

    # Save in standard format
    result = EmbeddingResult(
        embeddings=embeddings_np.astype(np.float16),
        image_paths=image_paths,
        album_ids=album_ids,
        model_id=args.model_id,
        embedding_dim=projection_dim,
    )

    save_path = save_embeddings(result, output_dir)

    logger.info(
        "embed_finetuned_complete",
        model_id=args.model_id,
        split=args.split,
        num_images=len(image_paths),
        embedding_shape=list(embeddings_np.shape),
        output=str(save_path),
    )

    print(f"\nEmbeddings saved: {save_path}")
    print(f"  {len(image_paths)} images, {projection_dim}-dim")
    eval_cmd = f"python scripts/evaluate.py --config {args.config} --model {args.model_id}"
    if args.output_dir is not None:
        eval_cmd += f" --embeddings-dir {output_dir}"
    print(f"  Run: {eval_cmd}")


if __name__ == "__main__":
    main()
