#!/usr/bin/env python3
"""Fine-tune an embedding model with metric learning losses.

Supports three loss strategies for the training strategy evaluation (#16):
- ArcFace (classification-based angular margin)
- Proxy-Anchor (proxy-based metric learning)
- SupCon (supervised contrastive with multi-view augmentation)

Usage:
    # Quick strategy eval (2000 albums, 10 epochs, DINOv2)
    python scripts/train.py --config configs/dataset.yaml \\
        --backbone dinov2 --loss arcface --subset-albums 2000 --epochs 10

    # Full B2 training
    python scripts/train.py --config configs/dataset.yaml \\
        --backbone dinov2 --loss arcface --epochs 50 --freeze-epochs 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import structlog
import torch
import yaml
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from vinylid_ml.dataset import (
    AlbumCoverDataset,
    get_eval_transforms,
    get_train_transforms,
    load_manifest,
    load_splits,
)
from vinylid_ml.eval_metrics import compute_retrieval_metrics
from vinylid_ml.losses import ArcFaceLoss, ProxyAnchorLoss, SupConLoss
from vinylid_ml.models import get_device
from vinylid_ml.training import (
    BACKBONE_DIMS,
    FineTuneModel,
    MultiViewTransform,
    TrainingConfig,
)

logger = structlog.get_logger()


# ── Helpers ─────────────────────────────────────────────────────────────


def _seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # type: ignore[reportUnknownMemberType]
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # type: ignore[reportUnknownMemberType]
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)  # type: ignore[reportUnknownMemberType]


def _get_git_sha() -> str | None:
    """Get current git commit SHA, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _manifest_hash(manifest_path: Path) -> str:
    """Compute a short hash of the manifest file for provenance."""
    h = hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:12]
    return h



def _build_llrd_param_groups(
    backbone: torch.nn.Module,
    backbone_name: str,
    backbone_lr: float,
    decay: float,
) -> list[dict[str, object]]:
    """Build parameter groups with layer-wise LR decay for ViT backbones.

    For DINOv2 ViT-S/14 (12 blocks), outer blocks (near output) get the
    highest backbone LR; inner blocks are multiplied by ``decay`` per step
    inward. Patch/pos embeddings receive the lowest LR (``decay^num_blocks``).
    The final LayerNorm receives the full ``backbone_lr``.

    Non-DINOv2 backbones fall back to a single uniform LR param group.

    Args:
        backbone: The backbone ``nn.Module``.
        backbone_name: Backbone identifier string (e.g. ``"dinov2"``).
        backbone_lr: Base backbone learning rate (post-multiplier).
        decay: LLRD decay factor per block depth (e.g. 0.9).

    Returns:
        List of ``{"params": [...], "lr": float}`` dicts for AdamW.
    """
    if backbone_name != "dinov2":
        trainable = [p for p in backbone.parameters() if p.requires_grad]
        return [{"params": trainable, "lr": backbone_lr}]

    # Discover number of transformer blocks
    block_indices = {
        int(name.split(".")[1])
        for name, _ in backbone.named_parameters()
        if name.startswith("blocks.")
    }
    num_blocks = max(block_indices) + 1 if block_indices else 12

    # Map each trainable param to its target LR
    param_lr: dict[int, float] = {}
    for name, param in backbone.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in param_lr:
            continue
        if name.startswith("blocks."):
            block_idx = int(name.split(".")[1])
            param_lr[pid] = backbone_lr * (decay ** (num_blocks - 1 - block_idx))
        elif name.startswith("norm"):
            param_lr[pid] = backbone_lr
        else:
            # patch_embed, pos_embed, cls_token, register_tokens → lowest LR
            param_lr[pid] = backbone_lr * (decay**num_blocks)

    # Group params by LR value
    lr_to_params: dict[float, list[torch.nn.Parameter]] = {}
    for _name, param in backbone.named_parameters():
        if id(param) in param_lr:
            lr_val = param_lr[id(param)]
            lr_to_params.setdefault(lr_val, []).append(param)

    groups: list[dict[str, object]] = [
        {"params": params, "lr": lr_val} for lr_val, params in sorted(lr_to_params.items())
    ]
    return groups if groups else [{"params": list(backbone.parameters()), "lr": backbone_lr}]


def _apply_backbone_training_strategy(
    model: FineTuneModel,
    unfreeze_blocks: int | None,
) -> None:
    """Apply the requested full or partial backbone unfreezing policy."""
    if unfreeze_blocks is not None:
        model.partial_unfreeze_backbone(unfreeze_blocks)
    elif model.is_backbone_frozen():
        model.unfreeze_backbone()


def _build_finetune_optimizer(
    model: FineTuneModel,
    loss_fn: torch.nn.Module,
    backbone_name: str,
    lr: float,
    weight_decay: float,
    backbone_lr_mult: float,
    llrd_decay: float | None,
) -> torch.optim.AdamW:
    """Build AdamW param groups for the model's current trainable state."""
    head_params = list(model.projection.parameters()) + list(loss_fn.parameters())
    trainable_backbone = [p for p in model.backbone.parameters() if p.requires_grad]

    if not trainable_backbone:
        return torch.optim.AdamW(
            [{"params": head_params, "lr": lr}],
            weight_decay=weight_decay,
        )

    backbone_lr = lr * backbone_lr_mult
    if llrd_decay is not None:
        backbone_groups = _build_llrd_param_groups(
            model.backbone,
            backbone_name,
            backbone_lr,
            llrd_decay,
        )
    else:
        backbone_groups = [{"params": trainable_backbone, "lr": backbone_lr}]

    return torch.optim.AdamW(
        [*backbone_groups, {"params": head_params, "lr": lr}],
        weight_decay=weight_decay,
    )


# ── Multi-view Dataset Wrapper ──────────────────────────────────────────


class MultiViewAlbumDataset(Dataset[tuple[torch.Tensor, int]]):
    """Wraps AlbumCoverDataset to produce multi-view tensors for SupCon.

    Each item returns ``(views_tensor, label)`` where ``views_tensor`` has
    shape ``(n_views, C, H, W)``.

    Args:
        manifest: Manifest DataFrame.
        splits: Split mapping.
        split_name: Split to use.
        multi_view_transform: MultiViewTransform instance.
        gallery_root: Root directory of album images.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        splits: dict[str, str],
        split_name: str,
        multi_view_transform: MultiViewTransform,
        gallery_root: Path,
    ) -> None:
        self._inner = AlbumCoverDataset(
            manifest=manifest,
            splits=splits,
            split_name=split_name,  # type: ignore[arg-type]
            transform=None,
            gallery_root=gallery_root,
        )
        self._mvt = multi_view_transform

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img, label = self._inner[idx]
        if not isinstance(img, Image.Image):
            raise TypeError(f"Expected PIL Image from AlbumCoverDataset, got {type(img)}")
        views: torch.Tensor = self._mvt(img)
        return views, label

    @property
    def num_classes(self) -> int:
        """Number of unique album classes."""
        return self._inner.num_classes

    @property
    def album_to_label(self) -> dict[str, int]:
        """Album ID to label mapping."""
        return self._inner.album_to_label


# ── Validation Evaluation ───────────────────────────────────────────────


@torch.inference_mode()
def _evaluate_val(
    model: FineTuneModel,
    val_dataset: AlbumCoverDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> dict[str, float]:
    """Evaluate model on validation split using retrieval metrics.

    Embeds all val images, splits into gallery (1 per album) and queries
    (remaining), computes cosine similarity, and returns R@1/R@5/mAP@5/MRR.

    Args:
        model: Fine-tune model in eval mode.
        val_dataset: Validation AlbumCoverDataset.
        batch_size: Batch size for embedding extraction.
        num_workers: DataLoader worker processes.
        pin_memory: Whether to pin host memory for CUDA transfer.

    Returns:
        Dict with recall_at_1, recall_at_5, map_at_5, mrr.
    """
    model.eval()
    loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    all_embeddings: list[np.ndarray] = []
    all_labels: list[int] = []

    for images, labels in loader:
        embeddings = model(images)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.extend(labels.tolist())

    embeddings_np = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    labels_np = np.array(all_labels, dtype=np.int64)

    # Split: 1 canonical image per album (first occurrence) → gallery, rest → queries
    unique_labels = np.unique(labels_np)
    gallery_indices: list[int] = []
    query_indices: list[int] = []

    for label in unique_labels:
        indices = np.where(labels_np == label)[0]
        gallery_indices.append(int(indices[0]))
        query_indices.extend(int(i) for i in indices[1:])

    if not query_indices:
        # All albums have only 1 image — no queries to evaluate
        logger.warning("val_no_queries", msg="All val albums have 1 image, skipping eval")
        return {"recall_at_1": 0.0, "recall_at_5": 0.0, "map_at_5": 0.0, "mrr": 0.0}

    gallery_emb = embeddings_np[gallery_indices]
    gallery_lab = labels_np[gallery_indices]
    query_emb = embeddings_np[query_indices]
    query_lab = labels_np[query_indices]

    # Cosine similarity (embeddings are L2-normalized)
    sim_matrix = query_emb @ gallery_emb.T

    metrics = compute_retrieval_metrics(sim_matrix, query_lab, gallery_lab)
    return {
        "recall_at_1": metrics.recall_at_1,
        "recall_at_5": metrics.recall_at_5,
        "map_at_5": metrics.map_at_5,
        "mrr": metrics.mrr,
    }


# ── Training Loop ───────────────────────────────────────────────────────


def _train_one_epoch(
    model: FineTuneModel,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader[tuple[torch.Tensor, int]],
    epoch: int,
    is_supcon: bool,
) -> float:
    """Run one training epoch.

    Args:
        model: FineTuneModel in train mode.
        loss_fn: Loss function (ArcFace, ProxyAnchor, or SupCon).
        optimizer: Optimiser.
        loader: Training DataLoader.
        epoch: Current epoch number (0-indexed).
        is_supcon: Whether using SupCon loss (multi-view input).

    Returns:
        Mean loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, labels) in enumerate(loader):
        labels = labels.to(model.device)

        if is_supcon:
            # images: (B, N_views, C, H, W) → embed each view separately
            b, n_views, c, h, w = images.shape
            flat_images = images.reshape(b * n_views, c, h, w)
            flat_emb = model(flat_images)
            embeddings = flat_emb.reshape(b, n_views, -1)  # (B, N_views, D)
        else:
            embeddings = model(images)  # (B, D)

        loss = loss_fn(embeddings, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 20 == 0:
            logger.info(
                "train_batch",
                epoch=epoch,
                batch=f"{batch_idx + 1}/{len(loader)}",
                loss=round(loss.item(), 4),
            )

    return total_loss / max(num_batches, 1)


# ── Main ────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    """Entry point for the training script."""
    parser = argparse.ArgumentParser(
        description="Fine-tune embedding model with metric learning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to dataset.yaml.")
    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov2",
        choices=list(BACKBONE_DIMS.keys()),
        help="Backbone architecture (default: dinov2).",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="arcface",
        choices=["arcface", "proxy-anchor", "supcon"],
        help="Loss function (default: arcface).",
    )
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=None,
        help="Projection dimension (default: backbone dim).",
    )
    parser.add_argument("--subset-albums", type=int, default=None, help="Train on N albums only.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes (default: 4).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--freeze-epochs", type=int, default=0, help="Epochs with backbone frozen.")
    parser.add_argument(
        "--margin",
        type=float,
        default=None,
        help="Loss margin. Default: 0.5 for ArcFace, 0.1 for Proxy-Anchor (auto-selected).",
    )
    parser.add_argument("--scale", type=float, default=64.0, help="ArcFace scale (default: 64).")
    parser.add_argument(
        "--alpha", type=float, default=32.0, help="ProxyAnchor alpha (default: 32)."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.07, help="SupCon temperature (default: 0.07)."
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Results directory.")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Model run name (default: auto-generated from backbone/loss).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience (epochs without val R@1 improvement). Disabled by default.",
    )
    parser.add_argument(
        "--backbone-lr-mult",
        type=float,
        default=0.1,
        help="Backbone LR multiplier at unfreeze (backbone_lr = lr * mult). Default 0.1; use 0.01 for ViT.",
    )
    parser.add_argument(
        "--llrd-decay",
        type=float,
        default=None,
        help="Layer-wise LR decay for ViT backbones (e.g. 0.9). DINOv2 only.",
    )
    parser.add_argument(
        "--unfreeze-blocks",
        type=int,
        default=None,
        help="Unfreeze only the last N transformer blocks. None = unfreeze all. DINOv2 only.",
    )

    args = parser.parse_args(argv)

    # Validate patience
    if args.patience is not None and args.patience < 1:
        logger.error("invalid_patience", value=args.patience, msg="--patience must be >= 1")
        sys.exit(1)
    if args.backbone_lr_mult <= 0:
        logger.error("invalid_backbone_lr_mult", value=args.backbone_lr_mult)
        sys.exit(1)
    if args.llrd_decay is not None and not (0.0 < args.llrd_decay < 1.0):
        logger.error(
            "invalid_llrd_decay", value=args.llrd_decay, msg="--llrd-decay must be in (0, 1)"
        )
        sys.exit(1)
    if args.unfreeze_blocks is not None and args.unfreeze_blocks < 1:
        logger.error(
            "invalid_unfreeze_blocks",
            value=args.unfreeze_blocks,
            msg="--unfreeze-blocks must be >= 1",
        )
        sys.exit(1)

    # ── Config ──────────────────────────────────────────────────────
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

    data_dir = (config_dir / config["paths"]["output_dir"]).resolve()
    manifest_path = data_dir / "manifest.parquet"
    splits_path = data_dir / "splits.json"
    results_dir = args.output_dir.resolve() if args.output_dir else Path.cwd() / "results"
    device = get_device()
    use_pin_memory = device.type == "cuda"
    backbone_dim = BACKBONE_DIMS[args.backbone]
    projection_dim = args.projection_dim or backbone_dim

    # Auto-select loss-appropriate margin default if not explicitly set
    if args.margin is None:
        args.margin = 0.1 if args.loss == "proxy-anchor" else 0.5

    _seed_everything(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())
    git_sha = _get_git_sha()

    # Model name for results directory
    if args.name:
        # Reject path traversal and unsafe characters
        if not re.match(r"^[A-Za-z0-9._-]+$", args.name):
            logger.error(
                "invalid_name",
                value=args.name,
                msg="--name must only contain [A-Za-z0-9._-]",
            )
            sys.exit(1)
        model_name = args.name
    else:
        loss_short = {"arcface": "af", "proxy-anchor": "pa", "supcon": "sc"}[args.loss]
        model_name = f"strategy-{args.backbone}-{loss_short}"
        if args.subset_albums:
            model_name += f"-{args.subset_albums}alb"

    run_dir = results_dir / model_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "training_start",
        backbone=args.backbone,
        loss=args.loss,
        epochs=args.epochs,
        subset_albums=args.subset_albums,
        device=str(device),
        run_dir=str(run_dir),
    )

    # ── Data ────────────────────────────────────────────────────────
    manifest = load_manifest(manifest_path)
    splits = load_splits(splits_path)

    # Optional: subset to N albums for quick strategy eval
    if args.subset_albums:
        train_album_ids = sorted({aid for aid, s in splits.items() if s == "train"})
        rng = random.Random(args.seed)
        subset_ids = set(rng.sample(train_album_ids, min(args.subset_albums, len(train_album_ids))))
        # Keep non-train splits as-is, keep only selected train albums as "train",
        # and mark unselected train albums as "excluded".
        splits_for_train = {
            aid: (s if (s != "train" or aid in subset_ids) else "excluded")
            for aid, s in splits.items()
        }
    else:
        splits_for_train = splits

    is_supcon = args.loss == "supcon"

    train_dataset: MultiViewAlbumDataset | AlbumCoverDataset
    if is_supcon:
        mvt = MultiViewTransform(get_train_transforms(), n_views=2)
        train_dataset = MultiViewAlbumDataset(
            manifest=manifest,
            splits=splits_for_train,
            split_name="train",
            multi_view_transform=mvt,
            gallery_root=gallery_root,
        )
    else:
        train_dataset = AlbumCoverDataset(
            manifest=manifest,
            splits=splits_for_train,
            split_name="train",
            transform=get_train_transforms(),
            gallery_root=gallery_root,
        )
    num_classes = train_dataset.num_classes

    val_dataset = AlbumCoverDataset(
        manifest=manifest,
        splits=splits,
        split_name="val",
        transform=get_eval_transforms(),
        gallery_root=gallery_root,
    )

    logger.info(
        "datasets_loaded",
        train_images=len(train_dataset),
        train_classes=num_classes,
        val_images=len(val_dataset),
        val_classes=val_dataset.num_classes,
    )

    train_loader = cast(
        "DataLoader[tuple[torch.Tensor, int]]",
        DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=use_pin_memory,
            drop_last=True,
        ),
    )

    # ── Model + Loss ────────────────────────────────────────────────
    model = FineTuneModel(
        backbone_name=args.backbone,  # type: ignore[arg-type]
        projection_dim=projection_dim,
        device=device,
        freeze_backbone=args.freeze_epochs > 0,
    )

    if args.loss == "arcface":
        loss_fn: torch.nn.Module = ArcFaceLoss(
            embedding_dim=projection_dim,
            num_classes=num_classes,
            margin=args.margin,
            scale=args.scale,
        ).to(device)
    elif args.loss == "proxy-anchor":
        loss_fn = ProxyAnchorLoss(
            embedding_dim=projection_dim,
            num_classes=num_classes,
            margin=args.margin,
            alpha=args.alpha,
        ).to(device)
    else:
        loss_fn = SupConLoss(temperature=args.temperature).to(device)
    if args.freeze_epochs == 0:
        _apply_backbone_training_strategy(model, args.unfreeze_blocks)

    optimizer = _build_finetune_optimizer(
        model=model,
        loss_fn=loss_fn,
        backbone_name=args.backbone,
        lr=args.lr,
        weight_decay=1e-4,
        backbone_lr_mult=args.backbone_lr_mult,
        llrd_decay=args.llrd_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Config logging ──────────────────────────────────────────────
    train_config = TrainingConfig(
        backbone=args.backbone,
        loss=args.loss,
        projection_dim=projection_dim,
        num_classes=num_classes,
        lr=args.lr,
        weight_decay=1e-4,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        subset_albums=args.subset_albums,
        freeze_epochs=args.freeze_epochs,
        margin=args.margin,
        scale=args.scale,
        alpha=args.alpha,
        temperature=args.temperature,
        n_views=2 if is_supcon else 1,
        input_size=224,
        device=str(device),
        git_sha=git_sha,
        manifest_hash=_manifest_hash(manifest_path),
        patience=args.patience,
        backbone_lr_mult=args.backbone_lr_mult,
        llrd_decay=args.llrd_decay,
        unfreeze_blocks=args.unfreeze_blocks,
        extra={
            "torch_version": torch.__version__,
            "model_name": model_name,
            "num_workers": args.num_workers,
            "timestamp": timestamp,
        },
    )
    train_config.save(run_dir / "config.json")

    # ── Training loop ───────────────────────────────────────────────
    best_val_r1 = -1.0  # ensures first epoch always saves a checkpoint
    best_epoch = 0
    no_improve_count = 0
    early_stopped = False
    training_log: list[dict[str, object]] = []

    for epoch in range(args.epochs):
        epoch_start = time.monotonic()

        # Staged unfreezing
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs:
            _apply_backbone_training_strategy(model, args.unfreeze_blocks)
            optimizer = _build_finetune_optimizer(
                model=model,
                loss_fn=loss_fn,
                backbone_name=args.backbone,
                lr=args.lr,
                weight_decay=1e-4,
                backbone_lr_mult=args.backbone_lr_mult,
                llrd_decay=args.llrd_decay,
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - epoch)
            logger.info(
                "unfreeze_at_epoch",
                epoch=epoch,
                backbone_lr=args.lr * args.backbone_lr_mult,
                llrd=args.llrd_decay is not None,
                num_param_groups=len(optimizer.param_groups),
            )

        avg_loss = _train_one_epoch(model, loss_fn, optimizer, train_loader, epoch, is_supcon)
        scheduler.step()

        # Val evaluation
        val_metrics = _evaluate_val(
            model,
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=use_pin_memory,
        )
        epoch_time = time.monotonic() - epoch_start

        # Log LR — single group (frozen), two groups (uniform unfreeze), or
        # many groups (LLRD): report min/max backbone LR + head LR.
        lr_info: dict[str, object] = {}
        n_groups = len(optimizer.param_groups)
        if n_groups == 1:
            lr_info["lr"] = round(optimizer.param_groups[0]["lr"], 8)
        elif n_groups == 2:
            lr_info["backbone_lr"] = round(optimizer.param_groups[0]["lr"], 8)
            lr_info["head_lr"] = round(optimizer.param_groups[1]["lr"], 8)
        else:
            # LLRD: backbone groups + 1 head group
            backbone_lrs = [g["lr"] for g in optimizer.param_groups[:-1]]
            lr_info["backbone_lr_min"] = round(min(backbone_lrs), 10)
            lr_info["backbone_lr_max"] = round(max(backbone_lrs), 10)
            lr_info["head_lr"] = round(optimizer.param_groups[-1]["lr"], 8)

        epoch_log: dict[str, object] = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 6),
            "val_recall_at_1": round(val_metrics["recall_at_1"], 4),
            "val_recall_at_5": round(val_metrics["recall_at_5"], 4),
            "val_map_at_5": round(val_metrics["map_at_5"], 4),
            "val_mrr": round(val_metrics["mrr"], 4),
            **lr_info,
            "epoch_time_s": round(epoch_time, 1),
        }
        training_log.append(epoch_log)

        logger.info("epoch_complete", **{k: v for k, v in epoch_log.items()})

        # Save best checkpoint
        if val_metrics["recall_at_1"] > best_val_r1:
            best_val_r1 = val_metrics["recall_at_1"]
            best_epoch = epoch
            no_improve_count = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss_state_dict": loss_fn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_recall_at_1": best_val_r1,
            }
            torch.save(checkpoint, run_dir / "best_checkpoint.pt")
            logger.info("checkpoint_saved", epoch=epoch, val_r1=round(best_val_r1, 4))
        else:
            no_improve_count += 1

        # Early stopping
        if args.patience is not None and no_improve_count >= args.patience:
            logger.info(
                "early_stopping",
                epoch=epoch,
                best_epoch=best_epoch,
                best_val_r1=round(best_val_r1, 4),
                patience=args.patience,
            )
            early_stopped = True
            break

    # ── Save results ────────────────────────────────────────────────
    total_epochs = len(training_log)
    final_metrics = {
        "best_val_recall_at_1": round(best_val_r1, 4),
        "best_epoch": best_epoch,
        "total_epochs_trained": total_epochs,
        "early_stopped": early_stopped,
        "final_epoch": training_log[-1] if training_log else {},
        "training_log": training_log,
    }
    with (run_dir / "metrics.json").open("w") as f:
        json.dump(final_metrics, f, indent=2)

    with (run_dir / "training_log.json").open("w") as f:
        json.dump(training_log, f, indent=2)

    logger.info(
        "training_complete",
        model_name=model_name,
        best_val_r1=round(best_val_r1, 4),
        total_epochs=args.epochs,
        run_dir=str(run_dir),
    )

    print(f"\nTraining complete: {model_name}")
    print(f"  Best val R@1: {best_val_r1:.4f}")
    print(f"  Results: {run_dir}")


if __name__ == "__main__":
    main()
