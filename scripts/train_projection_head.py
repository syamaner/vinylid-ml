#!/usr/bin/env python3
"""Train a 2-layer MLP projection head on top of frozen A4-sscd features (Phase 2).

Backbone features are **pre-extracted once** before the training loop so that
each epoch is a fast matrix operation rather than a full model forward pass.
This makes the script practical on Apple MPS where SSCD inference is ~30 ms/image.

Workflow
--------
1. Load 203 labelled phone→album pairs from ``test_complete_matched.csv``.
2. Split by album_id into train / val (``--val-fraction``, default 0.2).
3. Load A4-sscd; pre-extract 512-dim features for every phone photo and its
   matched album image.
4. Build the full 979-image evaluation gallery and pre-extract its features.
5. Train ``ProjectionHead`` with ``NTXentLoss`` + AdamW + cosine LR decay.
6. After each epoch, compute train loss and (if val set) val retrieval R@1.
7. Save the best checkpoint + ``metrics.json`` + ``config.json`` under
   ``results/E1-sscd-projhead/{timestamp}/``.

Evaluation caveat
-----------------
If ``--val-fraction 0.0`` (no val set), the final retrieval metrics are computed
on the **full 203-photo training set** — this is an in-sample upper-bound
estimate, NOT a fair out-of-sample evaluation.  The default ``--val-fraction 0.2``
reserves ~40 phone photos (by album) for held-out evaluation.

Usage::

    python scripts/train_projection_head.py \\
        --config configs/dataset.yaml \\
        --epochs 300 \\
        --batch-size 32 \\
        --val-fraction 0.2

    # No val split (uses all 203 for training; in-sample eval)
    python scripts/train_projection_head.py \\
        --config configs/dataset.yaml \\
        --val-fraction 0.0
"""

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import torch
import torch.optim as optim
import yaml
from numpy.typing import NDArray
from PIL import Image as PILImage
from PIL import ImageOps

from vinylid_ml.dataset import load_manifest, load_splits
from vinylid_ml.eval_metrics import compute_retrieval_metrics
from vinylid_ml.models import SSCDEmbedder
from vinylid_ml.projection import NTXentLoss, ProjectionHead

logger = structlog.get_logger()

#: Model identifier for Phase 2 projection head results.
_MODEL_ID: str = "E1-sscd-projhead"

#: Number of defined _tta_aug augmentation indices (0-4).
_DEFAULT_VAL_FRACTION: float = 0.2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_git_sha() -> str | None:
    """Return current git commit SHA, or None if unavailable."""
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


def _seed_everything(seed: int) -> None:
    """Seed all RNG sources for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_image_stripped(path: Path) -> PILImage.Image:
    """Load an image, apply EXIF orientation, and return a clean RGB copy.

    Args:
        path: Path to the image file.

    Returns:
        RGB PIL Image with no metadata.
    """
    with PILImage.open(path) as img:
        img = ImageOps.exif_transpose(img) or img
        img.load()
        clean = PILImage.new(img.mode, img.size)
        clean.paste(img)
    if clean.mode != "RGB":
        clean = clean.convert("RGB")
    return clean


def _extract_features(
    model: SSCDEmbedder,
    image_paths: list[Path],
    desc: str = "extracting",
) -> NDArray[np.float32]:
    """Extract A4-sscd 512-dim features for a list of image paths.

    Args:
        model: Loaded A4-sscd embedder (frozen).
        image_paths: Ordered list of image paths.
        desc: Short description for logging progress.

    Returns:
        Float32 array of shape ``(N, 512)`` with L2-normalised features.
    """
    tfm = model.get_transforms()
    features: list[NDArray[np.float32]] = []
    t0 = time.monotonic()

    for i, path in enumerate(image_paths):
        img = _load_image_stripped(path)
        tensor: torch.Tensor = tfm(img)  # type: ignore[assignment]
        emb = model.embed(tensor.unsqueeze(0)).numpy()[0].astype(np.float32)
        features.append(emb)

        if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
            elapsed = time.monotonic() - t0
            logger.info(
                "extract_progress",
                desc=desc,
                done=f"{i + 1}/{len(image_paths)}",
                elapsed_s=round(elapsed, 1),
            )

    return np.stack(features, axis=0)


def _retrieval_r_at_k(
    query_feats: NDArray[np.float32],
    query_labels: NDArray[np.intp],
    gallery_feats: NDArray[np.float32],
    gallery_labels: NDArray[np.intp],
    head: ProjectionHead,
    device: torch.device,
    k: int = 1,
) -> float:
    """Compute R@K after projecting query and gallery features through the head.

    Runs in eval mode (no dropout) so results are deterministic.

    Args:
        query_feats: (Q, 512) float32 backbone features.
        query_labels: (Q,) integer album labels aligned with query_feats.
        gallery_feats: (G, 512) float32 backbone features.
        gallery_labels: (G,) integer album labels.
        head: Trained ProjectionHead.
        device: Device for head forward pass.
        k: Retrieval cutoff.

    Returns:
        Recall@K as a float in [0, 1].
    """
    head.eval()
    with torch.inference_mode():
        z_q = head(torch.from_numpy(query_feats).to(device)).cpu().numpy().astype(np.float32)
        z_g = head(torch.from_numpy(gallery_feats).to(device)).cpu().numpy().astype(np.float32)

    scores = z_q @ z_g.T
    metrics = compute_retrieval_metrics(scores, query_labels, gallery_labels)
    return float(metrics.to_dict()[f"recall_at_{k}"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Entry point for projection head training."""
    parser = argparse.ArgumentParser(
        description="Train a projection head on frozen A4-sscd features (Phase 2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="Path to dataset.yaml config file (default: configs/dataset.yaml).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs (default: 300).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32; min 2 for NT-Xent).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="AdamW learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay (default: 1e-4).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="NT-Xent temperature (default: 0.07).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Projection head hidden layer width (default: 256).",
    )
    parser.add_argument(
        "--out-dim",
        type=int,
        default=128,
        help="Projection head output dimensionality (default: 128).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability in projection head (default: 0.2).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=_DEFAULT_VAL_FRACTION,
        help=(
            "Fraction of albums held out for validation (default: 0.2). "
            "Set to 0.0 to use all 203 pairs for training (in-sample eval)."
        ),
    )
    parser.add_argument(
        "--match-score-min",
        type=int,
        default=60,
        help="Minimum fuzzy-match score for phone photo inclusion (default: 60).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Top-level results directory (default: results/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args(argv)

    # ── Validate ─────────────────────────────────────────────────────────────
    if args.batch_size < 2:
        parser.error("--batch-size must be >= 2 (NT-Xent needs at least one negative)")
    if not 0.0 <= args.val_fraction < 1.0:
        parser.error("--val-fraction must be in [0, 1)")
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.lr <= 0:
        parser.error("--lr must be > 0")
    if args.temperature <= 0:
        parser.error("--temperature must be > 0")

    _seed_everything(args.seed)
    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())

    # ── Load config ───────────────────────────────────────────────────────────
    config_path: Path = args.config.resolve()
    if not config_path.exists():
        logger.error("config_not_found", path=str(config_path))
        sys.exit(1)

    with config_path.open() as f:
        config = yaml.safe_load(f)

    config_dir = config_path.parent
    data_dir = (config_dir / config["paths"]["output_dir"]).resolve()
    manifest_path = data_dir / "manifest.parquet"
    splits_path = data_dir / "splits.json"
    matched_csv_path = data_dir / "test_complete_matched.csv"

    photo_dir_raw = config["paths"].get("test_complete")
    if photo_dir_raw is None:
        logger.error("config_key_missing", key="paths.test_complete")
        sys.exit(1)
    photo_dir = Path(str(photo_dir_raw))
    if not photo_dir.is_absolute():
        photo_dir = (config_dir / photo_dir).resolve()

    results_dir: Path = args.results_dir.resolve() if args.results_dir else Path.cwd() / "results"

    for path, label in [
        (manifest_path, "manifest.parquet"),
        (splits_path, "splits.json"),
        (matched_csv_path, "test_complete_matched.csv"),
        (photo_dir, "test_complete directory"),
    ]:
        if not path.exists():
            logger.error("required_path_missing", label=label, path=str(path))
            sys.exit(1)

    # ── Load matched pairs ───────────────────────────────────────────────────
    manifest = load_manifest(manifest_path)
    splits = load_splits(splits_path)

    raw_matched = pd.read_csv(matched_csv_path)
    matched = raw_matched[
        (raw_matched["file_exists"] == True)  # noqa: E712
        & raw_matched["matched_album_id"].notna()
        & (raw_matched["matched_album_id"].astype(str).str.len() > 0)
        & (raw_matched["match_score"] >= args.match_score_min)
    ].copy()
    matched["matched_album_id"] = matched["matched_album_id"].astype(str)

    if len(matched) == 0:
        logger.error("no_matched_pairs", min_score=args.match_score_min)
        sys.exit(1)

    logger.info("matched_pairs_loaded", n=len(matched))

    # ── Album-level train / val split ────────────────────────────────────────
    all_album_ids = sorted(matched["matched_album_id"].unique())
    rng = np.random.default_rng(args.seed)

    if args.val_fraction > 0.0:
        n_val_albums = max(1, round(len(all_album_ids) * args.val_fraction))
        val_album_ids = set(rng.choice(all_album_ids, size=n_val_albums, replace=False).tolist())
        train_album_ids = set(all_album_ids) - val_album_ids
    else:
        val_album_ids = set()
        train_album_ids = set(all_album_ids)

    train_matched = matched[matched["matched_album_id"].isin(train_album_ids)].copy()
    val_matched = matched[matched["matched_album_id"].isin(val_album_ids)].copy()

    logger.info(
        "data_split",
        train_pairs=len(train_matched),
        val_pairs=len(val_matched),
        train_albums=len(train_album_ids),
        val_albums=len(val_album_ids),
        in_sample_eval=(args.val_fraction == 0.0),
    )

    if len(train_matched) < args.batch_size:
        logger.warning(
            "train_set_smaller_than_batch",
            train_pairs=len(train_matched),
            batch_size=args.batch_size,
            hint="Reducing batch_size to match training set size",
        )
        args.batch_size = len(train_matched)

    # ── Build manifest lookup: album_id → canonical image path ───────────────
    test_album_ids_in_splits = {aid for aid, s in splits.items() if s == "test"}
    extra_album_ids = {
        str(aid)
        for aid in matched["matched_album_id"].unique()
        if str(aid) not in test_album_ids_in_splits
    }
    gallery_album_set = test_album_ids_in_splits | extra_album_ids

    gallery_manifest = manifest[manifest["album_id"].isin(gallery_album_set)].copy()
    # Pick highest-resolution image per album (same logic as evaluate_phone_photos)
    album_to_path: dict[str, Path] = {}
    for album_id_raw, group in gallery_manifest.groupby("album_id"):
        album_id = str(album_id_raw)
        if len(group) == 1:
            row = group.iloc[0]
        else:
            res = group[["width", "height"]].max(axis=1)
            row = group.iloc[int(res.argmax())]
        album_to_path[album_id] = Path(str(row["image_path"]))

    # ── Load A4-sscd model ────────────────────────────────────────────────────
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    logger.info("loading_backbone", device=str(device))
    backbone = SSCDEmbedder(device=device)

    # ── Pre-extract phone features ────────────────────────────────────────────
    logger.info("extracting_phone_features", n_train=len(train_matched), n_val=len(val_matched))

    def _get_phone_paths(df: pd.DataFrame) -> list[Path]:
        return [photo_dir / str(row["filename"]) for _, row in df.iterrows()]

    train_phone_paths = _get_phone_paths(train_matched)
    train_phone_feats = _extract_features(backbone, train_phone_paths, "train_phone")

    val_phone_feats: NDArray[np.float32] | None = None
    if len(val_matched) > 0:
        val_phone_paths = _get_phone_paths(val_matched)
        val_phone_feats = _extract_features(backbone, val_phone_paths, "val_phone")

    # ── Pre-extract matched album features ────────────────────────────────────
    logger.info("extracting_matched_album_features")
    train_album_paths = [album_to_path[aid] for aid in train_matched["matched_album_id"]]
    train_album_feats = _extract_features(backbone, train_album_paths, "train_album")

    # ── Pre-extract gallery features (for retrieval evaluation) ───────────────
    gallery_album_ids_ordered = sorted(album_to_path.keys())
    gallery_paths = [album_to_path[aid] for aid in gallery_album_ids_ordered]

    logger.info("extracting_gallery_features", n=len(gallery_paths))
    gallery_feats = _extract_features(backbone, gallery_paths, "gallery")

    # Build gallery label arrays
    unique_gallery_albums = sorted(set(gallery_album_ids_ordered))
    album_to_label = {aid: i for i, aid in enumerate(unique_gallery_albums)}
    gallery_labels = np.array(
        [album_to_label[aid] for aid in gallery_album_ids_ordered], dtype=np.intp
    )

    # Build query label arrays
    def _make_query_labels(df: pd.DataFrame) -> NDArray[np.intp]:
        labels = []
        for _, row in df.iterrows():
            aid = str(row["matched_album_id"])
            if aid in album_to_label:
                labels.append(album_to_label[aid])
            else:
                labels.append(-1)  # shouldn't happen
        return np.array(labels, dtype=np.intp)

    train_query_labels = _make_query_labels(train_matched)
    val_query_labels = _make_query_labels(val_matched) if len(val_matched) > 0 else None

    # Convert to tensors on device for training
    t_phone = torch.from_numpy(train_phone_feats).to(device)
    t_album = torch.from_numpy(train_album_feats).to(device)

    # ── Build projection head + optimizer + scheduler ─────────────────────────
    head = ProjectionHead(
        in_dim=512,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        dropout=args.dropout,
    ).to(device)

    trainable_params = sum(p.numel() for p in head.parameters())
    logger.info(
        "projection_head_created",
        in_dim=512,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        dropout=args.dropout,
        trainable_params=trainable_params,
    )

    criterion = NTXentLoss(temperature=args.temperature)
    optimizer = optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # Cosine LR decay to 1e-5 minimum
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5,
    )

    n_train = len(t_phone)
    n_batches = math.ceil(n_train / args.batch_size)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_r1 = -1.0
    best_checkpoint: dict[str, object] = {}
    history: list[dict[str, float]] = []

    logger.info(
        "training_start",
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_train=n_train,
        n_batches_per_epoch=n_batches,
        temperature=args.temperature,
        lr=args.lr,
    )
    t_train_start = time.monotonic()

    for epoch in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0

        for batch_idx in range(n_batches):
            start = batch_idx * args.batch_size
            idx = perm[start : start + args.batch_size]

            if len(idx) < 2:
                # Last micro-batch too small for NT-Xent — skip
                continue

            z_phone = head(t_phone[idx])
            z_album = head(t_album[idx])

            loss = criterion(z_phone, z_album)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / n_batches

        # Evaluate every 10 epochs and on the final epoch
        row: dict[str, float] = {"epoch": float(epoch), "train_loss": avg_loss}

        if epoch % 10 == 0 or epoch == args.epochs:
            current_lr = float(scheduler.get_last_lr()[0])

            # Val retrieval (or in-sample if no val)
            if val_phone_feats is not None and val_query_labels is not None:
                val_r1 = _retrieval_r_at_k(
                    val_phone_feats,
                    val_query_labels,
                    gallery_feats,
                    gallery_labels,
                    head,
                    device,
                )
                row["val_r1"] = val_r1
                is_best = val_r1 > best_val_r1
                if is_best:
                    best_val_r1 = val_r1
                    best_checkpoint = {
                        k: v.clone() if isinstance(v, torch.Tensor) else v
                        for k, v in head.state_dict().items()
                    }
                logger.info(
                    "epoch_eval",
                    epoch=epoch,
                    train_loss=round(avg_loss, 4),
                    val_r1=round(val_r1, 4),
                    best_val_r1=round(best_val_r1, 4),
                    lr=round(current_lr, 6),
                    is_best=is_best,
                )
            else:
                # No val set — track in-sample train R@1
                train_r1 = _retrieval_r_at_k(
                    train_phone_feats,
                    train_query_labels,
                    gallery_feats,
                    gallery_labels,
                    head,
                    device,
                )
                row["train_r1_insample"] = train_r1
                is_best = train_r1 > best_val_r1
                if is_best:
                    best_val_r1 = train_r1
                    best_checkpoint = {
                        k: v.clone() if isinstance(v, torch.Tensor) else v
                        for k, v in head.state_dict().items()
                    }
                logger.info(
                    "epoch_eval",
                    epoch=epoch,
                    train_loss=round(avg_loss, 4),
                    train_r1_insample=round(train_r1, 4),
                    lr=round(current_lr, 6),
                    is_best=is_best,
                )
        history.append(row)

    total_elapsed = time.monotonic() - t_train_start
    logger.info(
        "training_complete",
        total_time_s=round(total_elapsed, 1),
        best_val_r1=round(best_val_r1, 4),
    )

    # ── Final evaluation with best checkpoint ─────────────────────────────────
    if best_checkpoint:
        head.load_state_dict(best_checkpoint)  # type: ignore[arg-type]

    # Val R@1 (out-of-sample, or in-sample if val_fraction=0)
    if val_phone_feats is not None and val_query_labels is not None:
        final_val_r1 = _retrieval_r_at_k(
            val_phone_feats,
            val_query_labels,
            gallery_feats,
            gallery_labels,
            head,
            device,
        )
        eval_label = "val (out-of-sample)"
        final_r1 = final_val_r1
    else:
        final_r1 = _retrieval_r_at_k(
            train_phone_feats,
            train_query_labels,
            gallery_feats,
            gallery_labels,
            head,
            device,
        )
        eval_label = "train (IN-SAMPLE upper bound)"
        final_val_r1 = None

    # Also compute full-set R@1 for reference (labelled as in-sample if val_fraction=0)
    all_phone_feats = np.concatenate(
        [train_phone_feats] + ([val_phone_feats] if val_phone_feats is not None else []),
        axis=0,
    )
    all_query_labels = np.concatenate(
        [train_query_labels] + ([val_query_labels] if val_query_labels is not None else []),
        axis=0,
    )
    full_r1 = _retrieval_r_at_k(
        all_phone_feats,
        all_query_labels,
        gallery_feats,
        gallery_labels,
        head,
        device,
    )

    print(
        f"\n{'=' * 60}\n"
        f"PROJECTION HEAD TRAINING COMPLETE\n"
        f"{'=' * 60}\n"
        f"  Train pairs:  {len(train_matched):>4}  |  Val pairs: {len(val_matched):>4}\n"
        f"  Gallery size: {len(gallery_feats):>4}\n"
        f"  Epochs:       {args.epochs}\n"
        f"  Best R@1 ({eval_label}):  {final_r1:.4f}\n"
        f"  Full 203-photo R@1 (mixed in/out): {full_r1:.4f}\n"
        f"  Baseline A4-sscd (no head):        0.7783\n"
        f"{'=' * 60}\n"
    )

    # ── Save results ──────────────────────────────────────────────────────────
    run_dir = results_dir / _MODEL_ID / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    checkpoint_path = run_dir / "projection_head.pt"
    torch.save(head.state_dict(), str(checkpoint_path))
    logger.info("checkpoint_saved", path=str(checkpoint_path))

    out_metrics: dict[str, object] = {
        "model_id": _MODEL_ID,
        "timestamp": timestamp,
        "git_sha": _get_git_sha(),
        "num_train_pairs": len(train_matched),
        "num_val_pairs": len(val_matched),
        "num_gallery": len(gallery_feats),
        "total_training_time_s": round(total_elapsed, 1),
        "best_val_r1": round(final_r1, 4),
        "best_val_r1_label": eval_label,
        "full_203_r1": round(full_r1, 4),
        "full_203_r1_label": (
            "in-sample" if args.val_fraction == 0.0 else "mixed in/out-of-sample"
        ),
        "baseline_a4sscd_r1": 0.7783,
        "history": history,
    }
    with (run_dir / "metrics.json").open("w") as f:
        json.dump(out_metrics, f, indent=2)

    out_config: dict[str, object] = {
        "model_id": _MODEL_ID,
        "timestamp": timestamp,
        "git_sha": _get_git_sha(),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "hidden_dim": args.hidden_dim,
        "out_dim": args.out_dim,
        "dropout": args.dropout,
        "val_fraction": args.val_fraction,
        "match_score_min": args.match_score_min,
        "backbone": "A4-sscd (frozen)",
        "trainable_params": trainable_params,
        "in_sample_eval": args.val_fraction == 0.0,
    }
    with (run_dir / "config.json").open("w") as f:
        json.dump(out_config, f, indent=2)

    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Metrics:    {run_dir / 'metrics.json'}")
    print(f"  Config:     {run_dir / 'config.json'}\n")


if __name__ == "__main__":
    main()
