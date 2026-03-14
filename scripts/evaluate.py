#!/usr/bin/env python3
"""Run retrieval evaluation on pre-computed gallery embeddings.

Loads embeddings from embed_gallery.py, constructs a gallery/query split,
computes all retrieval metrics + embedding space analysis, and saves
structured output (JSON, CSV, HTML report) to results/{model_id}/{timestamp}/.

Usage:
    python scripts/evaluate.py --config configs/dataset.yaml --model A1-dinov2-cls
    python scripts/evaluate.py --config configs/dataset.yaml --model A2-openclip --split val
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import yaml
from numpy.typing import NDArray

from vinylid_ml.dataset import load_manifest, load_splits
from vinylid_ml.eval_metrics import (
    compute_confidence_calibration,
    compute_nn_ambiguity,
    compute_retrieval_metrics,
    compute_stratified_metrics,
)
from vinylid_ml.gallery import load_embeddings
from vinylid_ml.report import generate_report

logger = structlog.get_logger()


@dataclass
class GalleryQuerySplit:
    """Result of splitting embeddings into gallery and query sets.

    Attributes:
        gallery_embeddings: Shape (N_gallery, D). One canonical image per album.
        gallery_labels: Shape (N_gallery,). Integer label per gallery item.
        query_embeddings: Shape (N_query, D). Remaining images as queries.
        query_labels: Shape (N_query,). Integer label per query.
        query_resolutions: Shape (N_query,). Max dimension per query source image.
        album_image_counts: Mapping of label -> total image count in the split.
        label_to_album_id: Mapping of integer label -> album_id string.
    """

    gallery_embeddings: NDArray[np.floating]
    gallery_labels: NDArray[np.integer]
    query_embeddings: NDArray[np.floating]
    query_labels: NDArray[np.integer]
    query_resolutions: NDArray[np.integer]
    album_image_counts: dict[int, int]
    label_to_album_id: dict[int, str]


def _build_gallery_query_split(
    embeddings: NDArray[np.floating],
    album_ids: list[str],
    manifest: pd.DataFrame,
    image_paths: list[str],
) -> GalleryQuerySplit:
    """Split embeddings into gallery (1 canonical per album) and queries.

    For each album with >1 image, selects the largest-resolution image as
    the canonical gallery item. Remaining images become queries. Single-image
    albums go to gallery only.

    Args:
        embeddings: Shape (N, D). All embeddings for the split.
        album_ids: Album ID strings aligned with embeddings.
        manifest: Manifest DataFrame (with width/height columns) for the split.
        image_paths: Image path strings aligned with embeddings.

    Returns:
        GalleryQuerySplit with separated gallery and query data.
    """
    # Build album_id -> integer label mapping (deterministic)
    unique_albums = sorted(set(album_ids))
    album_to_label = {aid: i for i, aid in enumerate(unique_albums)}
    label_to_album_id = {i: aid for aid, i in album_to_label.items()}

    # Build a lookup from image_path to manifest row for resolution data
    manifest_lookup: dict[str, pd.Series] = {}  # type: ignore[type-arg]
    for _, row in manifest.iterrows():
        manifest_lookup[str(row["image_path"])] = row

    # Group indices by album
    album_groups: dict[str, list[int]] = {}
    for idx, aid in enumerate(album_ids):
        album_groups.setdefault(aid, []).append(idx)

    gallery_indices: list[int] = []
    query_indices: list[int] = []

    for _aid, indices in album_groups.items():
        if len(indices) == 1:
            # Single-image album — gallery only
            gallery_indices.append(indices[0])
        else:
            # Multi-image: pick canonical (largest resolution) for gallery
            best_idx = indices[0]
            best_resolution = 0
            for idx in indices:
                row = manifest_lookup.get(image_paths[idx])
                if row is not None:
                    res = max(int(row.get("width", 0)), int(row.get("height", 0)))
                else:
                    res = 0
                if res > best_resolution:
                    best_resolution = res
                    best_idx = idx
            gallery_indices.append(best_idx)
            query_indices.extend(idx for idx in indices if idx != best_idx)

    gallery_indices.sort()
    query_indices.sort()

    # Build label arrays
    all_labels = np.array([album_to_label[aid] for aid in album_ids])
    gallery_labels = all_labels[gallery_indices]
    query_labels = all_labels[query_indices]

    # Build resolution array for queries
    query_resolutions_list: list[int] = []
    for idx in query_indices:
        row = manifest_lookup.get(image_paths[idx])
        if row is not None:
            query_resolutions_list.append(max(int(row.get("width", 0)), int(row.get("height", 0))))
        else:
            query_resolutions_list.append(0)

    # Album image counts (total images in the split, not just gallery)
    album_image_counts = {album_to_label[aid]: len(idxs) for aid, idxs in album_groups.items()}

    return GalleryQuerySplit(
        gallery_embeddings=embeddings[gallery_indices].astype(np.float32),
        gallery_labels=gallery_labels,
        query_embeddings=embeddings[query_indices].astype(np.float32),
        query_labels=query_labels,
        query_resolutions=np.array(query_resolutions_list, dtype=np.int64),
        album_image_counts=album_image_counts,
        label_to_album_id=label_to_album_id,
    )


def _compute_similarity_matrix(
    query_embeddings: NDArray[np.floating],
    gallery_embeddings: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute cosine similarity matrix between queries and gallery.

    Both inputs are assumed to be L2-normalized. Computation is done in
    float32 for numerical precision even if inputs are float16.

    Args:
        query_embeddings: Shape (N_query, D).
        gallery_embeddings: Shape (N_gallery, D).

    Returns:
        Similarity matrix of shape (N_query, N_gallery).
    """
    q = query_embeddings.astype(np.float32)
    g = gallery_embeddings.astype(np.float32)
    sim: NDArray[np.floating] = q @ g.T
    return sim


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


def _save_results(
    run_dir: Path,
    metrics_dict: dict[str, object],
    config_dict: dict[str, object],
    nn_ambiguity_data: list[dict[str, object]],
    calibration_data: list[dict[str, object]],
    per_query_data: list[dict[str, object]],
) -> None:
    """Save all structured output files to the run directory.

    Args:
        run_dir: Output directory for this evaluation run.
        metrics_dict: Combined metrics + stratified metrics + metadata.
        config_dict: Run configuration.
        nn_ambiguity_data: Per-gallery-item NN ambiguity rows.
        calibration_data: Calibration bin rows.
        per_query_data: Per-query prediction rows.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json
    with (run_dir / "metrics.json").open("w") as f:
        json.dump(metrics_dict, f, indent=2)

    # config.json
    with (run_dir / "config.json").open("w") as f:
        json.dump(config_dict, f, indent=2)

    # nn_ambiguity.csv
    if nn_ambiguity_data:
        with (run_dir / "nn_ambiguity.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["gallery_idx", "album_label", "nn_distance", "album_image_count"]
            )
            writer.writeheader()
            writer.writerows(nn_ambiguity_data)

    # calibration.csv
    if calibration_data:
        with (run_dir / "calibration.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["bin_lower", "bin_upper", "accuracy", "count"])
            writer.writeheader()
            writer.writerows(calibration_data)

    # per_query.csv
    if per_query_data:
        with (run_dir / "per_query.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["query_idx", "query_label", "top1_label", "top1_score", "correct"]
            )
            writer.writeheader()
            writer.writerows(per_query_data)


def _append_summary_csv(
    results_dir: Path,
    model_id: str,
    timestamp: str,
    metrics_dict: dict[str, object],
    num_gallery: int,
    num_queries: int,
) -> None:
    """Append a row to the top-level summary.csv for cross-run comparison.

    Args:
        results_dir: Top-level results directory.
        model_id: Model identifier.
        timestamp: Run timestamp.
        metrics_dict: Metrics dictionary with retrieval metrics.
        num_gallery: Number of gallery items.
        num_queries: Number of query items.
    """
    summary_path = results_dir / "summary.csv"
    write_header = not summary_path.exists()

    fieldnames = [
        "model_id",
        "timestamp",
        "recall_at_1",
        "recall_at_5",
        "map_at_5",
        "mrr",
        "num_gallery",
        "num_queries",
    ]

    retrieval_raw = metrics_dict.get("retrieval", {})
    # isinstance narrows object → dict[Unknown, Unknown]; extract values explicitly.
    retrieval: dict[str, object] = {}
    if isinstance(retrieval_raw, dict):
        for k, v in retrieval_raw.items():  # type: ignore[reportUnknownVariableType]
            if isinstance(k, str):
                retrieval[k] = v

    row: dict[str, object] = {
        "model_id": model_id,
        "timestamp": timestamp,
        "recall_at_1": retrieval.get("recall_at_1", ""),
        "recall_at_5": retrieval.get("recall_at_5", ""),
        "map_at_5": retrieval.get("map_at_5", ""),
        "mrr": retrieval.get("mrr", ""),
        "num_gallery": num_gallery,
        "num_queries": num_queries,
    }

    with summary_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Run retrieval evaluation on pre-computed embeddings.",
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
        help="Model ID (e.g., A1-dinov2-cls, A2-openclip).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Which split to evaluate (default: test).",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=None,
        help="Directory containing pre-computed embeddings (default: from config).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Top-level results directory (default: results/).",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=10,
        help="Number of bins for calibration curve (default: 10).",
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

    # Resolve embeddings dir
    if args.embeddings_dir is not None:
        embeddings_dir = args.embeddings_dir.resolve()
    else:
        embeddings_dir = (config_dir / config["paths"]["output_dir"]).resolve()

    # Resolve output dir
    if args.output_dir is not None:
        results_dir = args.output_dir.resolve()
    else:
        results_dir = Path.cwd() / "results"

    # Resolve manifest and splits paths
    data_dir = (config_dir / config["paths"]["output_dir"]).resolve()
    manifest_path = data_dir / "manifest.parquet"
    splits_path = data_dir / "splits.json"

    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())
    git_sha = _get_git_sha()

    logger.info(
        "evaluation_start",
        model=args.model,
        split=args.split,
        embeddings_dir=str(embeddings_dir),
        results_dir=str(results_dir),
    )

    # ── Load data ──────────────────────────────────────────────────────
    embedding_result = load_embeddings(embeddings_dir, args.model)
    manifest = load_manifest(manifest_path)
    splits = load_splits(splits_path)

    # Filter manifest to the requested split
    album_ids_in_split = {aid for aid, s in splits.items() if s == args.split}
    split_manifest = manifest[manifest["album_id"].astype(str).isin(album_ids_in_split)]

    logger.info(
        "data_loaded",
        num_embeddings=len(embedding_result.image_paths),
        num_manifest_rows=len(split_manifest),
        num_albums_in_split=len(album_ids_in_split),
    )

    # ── Build gallery/query split ──────────────────────────────────────
    split = _build_gallery_query_split(
        embedding_result.embeddings,
        embedding_result.album_ids,
        split_manifest,
        embedding_result.image_paths,
    )

    num_gallery = len(split.gallery_labels)
    num_queries = len(split.query_labels)

    logger.info(
        "split_constructed",
        num_gallery=num_gallery,
        num_queries=num_queries,
        num_albums=len(split.label_to_album_id),
    )

    if num_queries == 0:
        logger.warning(
            "no_queries",
            msg="No query images (all albums have exactly 1 image). "
            "Run augment_queries.py first for single-image evaluation.",
        )
        sys.exit(0)

    # ── Compute similarity matrix ──────────────────────────────────────
    similarity = _compute_similarity_matrix(split.query_embeddings, split.gallery_embeddings)

    # ── Compute metrics ────────────────────────────────────────────────
    retrieval_metrics = compute_retrieval_metrics(
        similarity, split.query_labels, split.gallery_labels
    )

    stratified_metrics = compute_stratified_metrics(
        similarity,
        split.query_labels,
        split.gallery_labels,
        album_image_counts=split.album_image_counts,
        query_resolutions=split.query_resolutions,
    )

    logger.info(
        "metrics_computed",
        recall_at_1=round(retrieval_metrics.recall_at_1, 4),
        recall_at_5=round(retrieval_metrics.recall_at_5, 4),
        map_at_5=round(retrieval_metrics.map_at_5, 4),
        mrr=round(retrieval_metrics.mrr, 4),
    )

    # ── Embedding space analysis ───────────────────────────────────────
    nn_ambiguity = compute_nn_ambiguity(
        split.gallery_embeddings,
        split.gallery_labels,
        album_image_counts=split.album_image_counts,
    )

    calibration = compute_confidence_calibration(
        similarity,
        split.query_labels,
        split.gallery_labels,
        num_bins=args.num_bins,
    )

    # ── Build per-query predictions ────────────────────────────────────
    top1_indices = np.argmax(similarity, axis=1)
    top1_scores = similarity[np.arange(num_queries), top1_indices]
    top1_labels = split.gallery_labels[top1_indices]
    correct = top1_labels == split.query_labels

    per_query_data: list[dict[str, object]] = []
    for i in range(num_queries):
        per_query_data.append(
            {
                "query_idx": i,
                "query_label": int(split.query_labels[i]),
                "top1_label": int(top1_labels[i]),
                "top1_score": round(float(top1_scores[i]), 6),
                "correct": bool(correct[i]),
            }
        )

    # ── Build NN ambiguity CSV data ────────────────────────────────────
    nn_ambiguity_csv: list[dict[str, object]] = []
    for i in range(num_gallery):
        label = int(nn_ambiguity.labels[i])
        nn_ambiguity_csv.append(
            {
                "gallery_idx": i,
                "album_label": label,
                "nn_distance": round(float(nn_ambiguity.nn_similarities[i]), 6),
                "album_image_count": nn_ambiguity.album_image_counts.get(label, 1),
            }
        )

    # ── Build calibration CSV data ─────────────────────────────────────
    calibration_csv: list[dict[str, object]] = []
    for i in range(len(calibration.bin_accuracies)):
        acc = calibration.bin_accuracies[i]
        calibration_csv.append(
            {
                "bin_lower": round(float(calibration.bin_edges[i]), 6),
                "bin_upper": round(float(calibration.bin_edges[i + 1]), 6),
                "accuracy": round(float(acc), 6) if not np.isnan(acc) else "",
                "count": int(calibration.bin_counts[i]),
            }
        )

    # ── Save structured output ─────────────────────────────────────────
    run_dir = results_dir / args.model / timestamp

    metrics_dict: dict[str, object] = {
        "model_id": args.model,
        "split": args.split,
        "timestamp": timestamp,
        "git_sha": git_sha,
        "num_gallery": num_gallery,
        "num_queries": num_queries,
        "retrieval": retrieval_metrics.to_dict(),
        "stratified": stratified_metrics.to_dict(),
    }

    config_dict: dict[str, object] = {
        "model_id": args.model,
        "split": args.split,
        "embeddings_dir": str(embeddings_dir),
        "config_path": str(config_path),
        "gallery_root": str(gallery_root),
        "num_bins": args.num_bins,
        "git_sha": git_sha,
        "timestamp": timestamp,
    }

    _save_results(
        run_dir,
        metrics_dict,
        config_dict,
        nn_ambiguity_csv,
        calibration_csv,
        per_query_data,
    )

    # ── Append to summary.csv ──────────────────────────────────────────
    _append_summary_csv(
        results_dir,
        args.model,
        timestamp,
        metrics_dict,
        num_gallery,
        num_queries,
    )

    # ── Generate HTML report ───────────────────────────────────────────
    report_path = generate_report(
        run_dir,
        retrieval_metrics,
        stratified_metrics,
        nn_ambiguity,
        calibration,
        model_id=args.model,
        split=args.split,
        timestamp=timestamp,
        num_gallery=num_gallery,
        git_sha=git_sha,
    )

    logger.info(
        "evaluation_complete",
        run_dir=str(run_dir),
        report=str(report_path),
        recall_at_1=round(retrieval_metrics.recall_at_1, 4),
    )


if __name__ == "__main__":
    main()
