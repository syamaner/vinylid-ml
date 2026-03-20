#!/usr/bin/env python3
"""Evaluate C1 DINOv2 patch-level matching on the test split.

Uses A4-sscd top-K pre-filtering (same pattern as C2 evaluation) to keep
runtime tractable: 4,564 queries x top-50 candidates = ~228k patch matching
operations instead of 3.9M full-pairwise.

Runs both ``best_avg`` and ``mutual_nn`` aggregation strategies and reports
metrics for each.

Usage::

    python scripts/evaluate_patch_matching.py --config configs/dataset.yaml
    python scripts/evaluate_patch_matching.py --config configs/dataset.yaml --top-k 50 --strategy best_avg
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import structlog
import yaml
from numpy.typing import NDArray

from vinylid_ml.dataset import load_manifest, load_splits
from vinylid_ml.gallery import load_embeddings
from vinylid_ml.patch_matching import (
    PATCH_MODEL_ID,
    DINOv2PatchExtractor,
    PatchMatcher,
    extract_with_cache,
)

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Gallery / query split helpers (mirrors evaluate.py + evaluate_local_features.py)
# ---------------------------------------------------------------------------


def _select_gallery_images(
    manifest: pd.DataFrame,
    splits: dict[str, str],
) -> tuple[list[Path], list[str], list[Path], list[str]]:
    """Select canonical gallery images (highest-res per album) for the test split.

    Args:
        manifest: Full manifest DataFrame.
        splits: Album ID -> split name mapping.

    Returns:
        ``(gallery_paths, gallery_album_ids, query_paths, query_album_ids)``.
    """
    test_album_ids = {aid for aid, s in splits.items() if s == "test"}
    test_df = manifest[manifest["album_id"].isin(test_album_ids)].copy()

    gallery_rows: list[pd.Series] = []  # type: ignore[type-arg]
    query_rows: list[pd.Series] = []  # type: ignore[type-arg]

    for _album_id, group in test_df.groupby("album_id"):
        if len(group) == 1:
            gallery_rows.append(group.iloc[0])
        else:
            res = group[["width", "height"]].max(axis=1)
            best_pos = int(res.argmax())
            gallery_rows.append(group.iloc[best_pos])
            for pos in range(len(group)):
                if pos != best_pos:
                    query_rows.append(group.iloc[pos])

    gallery_paths = [Path(str(r["image_path"])) for r in gallery_rows]
    gallery_album_ids = [str(r["album_id"]) for r in gallery_rows]
    query_paths = [Path(str(r["image_path"])) for r in query_rows]
    query_album_ids = [str(r["album_id"]) for r in query_rows]
    return gallery_paths, gallery_album_ids, query_paths, query_album_ids


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------


def _compute_metrics_from_score_matrix(
    score_matrix: NDArray[np.float32],
    query_labels: NDArray[np.intp],
    gallery_labels: NDArray[np.intp],
) -> dict[str, float]:
    """Compute R@1, R@5, mAP@5, MRR from a retrieval score matrix.

    Args:
        score_matrix: Shape ``(num_queries, num_gallery)``. Higher = better.
        query_labels: Shape ``(num_queries,)``.
        gallery_labels: Shape ``(num_gallery,)``.

    Returns:
        Dict with retrieval metrics.
    """
    sorted_idx = np.argsort(-score_matrix, axis=1)
    sorted_labels = gallery_labels[sorted_idx]
    matches = sorted_labels == query_labels[:, np.newaxis]

    r_at_1 = float(matches[:, :1].any(axis=1).mean())
    r_at_5 = float(matches[:, :5].any(axis=1).mean())

    k_max = min(5, matches.shape[1])
    ap_values: list[float] = []
    for i in range(len(query_labels)):
        ap = 0.0
        n_hits = 0
        for k in range(k_max):
            if matches[i, k]:
                n_hits += 1
                ap += n_hits / (k + 1)
        ap_values.append(ap / n_hits if n_hits > 0 else 0.0)
    map_at_5 = float(np.mean(ap_values))

    mrr_values: list[float] = []
    for i in range(len(query_labels)):
        hit_positions = np.where(matches[i])[0]
        if len(hit_positions) > 0:
            mrr_values.append(1.0 / (int(hit_positions[0]) + 1))
        else:
            mrr_values.append(0.0)
    mrr = float(np.mean(mrr_values))

    return {
        "recall_at_1": r_at_1,
        "recall_at_5": r_at_5,
        "map_at_5": map_at_5,
        "mrr": mrr,
    }


# ---------------------------------------------------------------------------
# Output helpers
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


def _save_run_results(
    run_dir: Path,
    metrics: dict[str, object],
    config: dict[str, object],
) -> None:
    """Save ``metrics.json`` and ``config.json`` to *run_dir*."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    with (run_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)


def _append_summary_csv(
    results_dir: Path,
    model_id: str,
    timestamp: str,
    metrics: dict[str, object],
    num_gallery: int,
    num_queries: int,
) -> None:
    """Append a row to ``results/summary.csv``."""
    summary_path = results_dir / "summary.csv"
    write_header = not summary_path.exists()

    retrieval: dict[str, object] = {}
    if "retrieval" in metrics and isinstance(metrics["retrieval"], dict):
        _ret: dict[str, object] = metrics["retrieval"]  # type: ignore[assignment]
        retrieval = _ret

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
    with summary_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run_evaluation(
    extractor: DINOv2PatchExtractor,
    gallery_paths: list[Path],
    gallery_album_ids: list[str],
    query_paths: list[Path],
    query_album_ids: list[str],
    a4_gallery_embeddings: NDArray[np.float32],
    a4_query_embeddings: NDArray[np.float32],
    run_dir: Path,
    results_dir: Path,
    timestamp: str,
    cache_dir: Path,
    top_k: int,
    strategy: Literal["best_avg", "mutual_nn", "both"],
) -> None:
    """Run C1 patch-level matching evaluation with A4-sscd pre-filtering.

    Args:
        extractor: Initialised ``DINOv2PatchExtractor``.
        gallery_paths: Canonical gallery image paths (855 images).
        gallery_album_ids: Album IDs aligned with gallery_paths.
        query_paths: Query image paths (4,564 images).
        query_album_ids: Album IDs aligned with query_paths.
        a4_gallery_embeddings: A4-sscd embeddings for gallery (855, 512).
        a4_query_embeddings: A4-sscd embeddings for queries (4564, 512).
        run_dir: Output directory for this run.
        results_dir: Top-level results directory.
        timestamp: ISO timestamp string.
        cache_dir: Directory for ``.npz`` patch feature cache.
        top_k: Number of A4-sscd candidates to re-rank per query.
        strategy: Matching strategy — ``"best_avg"``, ``"mutual_nn"``, or ``"both"``.
    """
    num_gallery = len(gallery_paths)
    num_queries = len(query_paths)

    logger.info(
        "c1_eval_start",
        num_queries=num_queries,
        num_gallery=num_gallery,
        top_k=top_k,
        strategy=strategy,
    )

    # Build label arrays
    unique_albums = sorted(set(gallery_album_ids))
    album_to_label = {aid: i for i, aid in enumerate(unique_albums)}
    gallery_labels = np.array([album_to_label[aid] for aid in gallery_album_ids], dtype=np.intp)
    query_labels = np.array([album_to_label.get(aid, -1) for aid in query_album_ids], dtype=np.intp)

    # A4-sscd cosine similarity matrix
    logger.info("computing_a4_cosine_similarity")
    a4_sim = a4_query_embeddings @ a4_gallery_embeddings.T

    # Extract gallery patch features (with caching)
    logger.info("extracting_gallery_patches", n=num_gallery)
    t0 = time.monotonic()
    gallery_patches = extract_with_cache(extractor, gallery_paths, cache_dir)
    logger.info(
        "gallery_extraction_done",
        elapsed_s=round(time.monotonic() - t0, 1),
    )

    strategies: list[Literal["best_avg", "mutual_nn"]] = (
        ["best_avg", "mutual_nn"] if strategy == "both" else [strategy]  # type: ignore[list-item]
    )

    # Pre-extract query patch features once to avoid duplicating the most
    # expensive step when running both strategies.
    logger.info("extracting_query_patches", n=num_queries)
    t_qe = time.monotonic()
    query_patches = extract_with_cache(extractor, query_paths, cache_dir)
    logger.info(
        "query_extraction_done",
        elapsed_s=round(time.monotonic() - t_qe, 1),
    )

    for strat in strategies:
        model_id = f"{PATCH_MODEL_ID}-{strat}"
        strat_run_dir = run_dir / strat

        logger.info("running_strategy", strategy=strat)

        # Score matrix: (num_queries, num_gallery)
        # For top-K candidates: patch match score + large offset
        # For non-top-K: raw A4-sscd score (baseline ranking)
        offset = 2.0
        score_matrix = a4_sim.copy().astype(np.float32)

        t0 = time.monotonic()
        for qi in range(num_queries):
            if query_labels[qi] < 0:
                continue

            # Get top-K A4-sscd candidates
            candidate_indices = np.argsort(-a4_sim[qi])[:top_k]

            query_pf = query_patches[qi]

            # Patch matching vs each candidate
            for gi in candidate_indices:
                result = PatchMatcher.match(query_pf, gallery_patches[int(gi)], strategy=strat)
                score_matrix[qi, gi] = offset + result.score

            if (qi + 1) % 200 == 0 or (qi + 1) == num_queries:
                elapsed = time.monotonic() - t0
                logger.info(
                    "eval_progress",
                    strategy=strat,
                    done=f"{qi + 1}/{num_queries}",
                    elapsed_s=round(elapsed, 1),
                    per_query_s=round(elapsed / (qi + 1), 2),
                )

        # Filter valid queries
        valid_mask = query_labels >= 0
        score_matrix_valid = score_matrix[valid_mask]
        query_labels_valid = query_labels[valid_mask]
        num_valid = int(valid_mask.sum())

        if num_valid == 0:
            logger.error("no_valid_queries")
            continue

        metrics_dict = _compute_metrics_from_score_matrix(
            score_matrix_valid, query_labels_valid, gallery_labels
        )
        metrics_dict["num_queries"] = num_valid
        metrics_dict["num_gallery"] = num_gallery

        total_elapsed = time.monotonic() - t0
        metrics_dict["latency_per_query_s"] = round(total_elapsed / max(1, num_valid), 3)

        logger.info(
            "c1_results",
            strategy=strat,
            R_at_1=round(float(metrics_dict["recall_at_1"]), 4),
            R_at_5=round(float(metrics_dict["recall_at_5"]), 4),
            mAP_at_5=round(float(metrics_dict["map_at_5"]), 4),
            MRR=round(float(metrics_dict["mrr"]), 4),
            num_queries=num_valid,
        )

        _save_run_results(
            strat_run_dir,
            {"retrieval": metrics_dict, "mode": "test", "strategy": strat},
            {
                "model_id": model_id,
                "mode": "test",
                "strategy": strat,
                "timestamp": timestamp,
                "top_k": top_k,
                "num_gallery": num_gallery,
                "num_queries": num_valid,
                "git_sha": _get_git_sha(),
            },
        )

        _append_summary_csv(
            results_dir,
            model_id,
            timestamp,
            {"retrieval": metrics_dict},
            num_gallery,
            num_valid,
        )

        print(
            f"\nC1 DINOv2 Patch Matching ({strat}, top-{top_k} pre-filter):\n"
            f"  R@1={metrics_dict['recall_at_1']:.3f}  "
            f"R@5={metrics_dict['recall_at_5']:.3f}  "
            f"mAP@5={metrics_dict['map_at_5']:.3f}  "
            f"MRR={metrics_dict['mrr']:.3f}\n"
            f"  queries={num_valid}  gallery={num_gallery}  "
            f"lat={metrics_dict['latency_per_query_s']:.2f}s/query\n"
            f"  Results: {strat_run_dir}\n"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Entry point for the C1 patch matching evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate C1 DINOv2 patch-level matching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="Path to dataset.yaml config file (default: configs/dataset.yaml).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="A4-sscd top-K candidates for patch re-ranking (default: 50).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="both",
        choices=["best_avg", "mutual_nn", "both"],
        help="Matching strategy: best_avg, mutual_nn, or both (default: both).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Top-level results directory (default: results/).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Patch feature cache directory (default: <data_dir>/patch_features/C1-dinov2-patches/).",
    )

    args = parser.parse_args(argv)

    # ── Load config ─────────────────────────────────────────────────────────
    config_path: Path = args.config.resolve()
    if not config_path.exists():
        logger.error("config_not_found", path=str(config_path))
        sys.exit(1)

    with config_path.open() as f:
        config = yaml.safe_load(f)

    config_dir = config_path.parent
    gallery_root = Path(str(config["paths"]["gallery_root"]))
    if not gallery_root.is_absolute():
        gallery_root = (config_dir / gallery_root).resolve()

    data_dir = (config_dir / config["paths"]["output_dir"]).resolve()
    manifest_path = data_dir / "manifest.parquet"
    splits_path = data_dir / "splits.json"

    results_dir: Path = args.results_dir.resolve() if args.results_dir else Path.cwd() / "results"
    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())

    cache_dir: Path = (
        args.cache_dir.resolve() if args.cache_dir else data_dir / "patch_features" / PATCH_MODEL_ID
    )

    run_dir = results_dir / PATCH_MODEL_ID / timestamp

    # ── Load manifest and splits ────────────────────────────────────────────
    manifest = load_manifest(manifest_path)
    splits = load_splits(splits_path)

    logger.info("building_gallery_query_split")
    gallery_paths, gallery_album_ids, query_paths, query_album_ids = _select_gallery_images(
        manifest, splits
    )
    logger.info(
        "split_built",
        num_gallery=len(gallery_paths),
        num_queries=len(query_paths),
    )

    # ── Load A4-sscd embeddings ─────────────────────────────────────────────
    a4_result = load_embeddings(data_dir, "A4-sscd")
    a4_paths = a4_result.image_paths
    a4_emb = a4_result.embeddings.astype(np.float32)
    a4_path_to_idx: dict[str, int] = {p: i for i, p in enumerate(a4_paths)}

    def _lookup_embeddings(paths: list[Path]) -> NDArray[np.float32]:
        indices: list[int] = []
        for p in paths:
            found: int | None = a4_path_to_idx.get(str(p))
            if found is None:
                found = a4_path_to_idx.get(str(gallery_root / p.name))
            if found is None:
                raise KeyError(f"Path not found in A4-sscd embeddings: {p}")
            indices.append(found)
        return a4_emb[np.array(indices)]

    try:
        a4_gallery_emb = _lookup_embeddings(gallery_paths)
        a4_query_emb = _lookup_embeddings(query_paths)
    except KeyError as exc:
        logger.error("a4_embedding_lookup_failed", error=str(exc))
        sys.exit(1)

    # ── Initialise extractor ────────────────────────────────────────────────
    extractor = DINOv2PatchExtractor()

    # ── Run evaluation ──────────────────────────────────────────────────────
    run_evaluation(
        extractor=extractor,
        gallery_paths=gallery_paths,
        gallery_album_ids=gallery_album_ids,
        query_paths=query_paths,
        query_album_ids=query_album_ids,
        a4_gallery_embeddings=a4_gallery_emb,
        a4_query_embeddings=a4_query_emb,
        run_dir=run_dir,
        results_dir=results_dir,
        timestamp=timestamp,
        cache_dir=cache_dir,
        top_k=args.top_k,
        strategy=args.strategy,
    )


if __name__ == "__main__":
    main()
