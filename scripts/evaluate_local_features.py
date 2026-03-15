#!/usr/bin/env python3
"""Evaluate SuperPoint + LightGlue local feature matching (model C2).

Two evaluation modes:

  ``--mode sample``  (primary — SIFT comparison):
    Phone-photo sample (55 photos, 50 high-confidence matched) vs 855-album
    test-split gallery.  Per-query A4-sscd cosine similarity selects top-50
    candidates (falls back to SuperPoint mean-descriptor if SSCD unavailable);
    LightGlue re-ranks them.  Reports R@1/R@5/mAP@5/MRR with direct SIFT
    baseline comparison.

  ``--mode test``  (full test split via A4-sscd pre-filtering):
    4 564 query images x top-K A4-sscd candidates re-ranked by LightGlue
    inlier count.  Runtime ~20 min on CPU.  Output in the same format as
    ``evaluate.py``.

Usage::

    python scripts/evaluate_local_features.py \\
        --config configs/dataset.yaml --mode sample

    python scripts/evaluate_local_features.py \\
        --config configs/dataset.yaml --mode test --top-k 50
"""

# pyright: reportPrivateUsage=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import argparse
import csv
import gc
import json
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
from numpy.typing import NDArray
from PIL import Image as PILImage

from vinylid_ml.dataset import load_manifest, load_splits
from vinylid_ml.gallery import load_embeddings
from vinylid_ml.local_features import (
    LOCAL_FEATURE_MODEL_ID,
    KeypointFeatures,
    LocalFeatureMatcher,
)
from vinylid_ml.models import SSCDEmbedder

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Gallery / query split helpers
# ---------------------------------------------------------------------------


def _select_gallery_images(
    manifest: pd.DataFrame,
    splits: dict[str, str],
) -> tuple[list[Path], list[str], list[Path], list[str]]:
    """Select canonical gallery images (highest-res per album) for the test split.

    Mirrors the logic in ``evaluate.py::_build_gallery_query_split``.

    Args:
        manifest: Full manifest DataFrame with ``image_path``, ``album_id``,
            ``width``, ``height`` columns.
        splits: Album ID -> split name mapping.

    Returns:
        ``(gallery_paths, gallery_album_ids, query_paths, query_album_ids)``
        where gallery contains exactly 1 image per test album.
    """
    test_album_ids = {aid for aid, s in splits.items() if s == "test"}
    test_df = manifest[manifest["album_id"].isin(test_album_ids)].copy()

    # Group by album
    gallery_rows: list[pd.Series] = []  # type: ignore[type-arg]
    query_rows: list[pd.Series] = []  # type: ignore[type-arg]

    for _album_id, group in test_df.groupby("album_id"):
        if len(group) == 1:
            gallery_rows.append(group.iloc[0])
        else:
            # Pick highest-resolution image as canonical gallery entry
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
# Metrics helpers
# ---------------------------------------------------------------------------


def _compute_metrics_from_inlier_matrix(
    inlier_matrix: NDArray[np.float32],
    query_labels: NDArray[np.intp],
    gallery_labels: NDArray[np.intp],
) -> dict[str, float]:
    """Compute R@1, R@5, mAP@5, MRR from an inlier-count matrix.

    Args:
        inlier_matrix: Shape ``(num_queries, num_gallery)``. Higher = better match.
        query_labels: Shape ``(num_queries,)``. Integer album label per query.
        gallery_labels: Shape ``(num_gallery,)``. Integer album label per gallery.

    Returns:
        Dict with keys ``recall_at_1``, ``recall_at_5``, ``map_at_5``, ``mrr``.
    """
    sorted_idx = np.argsort(-inlier_matrix, axis=1)
    sorted_labels = gallery_labels[sorted_idx]
    matches = sorted_labels == query_labels[:, np.newaxis]

    r_at_1 = float(matches[:, :1].any(axis=1).mean())
    r_at_5 = float(matches[:, :5].any(axis=1).mean())

    # mAP@5 — guard against gallery smaller than 5 and normalise by actual hits
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

    # MRR
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
    """Return current git commit SHA, or None if unavailable.

    Returns:
        Hex SHA string or ``None``.
    """
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
    """Save ``metrics.json`` and ``config.json`` to *run_dir*.

    Args:
        run_dir: Output directory for this evaluation run.
        metrics: Metrics to save.
        config: Configuration to save.
    """
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
    """Append a row to ``results/summary.csv`` for cross-model comparison.

    Args:
        results_dir: Top-level results directory.
        model_id: Model identifier.
        timestamp: Run timestamp string.
        metrics: Metrics dict with retrieval metrics under ``"retrieval"`` key.
        num_gallery: Number of gallery items.
        num_queries: Number of query items.
    """
    summary_path = results_dir / "summary.csv"
    write_header = not summary_path.exists()

    retrieval = metrics.get("retrieval", {})
    if not isinstance(retrieval, dict):
        retrieval = {}

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
# Pre-filter helpers
# ---------------------------------------------------------------------------


def _compute_global_descriptor(kp: KeypointFeatures) -> NDArray[np.float32]:
    """Compute a 256-dim global descriptor as score-weighted mean of keypoint descriptors.

    Fallback pre-filter when A4-sscd embeddings are unavailable.  For phone
    photo queries vs catalogue gallery images the cross-domain gap makes this
    a poor retrieval signal (recall @50 ~15%) — prefer the SSCD pre-filter.

    Args:
        kp: Keypoint features for a single image.

    Returns:
        L2-normalised 256-dim float32 vector.
    """
    weights = kp.scores.astype(np.float32)
    total = float(weights.sum())
    if total < 1e-8:
        desc = kp.descriptors.mean(axis=0).astype(np.float32)
    else:
        desc = (weights[:, None] * kp.descriptors).sum(axis=0) / total
        desc = desc.astype(np.float32)
    norm = float(np.linalg.norm(desc))
    if norm > 1e-8:
        desc = desc / norm
    return desc


def _build_gallery_sscd_matrix(
    sscd: SSCDEmbedder,
    gallery_paths: list[Path],
    data_dir: Path,
    gallery_root: Path,
) -> NDArray[np.float32]:
    """Build (N_gallery, 512) A4-sscd embedding matrix for the gallery.

    Uses pre-computed embeddings from ``data_dir/A4-sscd/`` where available;
    embeds remaining images (typically 0-2 extra albums) on-the-fly.

    Args:
        sscd: Loaded ``SSCDEmbedder`` instance.
        gallery_paths: Ordered list of gallery image paths.
        data_dir: Data directory containing ``A4-sscd/embeddings.npy``.
        gallery_root: Gallery root for relative-path lookup fallback.

    Returns:
        L2-normalised float32 array of shape ``(len(gallery_paths), 512)``.
    """
    tfm = sscd.get_transforms()

    # Build lookup from pre-computed embeddings (fast path for test-split gallery)
    a4_path_to_emb: dict[str, NDArray[np.float32]] = {}
    try:
        a4_result = load_embeddings(data_dir, "A4-sscd")
        for i, p in enumerate(a4_result.image_paths):
            a4_path_to_emb[p] = a4_result.embeddings[i].astype(np.float32)
    except FileNotFoundError:
        logger.warning("a4_sscd_not_found_embedding_all_gallery", data_dir=str(data_dir))

    embeddings: list[NDArray[np.float32]] = []
    n_cached = 0
    for gp in gallery_paths:
        emb: NDArray[np.float32] | None = a4_path_to_emb.get(str(gp))
        if emb is None:
            emb = a4_path_to_emb.get(str(gallery_root / gp.name))
        if emb is not None:
            n_cached += 1
            embeddings.append(emb)
        else:
            # Extra album image not in test split — embed on-the-fly
            with PILImage.open(gp) as img:
                t = cast("torch.Tensor", tfm(img.convert("RGB"))).unsqueeze(0)
            embeddings.append(sscd.embed(t).numpy()[0])

    logger.info(
        "gallery_sscd_matrix_built",
        n_gallery=len(gallery_paths),
        n_cached=n_cached,
        n_computed=len(gallery_paths) - n_cached,
    )
    return np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Sample mode
# ---------------------------------------------------------------------------


def _mode_sample(
    matcher: LocalFeatureMatcher,
    gallery_paths: list[Path],
    gallery_album_ids: list[str],
    manifest: pd.DataFrame,
    test_sample_dir: Path,
    test_sample_matched_csv: Path,
    run_dir: Path,
    results_dir: Path,
    timestamp: str,
    feature_cache_dir: Path,
    match_score_min: int,
    data_dir: Path,
    gallery_root: Path,
    sample_limit: int | None = None,
    sample_gallery_limit: int | None = None,
    top_k_sample: int = 50,
) -> None:
    """Run --mode sample evaluation.

    The gallery is the 855 test-split canonical images plus any additional
    canonical images for sample albums not already in the test split.
    This ensures every sample photo's correct album is retrievable.

    Gallery candidates are pre-filtered per-query using A4-sscd cosine similarity.
    LightGlue runs only on the ``top_k_sample`` most similar candidates, reducing
    per-query calls from ~856 to ~50.  A4-sscd is trained for phone→catalogue
    cross-domain matching (>90% recall @50 vs ~15% for SuperPoint mean descriptors).

    Args:
        matcher: Initialised ``LocalFeatureMatcher``.
        gallery_paths: Paths to the 855 canonical gallery images.
        gallery_album_ids: Album IDs aligned with gallery_paths.
        manifest: Full dataset manifest (to look up images for extra albums).
        test_sample_dir: Directory containing the 55 phone-photo JPEGs.
        test_sample_matched_csv: Path to ``test_sample_matched.csv``.
        run_dir: Output directory for this run.
        results_dir: Top-level results directory (for summary.csv).
        timestamp: ISO-style timestamp string.
        feature_cache_dir: Directory for ``.npz`` feature cache.
        match_score_min: Minimum match_score to include a sample entry.
        data_dir: Data directory with pre-computed A4-sscd embeddings.
        gallery_root: Gallery image root (for path normalisation fallback).
        sample_limit: Optional cap on number of matched sample queries.
        sample_gallery_limit: Optional cap on gallery size (for quick dry runs).
        top_k_sample: Gallery candidates selected by A4-sscd cosine similarity
            before LightGlue.  Falls back to SuperPoint mean descriptors if
            SSCD model unavailable.  0 = full pairwise (default: 50).
    """
    if not test_sample_matched_csv.exists():
        logger.error("sample_csv_not_found", path=str(test_sample_matched_csv))
        sys.exit(1)

    sample_df = pd.read_csv(test_sample_matched_csv)
    matched = sample_df[
        (sample_df["file_exists"] == True)  # noqa: E712
        & (sample_df["matched_album_id"].notna())
        & (sample_df["matched_album_id"].astype(str).str.len() > 0)
        & (sample_df["match_score"] >= match_score_min)
    ].copy()

    if len(matched) == 0:
        logger.error("no_matched_sample_entries", min_score=match_score_min)
        sys.exit(1)
    if sample_limit is not None and sample_limit > 0:
        matched = matched.head(sample_limit).copy()

    # ── Expand gallery to include sample albums not in test split ──────────
    gallery_album_set = set(gallery_album_ids)
    extra_album_ids = [
        str(aid)
        for aid in matched["matched_album_id"].dropna().unique()
        if str(aid) not in gallery_album_set
    ]
    if extra_album_ids:
        extra_manifest = manifest[manifest["album_id"].astype(str).isin(extra_album_ids)].copy()
        for extra_album, group in extra_manifest.groupby("album_id"):
            res = group[["width", "height"]].max(axis=1)
            best_pos = int(res.argmax())
            row = group.iloc[best_pos]
            gallery_paths = [*gallery_paths, Path(str(row["image_path"]))]
            gallery_album_ids = [*gallery_album_ids, str(extra_album)]
        logger.info(
            "gallery_expanded",
            extra_albums=len(extra_album_ids),
            total_gallery=len(gallery_paths),
        )
    if sample_gallery_limit is not None and sample_gallery_limit > 0:
        gallery_paths = gallery_paths[:sample_gallery_limit]
        gallery_album_ids = gallery_album_ids[:sample_gallery_limit]

    logger.info("sample_mode_start", num_photos=len(matched), num_gallery=len(gallery_paths))

    # ── Extract gallery features (with cache) ─────────────────────────────
    logger.info("extracting_gallery_features", n=len(gallery_paths))
    t0 = time.monotonic()
    gallery_feats = matcher.extract_features(gallery_paths, cache_dir=feature_cache_dir)
    logger.info("gallery_extraction_done", elapsed_s=round(time.monotonic() - t0, 1))

    # ── Build gallery SSCD embeddings for pre-filtering ─────────────────────
    # Primary: A4-sscd (cross-domain invariant, trained for phone→catalogue matching)
    # Fallback: SuperPoint mean descriptors (poor recall on cross-domain pairs, ~15%)
    sscd_model: SSCDEmbedder | None = None
    sscd_tfm = None
    gallery_sscd: NDArray[np.float32] | None = None
    gallery_global: NDArray[np.float32] | None = None
    if top_k_sample > 0:
        logger.info("loading_sscd_for_prefilter")
        try:
            sscd_model = SSCDEmbedder()
            sscd_tfm = sscd_model.get_transforms()
            gallery_sscd = _build_gallery_sscd_matrix(
                sscd_model, gallery_paths, data_dir, gallery_root
            )
        except Exception as exc:
            logger.warning(
                "sscd_prefilter_failed_falling_back_to_superpoint", error=str(exc)
            )
            sscd_model = None
            sscd_tfm = None
            gallery_sscd = None
            logger.info("building_gallery_global_descriptors", n=len(gallery_feats))
            gallery_global = np.array(
                [_compute_global_descriptor(gf) for gf in gallery_feats], dtype=np.float32
            )

    # Build label arrays
    unique_albums = sorted(set(gallery_album_ids))
    album_to_label = {aid: i for i, aid in enumerate(unique_albums)}
    gallery_labels = np.array([album_to_label[aid] for aid in gallery_album_ids], dtype=np.intp)

    # ── Run matching for each sample photo ────────────────────────────────
    # Offset ensures any evaluated candidate (inliers ≥ 0) outscores
    # non-evaluated items whose score is the pre-filter similarity (≤ 1).
    _offset: float = 10_000.0
    num_queries = len(matched)
    inlier_matrix = np.zeros((num_queries, len(gallery_paths)), dtype=np.float32)
    query_labels: list[int] = []
    per_query_rows: list[dict[str, object]] = []

    t0 = time.monotonic()
    for qi, (_, row) in enumerate(matched.iterrows()):
        album_id = str(row["matched_album_id"])
        filename = str(row["filename"])
        photo_path = test_sample_dir / filename

        if album_id not in album_to_label:
            logger.warning("sample_album_not_in_gallery", album_id=album_id)
            query_labels.append(-1)
            continue

        label = album_to_label[album_id]
        query_labels.append(label)

        try:
            with PILImage.open(photo_path) as _img:
                photo_img = _img.convert("RGB")
            query_feat = matcher._extractor.extract(photo_img)
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            logger.warning("sample_photo_error", photo=filename, error=str(exc))
            query_labels[-1] = -1
            continue

        # Pre-filter: select top-K gallery candidates via cosine similarity.
        # Set the row baseline to pre-filter sims so non-candidates always rank
        # below any evaluated candidate, even one with 0 LightGlue inliers.
        if gallery_sscd is not None and sscd_model is not None and sscd_tfm is not None:
            # Primary: A4-sscd cross-domain invariant pre-filter
            qt = cast("torch.Tensor", sscd_tfm(photo_img)).unsqueeze(0)
            query_sscd = sscd_model.embed(qt).numpy()[0]  # (512,)
            sims = gallery_sscd @ query_sscd  # (N_gallery,)
            k = min(top_k_sample, len(gallery_feats))
            candidate_indices: list[int] = np.argsort(-sims)[:k].tolist()
            inlier_matrix[qi] = sims  # non-candidates keep sim ∈ [-1, 1] as sentinel
        elif gallery_global is not None:
            # Fallback: SuperPoint mean descriptor (poor cross-domain recall)
            query_global = _compute_global_descriptor(query_feat)
            sims = gallery_global @ query_global  # (N_gallery,)
            k = min(top_k_sample, len(gallery_feats))
            candidate_indices = np.argsort(-sims)[:k].tolist()
            inlier_matrix[qi] = sims  # non-candidates keep sim ∈ [-1, 1] as sentinel
        else:
            candidate_indices = list(range(len(gallery_feats)))
            # Full pairwise: all items evaluated, zeros are fine as baseline

        use_offset = len(candidate_indices) < len(gallery_feats)
        query_prepared = matcher._matcher.prepare_features(query_feat)
        for gi in candidate_indices:
            g_prepared = matcher._matcher.prepare_features(gallery_feats[gi])
            inliers = float(matcher._matcher.match_num_inliers_prepared(query_prepared, g_prepared))
            # Candidates get offset+inliers so they always outscore non-candidates (sim ≤ 1)
            inlier_matrix[qi, gi] = (_offset + inliers) if use_offset else inliers
        del query_prepared

        # Per-query MPS memory cleanup (minimal cost for K=50 matches/query)
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        top1_idx = int(np.argmax(inlier_matrix[qi]))
        top1_album = gallery_album_ids[top1_idx]
        correct = top1_album == album_id
        # Report true inlier count: undo offset if candidate, clamp to 0 for non-candidates
        top1_raw = float(inlier_matrix[qi, top1_idx])
        top1_inliers_count = (
            int(top1_raw - _offset)
            if use_offset and top1_raw >= _offset
            else int(max(0.0, top1_raw))
        )
        per_query_rows.append(
            {
                "filename": filename,
                "true_album_id": album_id,
                "top1_album_id": top1_album,
                "top1_inliers": top1_inliers_count,
                "correct": correct,
            }
        )

        if (qi + 1) % 5 == 0 or (qi + 1) == num_queries:
            elapsed = time.monotonic() - t0
            logger.info(
                "sample_progress",
                done=f"{qi + 1}/{num_queries}",
                elapsed_s=round(elapsed, 1),
                per_query_s=round(elapsed / (qi + 1), 2),
            )

    # Filter out queries with unknown labels
    valid_mask = np.array(query_labels) >= 0
    inlier_matrix = inlier_matrix[valid_mask]
    query_labels_arr = np.array(query_labels, dtype=np.intp)[valid_mask]
    num_valid = int(valid_mask.sum())

    if num_valid == 0:
        logger.error("no_valid_queries_in_sample")
        sys.exit(1)

    metrics_dict = _compute_metrics_from_inlier_matrix(
        inlier_matrix, query_labels_arr, gallery_labels
    )
    metrics_dict["num_queries"] = num_valid
    metrics_dict["num_gallery"] = len(gallery_paths)
    metrics_dict["top_k_sample"] = top_k_sample

    total_elapsed = time.monotonic() - t0
    metrics_dict["latency_per_query_s"] = round(total_elapsed / num_valid, 3)

    logger.info(
        "sample_results",
        R_at_1=round(float(metrics_dict["recall_at_1"]), 4),
        R_at_5=round(float(metrics_dict["recall_at_5"]), 4),
        mAP_at_5=round(float(metrics_dict["map_at_5"]), 4),
        MRR=round(float(metrics_dict["mrr"]), 4),
        num_queries=num_valid,
        num_gallery=len(gallery_paths),
    )

    # Save
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_run_results(
        run_dir,
        {"retrieval": metrics_dict, "mode": "sample"},
        {
            "model_id": LOCAL_FEATURE_MODEL_ID,
            "mode": "sample",
            "timestamp": timestamp,
            "num_gallery": len(gallery_paths),
            "num_queries": num_valid,
            "match_score_min": match_score_min,
            "top_k_sample": top_k_sample,
            "git_sha": _get_git_sha(),
        },
    )

    # Per-query CSV
    if per_query_rows:
        per_query_path = run_dir / "per_query.csv"
        fieldnames = ["filename", "true_album_id", "top1_album_id", "top1_inliers", "correct"]
        with per_query_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_query_rows)

    _append_summary_csv(
        results_dir,
        LOCAL_FEATURE_MODEL_ID,
        timestamp,
        {"retrieval": metrics_dict},
        len(gallery_paths),
        num_valid,
    )

    print(
        f"\nC2 SuperPoint+LightGlue (sample mode, top-{top_k_sample} pre-filter):\n"
        f"  R@1={metrics_dict['recall_at_1']:.3f}  "
        f"R@5={metrics_dict['recall_at_5']:.3f}  "
        f"mAP@5={metrics_dict['map_at_5']:.3f}  "
        f"MRR={metrics_dict['mrr']:.3f}\n"
        f"  queries={num_valid}  gallery={len(gallery_paths)}  "
        f"lat={metrics_dict['latency_per_query_s']:.1f}s/query\n"
        f"  SIFT baseline: R@1=1.000 (2.3-3.5s/query)\n"
        f"  Results: {run_dir}\n"
    )


# ---------------------------------------------------------------------------
# Test mode
# ---------------------------------------------------------------------------


def _mode_test(
    matcher: LocalFeatureMatcher,
    gallery_paths: list[Path],
    gallery_album_ids: list[str],
    query_paths: list[Path],
    query_album_ids: list[str],
    a4_gallery_embeddings: NDArray[np.float32],
    a4_query_embeddings: NDArray[np.float32],
    run_dir: Path,
    results_dir: Path,
    timestamp: str,
    feature_cache_dir: Path,
    top_k: int,
    precompute_gallery_tensors: bool = False,
) -> None:
    """Run --mode test evaluation (A4-sscd pre-filter + LightGlue re-rank).

    Args:
        matcher: Initialised ``LocalFeatureMatcher``.
        gallery_paths: Canonical gallery image paths (855 images).
        gallery_album_ids: Album IDs aligned with gallery_paths.
        query_paths: Query image paths (4 564 images).
        query_album_ids: Album IDs aligned with query_paths.
        a4_gallery_embeddings: A4-sscd embeddings for gallery images (855, 512).
        a4_query_embeddings: A4-sscd embeddings for query images (4564, 512).
        run_dir: Output directory for this run.
        results_dir: Top-level results directory.
        timestamp: ISO timestamp string.
        feature_cache_dir: Directory for ``.npz`` feature cache.
        top_k: Number of A4-sscd candidates to re-rank with LightGlue.
        precompute_gallery_tensors: If true, precompute gallery tensors once for
            faster matching at the cost of higher device memory usage.
    """
    num_gallery = len(gallery_paths)
    num_queries = len(query_paths)

    logger.info(
        "test_mode_start",
        num_queries=num_queries,
        num_gallery=num_gallery,
        top_k=top_k,
    )

    # Build label arrays
    unique_albums = sorted(set(gallery_album_ids))
    album_to_label = {aid: i for i, aid in enumerate(unique_albums)}
    gallery_labels = np.array([album_to_label[aid] for aid in gallery_album_ids], dtype=np.intp)
    query_labels_arr = np.array(
        [album_to_label.get(aid, -1) for aid in query_album_ids], dtype=np.intp
    )

    # Cosine similarity matrix: (num_queries, num_gallery) — float32
    # A4 embeddings are already L2-normalised
    logger.info("computing_a4_cosine_similarity")
    a4_sim = a4_query_embeddings.astype(np.float32) @ a4_gallery_embeddings.astype(np.float32).T

    # Extract gallery LightGlue features (with cache)
    logger.info("extracting_gallery_features", n=num_gallery)
    t0 = time.monotonic()
    gallery_feats = matcher.extract_features(gallery_paths, cache_dir=feature_cache_dir)
    logger.info("gallery_extraction_done", elapsed_s=round(time.monotonic() - t0, 1))
    gallery_prepared: list[dict[str, torch.Tensor]] | None = None
    if precompute_gallery_tensors:
        logger.info("preparing_gallery_tensors", n=len(gallery_feats))
        t0 = time.monotonic()
        gallery_prepared = [matcher._matcher.prepare_features(gfeat) for gfeat in gallery_feats]
        logger.info("gallery_tensor_prep_done", elapsed_s=round(time.monotonic() - t0, 1))

    # Re-rank queries
    # inlier_matrix: (num_queries, num_gallery)
    # For top-K candidates: inlier count + large offset so they're ranked above non-candidates
    # For non-top-K: raw A4-sscd score (in [-1,1]) — ensures top-K candidates rank higher
    _offset: float = 10_000.0
    inlier_matrix = a4_sim.copy()  # Start with A4-sscd scores as baseline

    t0 = time.monotonic()
    for qi in range(num_queries):
        if query_labels_arr[qi] < 0:
            continue

        # Get top-K A4-sscd candidates
        candidate_indices = np.argsort(-a4_sim[qi])[:top_k]

        # Extract query features
        try:
            query_feat = matcher._extractor.extract(query_paths[qi])
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("query_extraction_error", idx=qi, error=str(exc))
            continue
        query_prepared = matcher._matcher.prepare_features(query_feat)

        # LightGlue match vs each candidate
        for gi in candidate_indices:
            g_prepared = (
                gallery_prepared[int(gi)]
                if gallery_prepared is not None
                else matcher._matcher.prepare_features(gallery_feats[int(gi)])
            )
            inliers = matcher._matcher.match_num_inliers_prepared(
                query_prepared, g_prepared
            )
            inlier_matrix[qi, gi] = _offset + float(inliers)
        del query_prepared

        # Free MPS/GPU memory accumulated during the inner loop
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if (qi + 1) % 500 == 0:
            elapsed = time.monotonic() - t0
            eta = elapsed / (qi + 1) * (num_queries - qi - 1)
            logger.info(
                "test_progress",
                done=f"{qi + 1}/{num_queries}",
                elapsed_s=round(elapsed, 1),
                eta_s=round(eta, 1),
            )

    # Filter valid queries (label >= 0)
    valid_mask = query_labels_arr >= 0
    inlier_matrix_valid = inlier_matrix[valid_mask]
    query_labels_valid = query_labels_arr[valid_mask]
    num_valid = int(valid_mask.sum())

    metrics_dict = _compute_metrics_from_inlier_matrix(
        inlier_matrix_valid, query_labels_valid, gallery_labels
    )
    metrics_dict["num_queries"] = num_valid
    metrics_dict["num_gallery"] = num_gallery

    total_elapsed = time.monotonic() - t0
    metrics_dict["latency_per_query_s"] = round(total_elapsed / max(1, num_valid), 3)

    logger.info(
        "test_results",
        R_at_1=round(float(metrics_dict["recall_at_1"]), 4),
        R_at_5=round(float(metrics_dict["recall_at_5"]), 4),
        mAP_at_5=round(float(metrics_dict["map_at_5"]), 4),
        MRR=round(float(metrics_dict["mrr"]), 4),
        num_queries=num_valid,
        num_gallery=num_gallery,
    )

    _save_run_results(
        run_dir,
        {"retrieval": metrics_dict, "mode": "test"},
        {
            "model_id": LOCAL_FEATURE_MODEL_ID,
            "mode": "test",
            "timestamp": timestamp,
            "top_k": top_k,
            "num_gallery": num_gallery,
            "num_queries": num_valid,
            "git_sha": _get_git_sha(),
        },
    )

    _append_summary_csv(
        results_dir,
        LOCAL_FEATURE_MODEL_ID,
        timestamp,
        {"retrieval": metrics_dict},
        num_gallery,
        num_valid,
    )

    print(
        f"\nC2 SuperPoint+LightGlue (test mode, top-{top_k} pre-filter):\n"
        f"  R@1={metrics_dict['recall_at_1']:.3f}  "
        f"R@5={metrics_dict['recall_at_5']:.3f}  "
        f"mAP@5={metrics_dict['map_at_5']:.3f}  "
        f"MRR={metrics_dict['mrr']:.3f}\n"
        f"  queries={num_valid}  gallery={num_gallery}  "
        f"lat={metrics_dict['latency_per_query_s']:.2f}s/query\n"
        f"  Results: {run_dir}\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Entry point for the local feature evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate C2 SuperPoint+LightGlue local feature matching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="Path to dataset.yaml config file (default: configs/dataset.yaml).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["sample", "test"],
        help="Evaluation mode: 'sample' (55-photo) or 'test' (full test split).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="A4-sscd top-K candidates for LightGlue re-ranking (test mode, default: 50).",
    )
    parser.add_argument(
        "--max-keypoints",
        type=int,
        default=2048,
        help="Maximum keypoints per image (default: 2048).",
    )
    parser.add_argument(
        "--match-score-min",
        type=int,
        default=60,
        help="Minimum match_score for sample CSV entries (default: 60).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional cap on number of sample-mode queries (for quick runs).",
    )
    parser.add_argument(
        "--sample-gallery-limit",
        type=int,
        default=None,
        help="Optional cap on sample-mode gallery size (for quick runs).",
    )
    parser.add_argument(
        "--top-k-sample",
        type=int,
        default=50,
        help=(
            "Sample mode: pre-filter to this many gallery candidates via A4-sscd cosine "
            "similarity (falls back to SuperPoint mean-descriptor if SSCD unavailable). "
            "0 = full pairwise matching (default: 50)."
        ),
    )
    parser.add_argument(
        "--precompute-gallery-tensors",
        action="store_true",
        help=(
            "Test mode: precompute gallery tensors once on device for faster matching "
            "(higher memory; known to be unstable on MPS — use with caution)."
        ),
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
        help="Feature cache directory (default: data/local_features/C2-superpoint-lightglue/).",
    )

    args = parser.parse_args(argv)

    # ── Load config ────────────────────────────────────────────────────────
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

    test_sample_dir = Path(str(config["paths"]["test_sample"]))
    if not test_sample_dir.is_absolute():
        test_sample_dir = (config_dir / test_sample_dir).resolve()
    data_dir = (config_dir / config["paths"]["output_dir"]).resolve()

    test_sample_matched_csv = data_dir / "test_sample_matched.csv"
    manifest_path = data_dir / "manifest.parquet"
    splits_path = data_dir / "splits.json"

    results_dir: Path = args.results_dir.resolve() if args.results_dir else Path.cwd() / "results"
    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())

    feature_cache_dir: Path = (
        args.cache_dir.resolve()
        if args.cache_dir
        else data_dir.parent / "data" / "local_features" / LOCAL_FEATURE_MODEL_ID
    )
    # Handle case where data_dir IS data/ (configs are in configs/)
    if not feature_cache_dir.exists():
        feature_cache_dir = data_dir / "local_features" / LOCAL_FEATURE_MODEL_ID

    run_dir = results_dir / LOCAL_FEATURE_MODEL_ID / timestamp

    # ── Load manifest and splits ───────────────────────────────────────────
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

    # ── Initialise matcher ─────────────────────────────────────────────────
    matcher = LocalFeatureMatcher(max_keypoints=args.max_keypoints)

    # ── Run evaluation mode ────────────────────────────────────────────────
    if args.mode == "sample":
        _mode_sample(
            matcher=matcher,
            gallery_paths=gallery_paths,
            gallery_album_ids=gallery_album_ids,
            manifest=manifest,
            test_sample_dir=test_sample_dir,
            test_sample_matched_csv=test_sample_matched_csv,
            run_dir=run_dir,
            results_dir=results_dir,
            timestamp=timestamp,
            feature_cache_dir=feature_cache_dir,
            match_score_min=args.match_score_min,
            data_dir=data_dir,
            gallery_root=gallery_root,
            sample_limit=args.sample_limit,
            sample_gallery_limit=args.sample_gallery_limit,
            top_k_sample=args.top_k_sample,
        )

    elif args.mode == "test":
        # Load A4-sscd embeddings and align with our gallery/query split
        a4_result = load_embeddings(data_dir, "A4-sscd")
        a4_paths = a4_result.image_paths
        a4_emb = a4_result.embeddings.astype(np.float32)

        # Build lookup: image_path -> embedding index in A4-sscd result
        a4_path_to_idx: dict[str, int] = {p: i for i, p in enumerate(a4_paths)}

        # Get A4-sscd embeddings for gallery and query paths (aligned with our split)
        def _lookup_embeddings(paths: list[Path]) -> NDArray[np.float32]:
            indices: list[int] = []
            for p in paths:
                found: int | None = a4_path_to_idx.get(str(p))
                if found is None:
                    # Try with gallery_root prefix if needed
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

        _mode_test(
            matcher=matcher,
            gallery_paths=gallery_paths,
            gallery_album_ids=gallery_album_ids,
            query_paths=query_paths,
            query_album_ids=query_album_ids,
            a4_gallery_embeddings=a4_gallery_emb,
            a4_query_embeddings=a4_query_emb,
            run_dir=run_dir,
            results_dir=results_dir,
            timestamp=timestamp,
            feature_cache_dir=feature_cache_dir,
            top_k=args.top_k,
            precompute_gallery_tensors=args.precompute_gallery_tensors,
        )


if __name__ == "__main__":
    main()
