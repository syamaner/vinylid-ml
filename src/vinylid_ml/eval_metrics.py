"""Retrieval evaluation metrics for album cover recognition.

All metric computations are centralized here — no ad-hoc calculations in scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CalibrationResult",
    "NNAmbiguityResult",
    "RetrievalMetrics",
    "StratifiedMetrics",
    "compute_confidence_calibration",
    "compute_nn_ambiguity",
    "compute_retrieval_metrics",
    "compute_stratified_metrics",
]


@dataclass(frozen=True)
class RetrievalMetrics:
    """Aggregate retrieval metrics for a set of queries.

    Attributes:
        recall_at_1: Fraction of queries where correct album is top-1.
        recall_at_5: Fraction of queries where correct album is in top-5.
        map_at_5: Mean average precision at 5.
        mrr: Mean reciprocal rank across all queries.
        num_queries: Total number of queries evaluated.
    """

    recall_at_1: float
    recall_at_5: float
    map_at_5: float
    mrr: float
    num_queries: int

    def to_dict(self) -> dict[str, float | int]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "recall_at_1": self.recall_at_1,
            "recall_at_5": self.recall_at_5,
            "map_at_5": self.map_at_5,
            "mrr": self.mrr,
            "num_queries": self.num_queries,
        }


@dataclass
class StratifiedMetrics:
    """Metrics broken down by subgroup for diagnostic analysis.

    Attributes:
        by_album_image_count: Recall@1 for albums with 1 / 2-5 / 6+ gallery images.
        by_resolution_bucket: Recall@1 for source images <400px / 400-800px / 800+px.
        by_augmentation_type: Recall@1 per augmentation type (if applicable).
    """

    by_album_image_count: dict[str, float] = field(default_factory=lambda: dict[str, float]())
    by_resolution_bucket: dict[str, float] = field(default_factory=lambda: dict[str, float]())
    by_augmentation_type: dict[str, float] = field(default_factory=lambda: dict[str, float]())

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "by_album_image_count": self.by_album_image_count,
            "by_resolution_bucket": self.by_resolution_bucket,
            "by_augmentation_type": self.by_augmentation_type,
        }


def compute_retrieval_metrics(
    similarity_matrix: NDArray[np.floating],
    query_labels: NDArray[np.integer],
    gallery_labels: NDArray[np.integer],
    k_values: tuple[int, ...] = (1, 5),
) -> RetrievalMetrics:
    """Compute retrieval metrics from a query-gallery similarity matrix.

    Args:
        similarity_matrix: Shape (num_queries, num_gallery). Higher = more similar.
        query_labels: Shape (num_queries,). Album ID for each query.
        gallery_labels: Shape (num_gallery,). Album ID for each gallery item.
        k_values: Values of K for Recall@K and mAP@K. Must include 1 and 5.

    Returns:
        RetrievalMetrics with Recall@1, Recall@5, mAP@5, and MRR.

    Raises:
        ValueError: If matrix dimensions don't match label arrays.
    """
    num_queries, num_gallery = similarity_matrix.shape

    if len(query_labels) != num_queries:
        raise ValueError(f"query_labels length {len(query_labels)} != matrix rows {num_queries}")
    if len(gallery_labels) != num_gallery:
        raise ValueError(
            f"gallery_labels length {len(gallery_labels)} != matrix cols {num_gallery}"
        )

    # Sort gallery indices by descending similarity for each query
    sorted_indices = np.argsort(-similarity_matrix, axis=1)

    # Build match matrix: (num_queries, num_gallery) boolean
    # sorted_gallery_labels[i, j] = label of the j-th ranked gallery item for query i
    sorted_gallery_labels = gallery_labels[sorted_indices]
    matches = sorted_gallery_labels == query_labels[:, np.newaxis]

    max_k = max(k_values)

    # Recall@K: is the correct label in the top-K?
    recall_at: dict[int, float] = {}
    for k in k_values:
        hits = matches[:, :k].any(axis=1)
        recall_at[k] = float(hits.mean())

    # mAP@K: mean average precision at K
    map_at_5 = _compute_map_at_k(matches, k=max_k)

    # MRR: mean reciprocal rank
    mrr = _compute_mrr(matches)

    return RetrievalMetrics(
        recall_at_1=recall_at.get(1, 0.0),
        recall_at_5=recall_at.get(5, 0.0),
        map_at_5=map_at_5,
        mrr=mrr,
        num_queries=num_queries,
    )


def compute_stratified_metrics(
    similarity_matrix: NDArray[np.floating],
    query_labels: NDArray[np.integer],
    gallery_labels: NDArray[np.integer],
    *,
    album_image_counts: dict[int, int] | None = None,
    query_resolutions: NDArray[np.integer] | None = None,
    query_augmentation_types: list[str] | None = None,
) -> StratifiedMetrics:
    """Compute Recall@1 stratified by subgroups.

    All computed from the same similarity matrix — no extra model inference.

    Args:
        similarity_matrix: Shape (num_queries, num_gallery).
        query_labels: Shape (num_queries,). Album ID for each query.
        gallery_labels: Shape (num_gallery,). Album ID for each gallery item.
        album_image_counts: Mapping of album_id -> number of gallery images.
        query_resolutions: Shape (num_queries,). Max dimension (px) of each query's
            source gallery image (not the query itself).
        query_augmentation_types: Per-query augmentation type label, if applicable.

    Returns:
        StratifiedMetrics with breakdowns by album image count, resolution, and augmentation.
    """
    # Compute per-query top-1 correctness
    top1_indices = np.argmax(similarity_matrix, axis=1)
    top1_labels = gallery_labels[top1_indices]
    correct = top1_labels == query_labels

    result = StratifiedMetrics()

    # Stratify by album image count
    if album_image_counts is not None:
        buckets: dict[str, list[bool]] = {"1": [], "2-5": [], "6+": []}
        for i, label in enumerate(query_labels):
            count = album_image_counts.get(int(label), 1)
            if count == 1:
                buckets["1"].append(bool(correct[i]))
            elif count <= 5:
                buckets["2-5"].append(bool(correct[i]))
            else:
                buckets["6+"].append(bool(correct[i]))

        for bucket_name, hits in buckets.items():
            if hits:
                result.by_album_image_count[bucket_name] = sum(hits) / len(hits)

    # Stratify by source resolution
    if query_resolutions is not None:
        res_buckets: dict[str, list[bool]] = {"<400px": [], "400-800px": [], "800+px": []}
        for i, res in enumerate(query_resolutions):
            if res < 400:
                res_buckets["<400px"].append(bool(correct[i]))
            elif res <= 800:
                res_buckets["400-800px"].append(bool(correct[i]))
            else:
                res_buckets["800+px"].append(bool(correct[i]))

        for bucket_name, hits in res_buckets.items():
            if hits:
                result.by_resolution_bucket[bucket_name] = sum(hits) / len(hits)

    # Stratify by augmentation type
    if query_augmentation_types is not None:
        aug_buckets: dict[str, list[bool]] = {}
        for i, aug_type in enumerate(query_augmentation_types):
            aug_buckets.setdefault(aug_type, []).append(bool(correct[i]))

        for aug_name, hits in aug_buckets.items():
            if hits:
                result.by_augmentation_type[aug_name] = sum(hits) / len(hits)

    return result


@dataclass(frozen=True)
class NNAmbiguityResult:
    """Per-gallery-item nearest-neighbor ambiguity data.

    For each gallery embedding, records the cosine similarity to the nearest
    embedding from a *different* album. High similarity to a different album
    indicates potential false-positive risk.

    Attributes:
        nn_similarities: Shape (N_gallery,). Max cosine similarity to a
            different-album gallery item for each embedding.
        labels: Shape (N_gallery,). Album label for each gallery item.
        album_image_counts: Mapping of album label -> number of gallery images.
    """

    nn_similarities: NDArray[np.floating]
    labels: NDArray[np.integer]
    album_image_counts: dict[int, int]

    def to_dict(self) -> dict[str, object]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "nn_similarities": self.nn_similarities.tolist(),
            "labels": self.labels.tolist(),
            "album_image_counts": self.album_image_counts,
        }


@dataclass(frozen=True)
class CalibrationResult:
    """Confidence calibration curve data.

    Bins top-1 cosine similarity scores and computes the actual accuracy
    within each bin. A well-calibrated model has accuracy ≈ bin midpoint.

    Attributes:
        bin_edges: Shape (num_bins + 1,). Edges of the similarity score bins.
        bin_accuracies: Shape (num_bins,). Fraction of correct top-1 predictions
            per bin. NaN for empty bins.
        bin_counts: Shape (num_bins,). Number of queries in each bin.
    """

    bin_edges: NDArray[np.floating]
    bin_accuracies: NDArray[np.floating]
    bin_counts: NDArray[np.integer]

    def to_dict(self) -> dict[str, object]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "bin_edges": self.bin_edges.tolist(),
            "bin_accuracies": [float(x) if not np.isnan(x) else None for x in self.bin_accuracies],
            "bin_counts": self.bin_counts.tolist(),
        }


def compute_nn_ambiguity(
    gallery_embeddings: NDArray[np.floating],
    gallery_labels: NDArray[np.integer],
    album_image_counts: dict[int, int] | None = None,
) -> NNAmbiguityResult:
    """Compute nearest-neighbor ambiguity for each gallery embedding.

    For each gallery embedding, find the maximum cosine similarity to any
    gallery embedding belonging to a *different* album. This reveals which
    albums are at risk of false-positive matches.

    Args:
        gallery_embeddings: Shape (N, D). L2-normalized gallery embeddings.
        gallery_labels: Shape (N,). Album label per gallery item.
        album_image_counts: Optional mapping of label -> image count. If None,
            counts are inferred from gallery_labels.

    Returns:
        NNAmbiguityResult with per-item NN similarities and metadata.

    Raises:
        ValueError: If embeddings and labels have mismatched lengths, or if
            fewer than 2 distinct album labels are present.
    """
    n = len(gallery_embeddings)
    if len(gallery_labels) != n:
        raise ValueError(f"gallery_labels length {len(gallery_labels)} != embeddings count {n}")

    unique_labels = set(gallery_labels.tolist())
    if len(unique_labels) < 2:
        raise ValueError(
            "compute_nn_ambiguity requires at least 2 distinct album labels, "
            f"got {len(unique_labels)}"
        )

    if album_image_counts is None:
        album_image_counts = {}
        for label in gallery_labels:
            album_image_counts[int(label)] = album_image_counts.get(int(label), 0) + 1

    # Compute pairwise cosine similarity (embeddings assumed L2-normalized)
    embeddings_f32 = gallery_embeddings.astype(np.float32)
    sim_matrix = embeddings_f32 @ embeddings_f32.T

    # Mask same-album pairs with -inf so they're never selected as NN
    same_album_mask = gallery_labels[:, np.newaxis] == gallery_labels[np.newaxis, :]
    sim_matrix[same_album_mask] = -np.inf

    # For each item, find max similarity to a different-album item
    nn_similarities = np.max(sim_matrix, axis=1)

    return NNAmbiguityResult(
        nn_similarities=nn_similarities,
        labels=gallery_labels,
        album_image_counts=album_image_counts,
    )


def compute_confidence_calibration(
    similarity_matrix: NDArray[np.floating],
    query_labels: NDArray[np.integer],
    gallery_labels: NDArray[np.integer],
    *,
    num_bins: int = 10,
) -> CalibrationResult:
    """Compute confidence calibration curve from query-gallery similarity.

    For each query, takes the top-1 similarity score and whether the top-1
    prediction was correct. Bins scores and computes accuracy per bin.

    Args:
        similarity_matrix: Shape (num_queries, num_gallery). Higher = more similar.
        query_labels: Shape (num_queries,). Album ID per query.
        gallery_labels: Shape (num_gallery,). Album ID per gallery item.
        num_bins: Number of bins for the calibration curve.

    Returns:
        CalibrationResult with bin edges, accuracies, and counts.

    Raises:
        ValueError: If num_bins < 1 or dimensions are mismatched.
    """
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}")

    num_queries = similarity_matrix.shape[0]
    if len(query_labels) != num_queries:
        raise ValueError(f"query_labels length {len(query_labels)} != matrix rows {num_queries}")
    if len(gallery_labels) != similarity_matrix.shape[1]:
        raise ValueError(
            f"gallery_labels length {len(gallery_labels)} != matrix cols "
            f"{similarity_matrix.shape[1]}"
        )

    # Top-1 predictions
    top1_indices = np.argmax(similarity_matrix, axis=1)
    top1_scores = similarity_matrix[np.arange(num_queries), top1_indices]
    top1_correct = gallery_labels[top1_indices] == query_labels

    # Bin by score
    bin_edges = np.linspace(float(top1_scores.min()), float(top1_scores.max()), num_bins + 1)
    bin_accuracies = np.full(num_bins, np.nan)
    bin_counts = np.zeros(num_bins, dtype=np.int64)

    for i in range(num_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        if i < num_bins - 1:
            mask = (top1_scores >= lower) & (top1_scores < upper)
        else:
            # Last bin is inclusive on both ends
            mask = (top1_scores >= lower) & (top1_scores <= upper)

        count = int(mask.sum())
        bin_counts[i] = count
        if count > 0:
            bin_accuracies[i] = float(top1_correct[mask].mean())

    return CalibrationResult(
        bin_edges=bin_edges,
        bin_accuracies=bin_accuracies,
        bin_counts=bin_counts,
    )


def _compute_map_at_k(matches: NDArray[np.bool_], k: int) -> float:
    """Compute mean average precision at K.

    For our single-label retrieval task (one correct album per query),
    AP@K simplifies to 1/rank if rank <= K, else 0.

    Args:
        matches: Shape (num_queries, num_gallery). Boolean match matrix.
        k: Cutoff for precision computation.

    Returns:
        Mean AP@K across all queries.
    """
    matches_at_k = matches[:, :k]
    # For single-label: first match position gives 1/rank
    ap_scores: list[float] = []
    for row in matches_at_k:
        positions = np.where(row)[0]
        if len(positions) > 0:
            # AP for single relevant item = 1 / (rank), where rank is 1-indexed
            ap_scores.append(1.0 / (positions[0] + 1))
        else:
            ap_scores.append(0.0)

    return float(np.mean(ap_scores))


def _compute_mrr(matches: NDArray[np.bool_]) -> float:
    """Compute mean reciprocal rank over all gallery items (not truncated).

    Args:
        matches: Shape (num_queries, num_gallery). Boolean match matrix.

    Returns:
        Mean reciprocal rank across all queries.
    """
    rr_scores: list[float] = []
    for row in matches:
        positions = np.where(row)[0]
        if len(positions) > 0:
            rr_scores.append(1.0 / (positions[0] + 1))
        else:
            rr_scores.append(0.0)

    return float(np.mean(rr_scores))
