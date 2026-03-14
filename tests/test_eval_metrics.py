"""Tests for vinylid_ml.eval_metrics."""

from __future__ import annotations

import numpy as np
import pytest

from vinylid_ml.eval_metrics import (
    RetrievalMetrics,
    compute_retrieval_metrics,
    compute_stratified_metrics,
)


class TestComputeRetrievalMetrics:
    """Tests for compute_retrieval_metrics()."""

    def test_perfect_retrieval(self) -> None:
        """All queries match top-1 → Recall@1=1.0, MRR=1.0."""
        # 3 queries, 3 gallery items — identity similarity matrix
        similarity = np.array([
            [1.0, 0.1, 0.2],
            [0.2, 1.0, 0.1],
            [0.1, 0.2, 1.0],
        ])
        query_labels = np.array([0, 1, 2])
        gallery_labels = np.array([0, 1, 2])

        metrics = compute_retrieval_metrics(similarity, query_labels, gallery_labels)

        assert metrics.recall_at_1 == pytest.approx(1.0)
        assert metrics.recall_at_5 == pytest.approx(1.0)
        assert metrics.map_at_5 == pytest.approx(1.0)
        assert metrics.mrr == pytest.approx(1.0)
        assert metrics.num_queries == 3

    def test_top1_all_wrong(self) -> None:
        """No query finds correct album at rank 1, but correct exists at rank 2."""
        similarity = np.array([
            [0.1, 0.9],
            [0.9, 0.1],
        ])
        # Query 0 (label 0) picks gallery[1] (label 1) — wrong at top-1
        # Query 1 (label 1) picks gallery[0] (label 0) — wrong at top-1
        query_labels = np.array([0, 1])
        gallery_labels = np.array([0, 1])

        metrics = compute_retrieval_metrics(similarity, query_labels, gallery_labels)

        assert metrics.recall_at_1 == pytest.approx(0.0)
        # Correct label at rank 2 → MRR = (1/2 + 1/2) / 2
        assert metrics.mrr == pytest.approx(0.5)
        # Both correct within top-5
        assert metrics.recall_at_5 == pytest.approx(1.0)

    def test_label_absent_from_gallery(self) -> None:
        """Query label not in gallery at all → all metrics 0."""
        similarity = np.array([[0.5, 0.9]])
        query_labels = np.array([99])  # not in gallery
        gallery_labels = np.array([0, 1])

        metrics = compute_retrieval_metrics(similarity, query_labels, gallery_labels)

        assert metrics.recall_at_1 == pytest.approx(0.0)
        assert metrics.recall_at_5 == pytest.approx(0.0)
        assert metrics.mrr == pytest.approx(0.0)

    def test_partial_recall(self) -> None:
        """One of two queries matches top-1 → Recall@1=0.5."""
        similarity = np.array([
            [0.9, 0.1, 0.05],  # top-1 is gallery[0]=label 0, correct
            [0.1, 0.3, 0.8],   # top-1 is gallery[2]=label 2, but query is label 1
        ])
        query_labels = np.array([0, 1])
        gallery_labels = np.array([0, 1, 2])

        metrics = compute_retrieval_metrics(similarity, query_labels, gallery_labels)

        assert metrics.recall_at_1 == pytest.approx(0.5)
        # Query 1: correct label 1 is at rank 2 → in top-5
        assert metrics.recall_at_5 == pytest.approx(1.0)

    def test_mrr_calculation(self) -> None:
        """MRR with known rank positions."""
        # Query 0: correct at rank 1, Query 1: correct at rank 3
        similarity = np.array([
            [0.9, 0.5, 0.1],  # rank: 0=0.9, 1=0.5, 2=0.1 → label 0 at rank 1
            [0.1, 0.3, 0.8],  # rank: 2=0.8, 1=0.3, 0=0.1 → label 1 at rank 2
        ])
        query_labels = np.array([0, 1])
        gallery_labels = np.array([0, 1, 2])

        metrics = compute_retrieval_metrics(similarity, query_labels, gallery_labels)

        # MRR = (1/1 + 1/2) / 2 = 0.75
        assert metrics.mrr == pytest.approx(0.75)

    def test_dimension_mismatch_raises(self) -> None:
        """Mismatched dimensions should raise ValueError."""
        similarity = np.array([[0.5, 0.5]])
        query_labels = np.array([0, 1])  # 2 labels but 1 row
        gallery_labels = np.array([0, 1])

        with pytest.raises(ValueError, match="query_labels length"):
            compute_retrieval_metrics(similarity, query_labels, gallery_labels)

    def test_to_dict(self) -> None:
        """RetrievalMetrics.to_dict() produces JSON-serializable output."""
        metrics = RetrievalMetrics(
            recall_at_1=0.9, recall_at_5=0.95, map_at_5=0.92, mrr=0.91, num_queries=100
        )
        d = metrics.to_dict()
        assert d["recall_at_1"] == 0.9
        assert d["num_queries"] == 100
        assert isinstance(d, dict)


class TestComputeStratifiedMetrics:
    """Tests for compute_stratified_metrics()."""

    def test_by_album_image_count(self) -> None:
        """Stratification by 1 / 2-5 / 6+ image buckets."""
        # 4 queries, 4 gallery. Query 0,1 correct; 2,3 wrong.
        similarity = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.9],  # wrong: picks label 3 instead of 2
            [0.0, 0.0, 0.9, 0.1],  # wrong: picks label 2 instead of 3
        ])
        query_labels = np.array([0, 1, 2, 3])
        gallery_labels = np.array([0, 1, 2, 3])

        # Labels 0,2 have 1 image; labels 1,3 have 3 images
        album_counts = {0: 1, 1: 3, 2: 1, 3: 3}

        strat = compute_stratified_metrics(
            similarity, query_labels, gallery_labels,
            album_image_counts=album_counts,
        )

        # Single-image: label 0 correct, label 2 wrong → 50%
        assert strat.by_album_image_count["1"] == pytest.approx(0.5)
        # Multi-image (2-5): label 1 correct, label 3 wrong → 50%
        assert strat.by_album_image_count["2-5"] == pytest.approx(0.5)

    def test_by_resolution_bucket(self) -> None:
        """Stratification by resolution buckets."""
        similarity = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        query_labels = np.array([0, 1])
        gallery_labels = np.array([0, 1])
        resolutions = np.array([300, 900])  # <400, 800+

        strat = compute_stratified_metrics(
            similarity, query_labels, gallery_labels,
            query_resolutions=resolutions,
        )

        assert strat.by_resolution_bucket["<400px"] == pytest.approx(1.0)
        assert strat.by_resolution_bucket["800+px"] == pytest.approx(1.0)

    def test_by_augmentation_type(self) -> None:
        """Stratification by augmentation type."""
        similarity = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.6],  # wrong
        ])
        query_labels = np.array([0, 1, 0])
        gallery_labels = np.array([0, 1])
        aug_types = ["blur", "crop", "blur"]

        strat = compute_stratified_metrics(
            similarity, query_labels, gallery_labels,
            query_augmentation_types=aug_types,
        )

        # blur: query 0 correct, query 2 wrong → 50%
        assert strat.by_augmentation_type["blur"] == pytest.approx(0.5)
        # crop: query 1 correct → 100%
        assert strat.by_augmentation_type["crop"] == pytest.approx(1.0)

    def test_empty_buckets_omitted(self) -> None:
        """Buckets with no queries are not included in results."""
        similarity = np.array([[1.0]])
        query_labels = np.array([0])
        gallery_labels = np.array([0])
        album_counts = {0: 1}

        strat = compute_stratified_metrics(
            similarity, query_labels, gallery_labels,
            album_image_counts=album_counts,
        )

        assert "1" in strat.by_album_image_count
        assert "2-5" not in strat.by_album_image_count
        assert "6+" not in strat.by_album_image_count
