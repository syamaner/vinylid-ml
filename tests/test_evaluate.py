"""Tests for scripts/evaluate.py helper functions."""

# pyright: reportPrivateUsage=false
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import helper functions from the script module
from scripts.evaluate import (
    _append_summary_csv,
    _build_gallery_query_split,
    _compute_similarity_matrix,
    _save_results,
)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-8, None)


def _make_manifest(
    image_paths: list[str],
    album_ids: list[str],
    widths: list[int] | None = None,
    heights: list[int] | None = None,
) -> pd.DataFrame:
    """Build a minimal manifest DataFrame for testing."""
    n = len(image_paths)
    if widths is None:
        widths = [500] * n
    if heights is None:
        heights = [500] * n
    return pd.DataFrame(
        {
            "image_path": image_paths,
            "album_id": album_ids,
            "width": widths,
            "height": heights,
            "artist": ["artist"] * n,
            "album": ["album"] * n,
        }
    )


class TestBuildGalleryQuerySplit:
    """Tests for _build_gallery_query_split()."""

    def test_single_image_albums_gallery_only(self) -> None:
        """Albums with 1 image go to gallery only, no queries."""
        embeddings = _l2_normalize(np.eye(3, dtype=np.float32))
        album_ids = ["a", "b", "c"]
        image_paths = ["img0.jpg", "img1.jpg", "img2.jpg"]
        manifest = _make_manifest(image_paths, album_ids)

        split = _build_gallery_query_split(
            embeddings, album_ids, manifest, image_paths
        )

        assert len(split.gallery_labels) == 3
        assert len(split.query_labels) == 0

    def test_multi_image_album_splits_correctly(self) -> None:
        """Album with 3 images: 1 to gallery, 2 to queries."""
        embeddings = _l2_normalize(
            np.random.default_rng(42).standard_normal((5, 8)).astype(np.float32)
        )
        album_ids = ["a", "a", "a", "b", "c"]
        image_paths = ["img0.jpg", "img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
        widths = [300, 800, 500, 600, 400]
        heights = [300, 800, 500, 600, 400]
        manifest = _make_manifest(image_paths, album_ids, widths, heights)

        split = _build_gallery_query_split(
            embeddings, album_ids, manifest, image_paths
        )

        # 3 albums → 3 gallery items
        assert len(split.gallery_labels) == 3
        # Album "a" had 3 images → 2 queries
        assert len(split.query_labels) == 2

    def test_canonical_is_largest_resolution(self) -> None:
        """Canonical gallery image is the one with largest resolution."""
        embeddings = _l2_normalize(
            np.random.default_rng(42).standard_normal((3, 4)).astype(np.float32)
        )
        album_ids = ["a", "a", "a"]
        image_paths = ["small.jpg", "large.jpg", "medium.jpg"]
        widths = [200, 1000, 500]
        heights = [200, 1000, 500]
        manifest = _make_manifest(image_paths, album_ids, widths, heights)

        split = _build_gallery_query_split(
            embeddings, album_ids, manifest, image_paths
        )

        # Gallery should contain the embedding for "large.jpg" (index 1)
        # The gallery embedding should match the original index-1 embedding
        expected_emb = embeddings[1].astype(np.float32)
        np.testing.assert_allclose(split.gallery_embeddings[0], expected_emb, atol=1e-6)

    def test_label_mapping_is_deterministic(self) -> None:
        """Integer labels are assigned in sorted album_id order."""
        embeddings = _l2_normalize(np.eye(3, dtype=np.float32))
        album_ids = ["c", "a", "b"]
        image_paths = ["img0.jpg", "img1.jpg", "img2.jpg"]
        manifest = _make_manifest(image_paths, album_ids)

        split = _build_gallery_query_split(
            embeddings, album_ids, manifest, image_paths
        )

        assert split.label_to_album_id == {0: "a", 1: "b", 2: "c"}

    def test_album_image_counts(self) -> None:
        """album_image_counts reflects total images per album in the split."""
        embeddings = _l2_normalize(
            np.random.default_rng(42).standard_normal((5, 4)).astype(np.float32)
        )
        album_ids = ["a", "a", "a", "b", "b"]
        image_paths = [f"img{i}.jpg" for i in range(5)]
        manifest = _make_manifest(image_paths, album_ids)

        split = _build_gallery_query_split(
            embeddings, album_ids, manifest, image_paths
        )

        # label 0 = "a" (3 images), label 1 = "b" (2 images)
        assert split.album_image_counts[0] == 3
        assert split.album_image_counts[1] == 2

    def test_query_resolutions_populated(self) -> None:
        """Query resolutions are set from manifest width/height data."""
        embeddings = _l2_normalize(
            np.random.default_rng(42).standard_normal((3, 4)).astype(np.float32)
        )
        album_ids = ["a", "a", "a"]
        image_paths = ["img0.jpg", "img1.jpg", "img2.jpg"]
        widths = [100, 800, 300]
        heights = [200, 600, 400]
        manifest = _make_manifest(image_paths, album_ids, widths, heights)

        split = _build_gallery_query_split(
            embeddings, album_ids, manifest, image_paths
        )

        # img1 (800x600) is canonical → queries are img0 (200) and img2 (400)
        assert len(split.query_resolutions) == 2
        assert all(r > 0 for r in split.query_resolutions)

    def test_embeddings_dtype_is_float32(self) -> None:
        """Gallery and query embeddings are upcast to float32."""
        embeddings = _l2_normalize(np.eye(3, dtype=np.float16))
        album_ids = ["a", "a", "b"]
        image_paths = ["img0.jpg", "img1.jpg", "img2.jpg"]
        manifest = _make_manifest(image_paths, album_ids)

        split = _build_gallery_query_split(
            embeddings, album_ids, manifest, image_paths
        )

        assert split.gallery_embeddings.dtype == np.float32
        assert split.query_embeddings.dtype == np.float32


class TestComputeSimilarityMatrix:
    """Tests for _compute_similarity_matrix()."""

    def test_identity_produces_high_diagonal(self) -> None:
        """Identical query/gallery embeddings produce high self-similarity."""
        embeddings = _l2_normalize(np.eye(3, dtype=np.float32))
        sim = _compute_similarity_matrix(embeddings, embeddings)

        assert sim.shape == (3, 3)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-6)

    def test_orthogonal_produces_zero(self) -> None:
        """Orthogonal embeddings produce zero similarity."""
        queries = _l2_normalize(np.array([[1, 0, 0]], dtype=np.float32))
        gallery = _l2_normalize(np.array([[0, 1, 0]], dtype=np.float32))

        sim = _compute_similarity_matrix(queries, gallery)

        assert sim.shape == (1, 1)
        assert abs(float(sim[0, 0])) < 1e-6

    def test_float16_inputs_upcast(self) -> None:
        """Float16 inputs are computed correctly via float32 upcast."""
        q = _l2_normalize(np.array([[1.0, 0.0]], dtype=np.float16))
        g = _l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16))

        sim = _compute_similarity_matrix(q, g)

        assert sim.dtype == np.float32
        assert sim[0, 0] == pytest.approx(1.0, abs=1e-3)
        assert abs(float(sim[0, 1])) < 1e-3


class TestSaveResults:
    """Tests for _save_results()."""

    def test_creates_all_files(self, tmp_path: Path) -> None:
        """_save_results creates metrics.json, config.json, and all CSVs."""
        run_dir = tmp_path / "run"
        _save_results(
            run_dir,
            metrics_dict={"retrieval": {"recall_at_1": 0.9}},
            config_dict={"model_id": "test"},
            nn_ambiguity_data=[
                {"gallery_idx": 0, "album_label": 0, "nn_distance": 0.5, "album_image_count": 1}
            ],
            calibration_data=[{"bin_lower": 0.0, "bin_upper": 0.5, "accuracy": 0.8, "count": 10}],
            per_query_data=[
                {
                    "query_idx": 0,
                    "query_label": 0,
                    "top1_label": 0,
                    "top1_score": 0.95,
                    "correct": True,
                }
            ],
        )

        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "config.json").exists()
        assert (run_dir / "nn_ambiguity.csv").exists()
        assert (run_dir / "calibration.csv").exists()
        assert (run_dir / "per_query.csv").exists()

    def test_metrics_json_content(self, tmp_path: Path) -> None:
        """metrics.json contains the expected data."""
        run_dir = tmp_path / "run"
        _save_results(
            run_dir,
            metrics_dict={"retrieval": {"recall_at_1": 0.85}},
            config_dict={"model_id": "test"},
            nn_ambiguity_data=[],
            calibration_data=[],
            per_query_data=[],
        )

        with (run_dir / "metrics.json").open() as f:
            data = json.load(f)
        assert data["retrieval"]["recall_at_1"] == 0.85

    def test_empty_csv_data_skips_file(self, tmp_path: Path) -> None:
        """Empty CSV data lists don't create files."""
        run_dir = tmp_path / "run"
        _save_results(
            run_dir,
            metrics_dict={},
            config_dict={},
            nn_ambiguity_data=[],
            calibration_data=[],
            per_query_data=[],
        )

        assert not (run_dir / "nn_ambiguity.csv").exists()
        assert not (run_dir / "calibration.csv").exists()
        assert not (run_dir / "per_query.csv").exists()


class TestAppendSummaryCSV:
    """Tests for _append_summary_csv()."""

    def test_creates_summary_with_header(self, tmp_path: Path) -> None:
        """First call creates summary.csv with header row."""
        _append_summary_csv(
            tmp_path,
            "A1-dinov2-cls",
            "2026-03-14T21-00-00",
            {
                "retrieval": {
                    "recall_at_1": 0.85,
                    "recall_at_5": 0.95,
                    "map_at_5": 0.88,
                    "mrr": 0.90,
                }
            },
            500,
            200,
        )

        summary_path = tmp_path / "summary.csv"
        assert summary_path.exists()
        with summary_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["model_id"] == "A1-dinov2-cls"
        assert rows[0]["recall_at_1"] == "0.85"

    def test_appends_without_duplicate_header(self, tmp_path: Path) -> None:
        """Subsequent calls append rows without repeating the header."""
        for i in range(3):
            _append_summary_csv(
                tmp_path,
                f"model-{i}",
                f"ts-{i}",
                {"retrieval": {"recall_at_1": 0.8 + i * 0.05}},
                500,
                200,
            )

        with (tmp_path / "summary.csv").open() as f:
            lines = f.readlines()
        # 1 header + 3 data rows
        assert len(lines) == 4
