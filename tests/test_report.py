"""Tests for vinylid_ml.report — HTML evaluation report generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vinylid_ml.eval_metrics import (
    CalibrationResult,
    NNAmbiguityResult,
    RetrievalMetrics,
    StratifiedMetrics,
)
from vinylid_ml.report import generate_report


def _make_metrics() -> RetrievalMetrics:
    return RetrievalMetrics(
        recall_at_1=0.85,
        recall_at_5=0.95,
        map_at_5=0.88,
        mrr=0.90,
        num_queries=100,
    )


def _make_stratified() -> StratifiedMetrics:
    return StratifiedMetrics(
        by_album_image_count={"1": 0.80, "2-5": 0.90, "6+": 0.95},
        by_resolution_bucket={"<400px": 0.75, "400-800px": 0.88, "800+px": 0.92},
        by_augmentation_type={"blur": 0.82, "crop": 0.78, "noise": 0.91},
    )


def _make_nn_ambiguity() -> NNAmbiguityResult:
    rng = np.random.default_rng(42)
    return NNAmbiguityResult(
        nn_similarities=rng.random(50).astype(np.float32),
        labels=np.repeat(np.arange(10), 5),
        album_image_counts={i: (i % 3) + 1 for i in range(10)},
    )


def _make_calibration() -> CalibrationResult:
    return CalibrationResult(
        bin_edges=np.linspace(0.0, 1.0, 6),
        bin_accuracies=np.array([0.2, 0.5, 0.7, 0.85, 0.95]),
        bin_counts=np.array([5, 15, 30, 25, 25]),
    )


class TestGenerateReport:
    """Tests for generate_report()."""

    def test_creates_report_html(self, tmp_path: Path) -> None:
        """generate_report() creates a report.html file."""
        report_path = generate_report(
            tmp_path,
            _make_metrics(),
            _make_stratified(),
            _make_nn_ambiguity(),
            _make_calibration(),
            model_id="A1-dinov2-cls",
            split="test",
            timestamp="2026-03-14T21:00:00Z",
            num_gallery=500,
        )
        assert report_path.exists()
        assert report_path.name == "report.html"

    def test_report_contains_model_id(self, tmp_path: Path) -> None:
        """Report HTML contains the model ID."""
        generate_report(
            tmp_path,
            _make_metrics(),
            _make_stratified(),
            None,
            None,
            model_id="A2-openclip",
            split="val",
            timestamp="2026-03-14T21:00:00Z",
            num_gallery=300,
        )
        html = (tmp_path / "report.html").read_text()
        assert "A2-openclip" in html

    def test_report_contains_metrics(self, tmp_path: Path) -> None:
        """Report HTML contains all retrieval metric values."""
        generate_report(
            tmp_path,
            _make_metrics(),
            StratifiedMetrics(),
            None,
            None,
            model_id="test-model",
            split="test",
            timestamp="2026-03-14T21:00:00Z",
            num_gallery=100,
        )
        html = (tmp_path / "report.html").read_text()
        assert "85.0%" in html  # recall_at_1
        assert "95.0%" in html  # recall_at_5
        assert "0.880" in html  # map_at_5
        assert "0.900" in html  # mrr

    def test_report_contains_git_sha(self, tmp_path: Path) -> None:
        """Report includes truncated git SHA when provided."""
        generate_report(
            tmp_path,
            _make_metrics(),
            StratifiedMetrics(),
            None,
            None,
            model_id="test",
            split="test",
            timestamp="2026-03-14T21:00:00Z",
            num_gallery=50,
            git_sha="abcdef1234567890",
        )
        html = (tmp_path / "report.html").read_text()
        assert "abcdef12" in html

    def test_report_without_optional_sections(self, tmp_path: Path) -> None:
        """Report is valid without NN ambiguity and calibration data."""
        report_path = generate_report(
            tmp_path,
            _make_metrics(),
            StratifiedMetrics(),
            None,
            None,
            model_id="test",
            split="test",
            timestamp="2026-03-14T21:00:00Z",
            num_gallery=50,
        )
        html = report_path.read_text()
        assert "<!DOCTYPE html>" in html
        assert "Retrieval Metrics" in html
        # NN ambiguity and calibration sections should not appear
        assert "Nearest-Neighbor Ambiguity" not in html
        assert "Confidence Calibration" not in html

    def test_report_with_all_sections(self, tmp_path: Path) -> None:
        """Report includes all sections when all data is provided."""
        report_path = generate_report(
            tmp_path,
            _make_metrics(),
            _make_stratified(),
            _make_nn_ambiguity(),
            _make_calibration(),
            model_id="test",
            split="test",
            timestamp="2026-03-14T21:00:00Z",
            num_gallery=500,
        )
        html = report_path.read_text()
        assert "Retrieval Metrics" in html
        assert "Stratified Metrics" in html
        assert "Nearest-Neighbor Ambiguity" in html
        assert "Confidence Calibration" in html

    def test_report_contains_plotly(self, tmp_path: Path) -> None:
        """Report includes Plotly JS via CDN when charts are present."""
        generate_report(
            tmp_path,
            _make_metrics(),
            _make_stratified(),
            _make_nn_ambiguity(),
            _make_calibration(),
            model_id="test",
            split="test",
            timestamp="2026-03-14T21:00:00Z",
            num_gallery=500,
        )
        html = (tmp_path / "report.html").read_text()
        assert "plotly" in html.lower()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """generate_report() creates parent directories if they don't exist."""
        nested_dir = tmp_path / "results" / "model" / "timestamp"
        report_path = generate_report(
            nested_dir,
            _make_metrics(),
            StratifiedMetrics(),
            None,
            None,
            model_id="test",
            split="test",
            timestamp="2026-03-14T21:00:00Z",
            num_gallery=50,
        )
        assert report_path.exists()
