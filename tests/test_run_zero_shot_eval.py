"""Tests for scripts/run_zero_shot_eval.py helper functions."""

# pyright: reportPrivateUsage=false
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import transforms

from scripts.run_zero_shot_eval import (
    _decision_verdict,
    _find_latest_metrics,
    _save_latency_csv,
    measure_single_image_latency,
)
from vinylid_ml.models import EmbeddingModel

# ── Minimal mock model (no downloads) ─────────────────────────────────────────


class _MockEmbeddingModel(EmbeddingModel):
    """Deterministic mock model for latency testing."""

    @property
    def model_id(self) -> str:
        return "mock"

    @property
    def embedding_dim(self) -> int:
        return 4

    @property
    def input_size(self) -> int:
        return 8

    def get_transforms(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(8),
                transforms.CenterCrop(8),
                transforms.ToTensor(),
            ]
        )

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Return fixed zero embedding regardless of input."""
        return torch.zeros(images.shape[0], self.embedding_dim)


@pytest.fixture
def tmp_image(tmp_path: Path) -> Path:
    """A minimal 16x16 red JPEG for use as the latency sample image."""
    img_path = tmp_path / "sample.jpg"
    Image.new("RGB", (16, 16), color=(255, 0, 0)).save(img_path)
    return img_path


# ── measure_single_image_latency ───────────────────────────────────────────────


class TestMeasureSingleImageLatency:
    """Tests for measure_single_image_latency()."""

    def test_returns_three_percentile_keys(self, tmp_image: Path) -> None:
        """Return dict must contain exactly p50_ms, p95_ms, p99_ms."""
        model = _MockEmbeddingModel()
        result = measure_single_image_latency(model, tmp_image, n_warmup=2, n_timed=10)
        assert set(result.keys()) == {"p50_ms", "p95_ms", "p99_ms"}

    def test_percentiles_are_positive(self, tmp_image: Path) -> None:
        """All latency values must be positive."""
        model = _MockEmbeddingModel()
        result = measure_single_image_latency(model, tmp_image, n_warmup=2, n_timed=10)
        assert result["p50_ms"] > 0
        assert result["p95_ms"] > 0
        assert result["p99_ms"] > 0

    def test_p50_le_p95_le_p99(self, tmp_image: Path) -> None:
        """Percentile ordering must be preserved: p50 ≤ p95 ≤ p99."""
        model = _MockEmbeddingModel()
        result = measure_single_image_latency(model, tmp_image, n_warmup=2, n_timed=20)
        assert result["p50_ms"] <= result["p95_ms"]
        assert result["p95_ms"] <= result["p99_ms"]

    def test_single_timed_run_returns_same_value(self, tmp_image: Path) -> None:
        """With n_timed=1, all three percentiles are the same value."""
        model = _MockEmbeddingModel()
        result = measure_single_image_latency(model, tmp_image, n_warmup=0, n_timed=1)
        assert result["p50_ms"] == result["p95_ms"] == result["p99_ms"]


# ── _decision_verdict ──────────────────────────────────────────────────────────


class TestDecisionVerdict:
    """Tests for _decision_verdict()."""

    @pytest.mark.parametrize(
        "recall,expected",
        [
            (1.0, "zero-shot sufficient"),
            (0.90, "zero-shot sufficient"),
            (0.95, "zero-shot sufficient"),
            (0.899, "fine-tuning likely needed"),
            (0.80, "fine-tuning likely needed"),
            (0.85, "fine-tuning likely needed"),
            (0.799, "fine-tuning required"),
            (0.50, "fine-tuning required"),
            (0.0, "fine-tuning required"),
        ],
    )
    def test_thresholds(self, recall: float, expected: str) -> None:
        assert _decision_verdict(recall) == expected


# ── _save_latency_csv ──────────────────────────────────────────────────────────


class TestSaveLatencyCsv:
    """Tests for _save_latency_csv()."""

    def test_creates_csv_with_correct_columns(self, tmp_path: Path) -> None:
        rows = [
            {"model_id": "A1-dinov2-cls", "p50_ms": 12.3, "p95_ms": 15.0, "p99_ms": 18.0},
            {"model_id": "A4-sscd", "p50_ms": 8.0, "p95_ms": 10.0, "p99_ms": 11.0},
        ]
        out = _save_latency_csv(tmp_path / "results", rows)
        assert out.exists()
        with out.open(newline="") as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames or []) == {"model_id", "p50_ms", "p95_ms", "p99_ms"}
            data = list(reader)
        assert len(data) == 2
        assert data[0]["model_id"] == "A1-dinov2-cls"

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Output directory is created if it does not exist."""
        nested = tmp_path / "a" / "b" / "c"
        _save_latency_csv(nested, [])
        assert nested.is_dir()


# ── _find_latest_metrics ───────────────────────────────────────────────────────


class TestFindLatestMetrics:
    """Tests for _find_latest_metrics()."""

    def test_returns_none_for_missing_dir(self, tmp_path: Path) -> None:
        assert _find_latest_metrics(tmp_path, "nonexistent-model") is None

    def test_returns_none_for_empty_model_dir(self, tmp_path: Path) -> None:
        (tmp_path / "my-model").mkdir()
        assert _find_latest_metrics(tmp_path, "my-model") is None

    def test_loads_metrics_from_latest_run(self, tmp_path: Path) -> None:
        """Returns metrics from the lexicographically latest timestamp dir."""
        model_dir = tmp_path / "A1-dinov2-cls"
        # Older run
        old_run = model_dir / "2026-01-01T00-00-00"
        old_run.mkdir(parents=True)
        (old_run / "metrics.json").write_text(json.dumps({"recall": 0.80}))
        # Newer run
        new_run = model_dir / "2026-03-01T12-00-00"
        new_run.mkdir(parents=True)
        (new_run / "metrics.json").write_text(json.dumps({"recall": 0.92}))

        result = _find_latest_metrics(tmp_path, "A1-dinov2-cls")
        assert result is not None
        assert result["recall"] == pytest.approx(0.92)  # type: ignore[reportUnknownMemberType]

    def test_skips_run_dirs_without_metrics_json(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "A2-openclip"
        (model_dir / "2026-03-14T10-00-00").mkdir(parents=True)  # no metrics.json
        valid_run = model_dir / "2026-03-14T09-00-00"
        valid_run.mkdir(parents=True)
        (valid_run / "metrics.json").write_text(json.dumps({"recall": 0.75}))

        result = _find_latest_metrics(tmp_path, "A2-openclip")
        assert result is not None
        assert result["recall"] == pytest.approx(0.75)  # type: ignore[reportUnknownMemberType]
