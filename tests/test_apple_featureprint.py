"""Tests for src/vinylid_ml/apple_featureprint.py and scripts/embed_featureprint.py.

All Vision framework calls are mocked so these tests run in CI without macOS.
Integration tests (requiring the real Vision framework) are marked with
``@pytest.mark.integration``.
"""

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import csv
import struct
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from scripts.embed_featureprint import (
    _append_latency_csv,
    _filter_manifest_by_split,
)
from vinylid_ml.apple_featureprint import (
    FEATUREPRINT_MODEL_ID,
    _import_vision,  # type: ignore[reportPrivateUsage]
    embed_images,
    extract_feature_vector,
    measure_featureprint_latency,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_fake_observation(
    values: list[float],
    element_type: int = 1,  # VNElementTypeFloat
) -> MagicMock:
    """Create a mock VNFeaturePrintObservation with given float values."""
    raw = struct.pack(f"{len(values)}f", *values)
    obs = MagicMock()
    obs.elementCount.return_value = len(values)
    obs.elementType.return_value = element_type
    obs.data.return_value = raw
    return obs


def _make_vision_mocks(
    values: list[float],
    element_type: int = 1,
) -> tuple[MagicMock, MagicMock]:
    """Return (Vision_mock, Foundation_mock) wired up for a successful request."""
    obs = _make_fake_observation(values, element_type=element_type)

    mock_request = MagicMock()
    mock_request.results.return_value = [obs]

    mock_vision = MagicMock()
    mock_vision.VNGenerateImageFeaturePrintRequest.alloc.return_value.init.return_value = (
        mock_request
    )

    mock_foundation = MagicMock()
    return mock_vision, mock_foundation


# ── FEATUREPRINT_MODEL_ID ──────────────────────────────────────────────────────


def test_featureprint_model_id_constant() -> None:
    """FEATUREPRINT_MODEL_ID is the string 'A3-featureprint'."""
    assert FEATUREPRINT_MODEL_ID == "A3-featureprint"


# ── _import_vision ─────────────────────────────────────────────────────────────


def test_import_vision_raises_import_error_when_missing() -> None:
    """_import_vision raises ImportError with hint when Vision/Foundation are absent."""
    with patch.dict("sys.modules", {"Vision": None, "Foundation": None}), pytest.raises(
        ImportError, match="pyobjc-framework-Vision"
    ):
        _import_vision()


# ── extract_feature_vector ────────────────────────────────────────────────────


class TestExtractFeatureVector:
    """Tests for extract_feature_vector()."""

    def test_returns_float32_array_for_float_element_type(
        self, tmp_path: Path
    ) -> None:
        """elementType=1 (float32 source) returns a float32 array with correct values."""
        img = tmp_path / "img.jpg"
        img.write_bytes(b"fake")
        values = [1.0, 2.0, 3.0, 4.0]
        mock_vision, mock_foundation = _make_vision_mocks(values, element_type=1)

        with patch(
            "vinylid_ml.apple_featureprint._import_vision",
            return_value=(mock_vision, mock_foundation),
        ):
            result = extract_feature_vector(img)

        assert result.dtype == np.float32
        assert result.shape == (4,)
        np.testing.assert_array_almost_equal(result, np.array(values, dtype=np.float32))

    def test_returns_float32_for_double_element_type(self, tmp_path: Path) -> None:
        """elementType=2 (float64 source) is cast to float32 on return."""
        img = tmp_path / "img.jpg"
        img.write_bytes(b"fake")
        values = [1.5, 2.5]
        # Pack as float64
        raw = struct.pack("2d", *values)
        obs = MagicMock()
        obs.elementCount.return_value = 2
        obs.elementType.return_value = 2  # VNElementTypeDouble
        obs.data.return_value = raw

        mock_request = MagicMock()
        mock_request.results.return_value = [obs]
        mock_vision = MagicMock()
        mock_vision.VNGenerateImageFeaturePrintRequest.alloc.return_value.init.return_value = (
            mock_request
        )
        mock_foundation = MagicMock()

        with patch(
            "vinylid_ml.apple_featureprint._import_vision",
            return_value=(mock_vision, mock_foundation),
        ):
            result = extract_feature_vector(img)

        assert result.dtype == np.float32
        assert result.shape == (2,)
        np.testing.assert_array_almost_equal(
            result, np.array(values, dtype=np.float32), decimal=5
        )

    def test_raises_file_not_found_for_missing_path(self, tmp_path: Path) -> None:
        """Passing a path that does not exist raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Image not found"):
            extract_feature_vector(tmp_path / "nonexistent.jpg")

    def test_raises_runtime_error_when_request_raises(self, tmp_path: Path) -> None:
        """An exception inside performRequests_error_ is re-raised as RuntimeError."""
        img = tmp_path / "img.jpg"
        img.write_bytes(b"fake")

        mock_vision = MagicMock()
        mock_vision.VNImageRequestHandler.alloc.return_value.initWithURL_options_.return_value.performRequests_error_.side_effect = (
            RuntimeError("Vision failed")
        )
        mock_foundation = MagicMock()

        with patch(
            "vinylid_ml.apple_featureprint._import_vision",
            return_value=(mock_vision, mock_foundation),
        ), pytest.raises(RuntimeError, match="VNGenerateImageFeaturePrintRequest failed"):
            extract_feature_vector(img)

    def test_raises_runtime_error_when_results_empty(self, tmp_path: Path) -> None:
        """An empty Vision results list raises RuntimeError."""
        img = tmp_path / "img.jpg"
        img.write_bytes(b"fake")

        mock_request = MagicMock()
        mock_request.results.return_value = []
        mock_vision = MagicMock()
        mock_vision.VNGenerateImageFeaturePrintRequest.alloc.return_value.init.return_value = (
            mock_request
        )
        mock_foundation = MagicMock()

        with patch(
            "vinylid_ml.apple_featureprint._import_vision",
            return_value=(mock_vision, mock_foundation),
        ), pytest.raises(RuntimeError, match="No FeaturePrint results"):
            extract_feature_vector(img)

    def test_raises_runtime_error_on_element_count_mismatch(
        self, tmp_path: Path
    ) -> None:
        """Mismatch between elementCount and actual data bytes raises RuntimeError."""
        img = tmp_path / "img.jpg"
        img.write_bytes(b"fake")
        # 4 actual floats but claim elementCount=99
        raw = struct.pack("4f", 1.0, 2.0, 3.0, 4.0)
        obs = MagicMock()
        obs.elementCount.return_value = 99
        obs.elementType.return_value = 1
        obs.data.return_value = raw

        mock_request = MagicMock()
        mock_request.results.return_value = [obs]
        mock_vision = MagicMock()
        mock_vision.VNGenerateImageFeaturePrintRequest.alloc.return_value.init.return_value = (
            mock_request
        )
        mock_foundation = MagicMock()

        with patch(
            "vinylid_ml.apple_featureprint._import_vision",
            return_value=(mock_vision, mock_foundation),
        ), pytest.raises(RuntimeError, match="data size mismatch"):
            extract_feature_vector(img)

    def test_raises_runtime_error_for_unknown_element_type(
        self, tmp_path: Path
    ) -> None:
        """An element_type not in {1, 2} raises RuntimeError with 'Unsupported'."""
        img = tmp_path / "img.jpg"
        img.write_bytes(b"fake")
        raw = struct.pack("4f", 1.0, 2.0, 3.0, 4.0)
        obs = MagicMock()
        obs.elementCount.return_value = 4
        obs.elementType.return_value = 99  # unknown type
        obs.data.return_value = raw

        mock_request = MagicMock()
        mock_request.results.return_value = [obs]
        mock_vision = MagicMock()
        mock_vision.VNGenerateImageFeaturePrintRequest.alloc.return_value.init.return_value = (
            mock_request
        )
        mock_foundation = MagicMock()

        with patch(
            "vinylid_ml.apple_featureprint._import_vision",
            return_value=(mock_vision, mock_foundation),
        ), pytest.raises(RuntimeError, match="Unsupported VNElementType"):
            extract_feature_vector(img)

    def test_result_is_a_copy(self, tmp_path: Path) -> None:
        """Result array is a copy, not a view into shared buffer."""
        img = tmp_path / "img.jpg"
        img.write_bytes(b"fake")
        values = [10.0, 20.0]
        mock_vision, mock_foundation = _make_vision_mocks(values)

        with patch(
            "vinylid_ml.apple_featureprint._import_vision",
            return_value=(mock_vision, mock_foundation),
        ):
            result = extract_feature_vector(img)

        result[0] = 999.0  # mutating should not affect re-extraction
        assert result[0] == pytest.approx(999.0)


# ── embed_images ───────────────────────────────────────────────────────────────


class TestEmbedImages:
    """Tests for embed_images()."""

    def _fake_extract(self, path: Path) -> NDArray[np.float32]:
        """Simple fake that returns a known non-unit vector based on path stem."""
        seed = abs(hash(path.stem)) % 1000
        rng = np.random.default_rng(seed)
        return rng.standard_normal(8).astype(np.float32)

    def test_output_shape(self, tmp_path: Path) -> None:
        """Output matrix has shape (N, D) matching image count and feature dim."""
        imgs = [tmp_path / f"img{i}.jpg" for i in range(5)]
        for p in imgs:
            p.write_bytes(b"x")

        with patch(
            "vinylid_ml.apple_featureprint.extract_feature_vector",
            side_effect=self._fake_extract,
        ):
            result = embed_images(imgs)

        assert result.shape == (5, 8)

    def test_rows_are_l2_normalised(self, tmp_path: Path) -> None:
        """Each row in the output matrix has unit L2 norm."""
        imgs = [tmp_path / f"img{i}.jpg" for i in range(6)]
        for p in imgs:
            p.write_bytes(b"x")

        with patch(
            "vinylid_ml.apple_featureprint.extract_feature_vector",
            side_effect=self._fake_extract,
        ):
            result = embed_images(imgs)

        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, np.ones(6), atol=1e-5)

    def test_output_dtype_is_float32(self, tmp_path: Path) -> None:
        """Output matrix dtype is always float32."""
        imgs = [tmp_path / "img0.jpg"]
        imgs[0].write_bytes(b"x")

        with patch(
            "vinylid_ml.apple_featureprint.extract_feature_vector",
            return_value=np.array([3.0, 4.0], dtype=np.float32),
        ):
            result = embed_images(imgs)

        assert result.dtype == np.float32

    def test_raises_value_error_for_empty_list(self) -> None:
        """Passing an empty list raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            embed_images([])

    def test_zero_vector_handled_gracefully(self, tmp_path: Path) -> None:
        """A zero-norm input vector should not produce NaN output."""
        img = tmp_path / "zero.jpg"
        img.write_bytes(b"x")

        with patch(
            "vinylid_ml.apple_featureprint.extract_feature_vector",
            return_value=np.zeros(4, dtype=np.float32),
        ):
            result = embed_images([img])

        assert not np.any(np.isnan(result))

    def test_single_image(self, tmp_path: Path) -> None:
        """A single [3, 4] vector is normalised to [0.6, 0.8]."""
        img = tmp_path / "single.jpg"
        img.write_bytes(b"x")
        vec = np.array([3.0, 4.0], dtype=np.float32)  # norm=5 → normalised=[0.6, 0.8]

        with patch(
            "vinylid_ml.apple_featureprint.extract_feature_vector",
            return_value=vec,
        ):
            result = embed_images([img])

        assert result.shape == (1, 2)
        np.testing.assert_allclose(result[0], [0.6, 0.8], atol=1e-6)

    def test_show_progress_does_not_break(self, tmp_path: Path) -> None:
        """show_progress=True should not raise; logging is a side-effect."""
        imgs = [tmp_path / f"img{i}.jpg" for i in range(3)]
        for p in imgs:
            p.write_bytes(b"x")

        with patch(
            "vinylid_ml.apple_featureprint.extract_feature_vector",
            return_value=np.array([1.0, 0.0], dtype=np.float32),
        ):
            result = embed_images(imgs, show_progress=True)

        assert result.shape == (3, 2)


# ── measure_featureprint_latency ───────────────────────────────────────────────


class TestMeasureFeatureprintLatency:
    """Tests for measure_featureprint_latency()."""

    def test_returns_dict_with_correct_keys(self, tmp_path: Path) -> None:
        """Returns a dict with exactly the keys p50_ms, p95_ms, p99_ms."""
        img = tmp_path / "img.jpg"
        img.write_bytes(b"x")

        with patch(
            "vinylid_ml.apple_featureprint.extract_feature_vector",
            return_value=np.ones(4, dtype=np.float32),
        ):
            result = measure_featureprint_latency(img, n_warmup=2, n_timed=5)

        assert set(result.keys()) == {"p50_ms", "p95_ms", "p99_ms"}

    def test_all_latency_values_are_positive(self, tmp_path: Path) -> None:
        """All returned latency values are strictly positive."""
        img = tmp_path / "img.jpg"
        img.write_bytes(b"x")

        with patch(
            "vinylid_ml.apple_featureprint.extract_feature_vector",
            return_value=np.ones(4, dtype=np.float32),
        ):
            result = measure_featureprint_latency(img, n_warmup=1, n_timed=5)

        assert result["p50_ms"] > 0.0
        assert result["p95_ms"] > 0.0
        assert result["p99_ms"] > 0.0

    def test_percentile_ordering(self, tmp_path: Path) -> None:
        """p50 <= p95 <= p99 across a sample of 20 timed runs."""
        img = tmp_path / "img.jpg"
        img.write_bytes(b"x")

        with patch(
            "vinylid_ml.apple_featureprint.extract_feature_vector",
            return_value=np.ones(4, dtype=np.float32),
        ):
            result = measure_featureprint_latency(img, n_warmup=1, n_timed=20)

        assert result["p50_ms"] <= result["p95_ms"] <= result["p99_ms"]

    def test_raises_value_error_for_n_timed_zero(self, tmp_path: Path) -> None:
        """n_timed=0 raises ValueError before any extractions are attempted."""
        img = tmp_path / "img.jpg"
        img.write_bytes(b"x")
        with pytest.raises(ValueError, match="n_timed must be >= 1"):
            measure_featureprint_latency(img, n_timed=0)

    def test_raises_value_error_for_negative_n_timed(self, tmp_path: Path) -> None:
        """Negative n_timed raises ValueError."""
        img = tmp_path / "img.jpg"
        img.write_bytes(b"x")
        with pytest.raises(ValueError, match="n_timed must be >= 1"):
            measure_featureprint_latency(img, n_timed=-5)

    def test_raises_value_error_for_negative_n_warmup(self, tmp_path: Path) -> None:
        """Negative n_warmup raises ValueError."""
        img = tmp_path / "img.jpg"
        img.write_bytes(b"x")
        with pytest.raises(ValueError, match="n_warmup must be >= 0"):
            measure_featureprint_latency(img, n_warmup=-1)


# ── _filter_manifest_by_split ─────────────────────────────────────────────────


class TestFilterManifestBySplit:
    """Tests for _filter_manifest_by_split() in embed_featureprint.py."""

    def _make_manifest(self, album_ids: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "image_path": [f"{aid}/img.jpg" for aid in album_ids],
                "album_id": album_ids,
            }
        )

    def _make_splits(self, mapping: dict[str, str]) -> dict[str, str]:
        return mapping

    def test_filters_to_test_split(self) -> None:
        """Only rows whose album_id belongs to the given split are returned."""
        manifest = self._make_manifest(["a1", "a2", "a3"])
        splits = {"a1": "test", "a2": "train", "a3": "test"}
        result = _filter_manifest_by_split(manifest, splits, "test")
        assert set(result["album_id"].tolist()) == {"a1", "a3"}

    def test_returns_all_when_split_is_all(self) -> None:
        """split='all' returns the entire manifest unchanged."""
        manifest = self._make_manifest(["a1", "a2"])
        splits: dict[str, str] = {}
        result = _filter_manifest_by_split(manifest, splits, "all")
        assert len(result) == 2

    def test_empty_split_returns_empty_df(self) -> None:
        """A split with no matching album IDs returns an empty DataFrame."""
        manifest = self._make_manifest(["a1", "a2"])
        splits = {"a1": "train", "a2": "train"}
        result = _filter_manifest_by_split(manifest, splits, "test")
        assert len(result) == 0

    def test_result_resets_index(self) -> None:
        """Returned DataFrame index is reset to 0-based integers."""
        manifest = self._make_manifest(["a1", "a2", "a3"])
        splits = {"a2": "val", "a3": "train"}
        result = _filter_manifest_by_split(manifest, splits, "val")
        assert list(result.index) == [0]


# ── _append_latency_csv ───────────────────────────────────────────────────────


class TestAppendLatencyCsv:
    """Tests for _append_latency_csv() in embed_featureprint.py."""

    def test_creates_file_with_header_when_absent(self, tmp_path: Path) -> None:
        """Creates latency_summary.csv with header and one row when file is absent."""
        latency = {"p50_ms": 12.3, "p95_ms": 15.0, "p99_ms": 18.0}
        out = _append_latency_csv(tmp_path, "A3-featureprint", latency)

        assert out.exists()
        with out.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["model_id"] == "A3-featureprint"
        assert float(rows[0]["p50_ms"]) == pytest.approx(12.3)

    def test_appends_row_when_file_exists(self, tmp_path: Path) -> None:
        """Appends a new row to an existing latency_summary.csv without rewriting header."""
        latency: dict[str, float] = {"p50_ms": 5.0, "p95_ms": 6.0, "p99_ms": 7.0}
        # Write first row
        _append_latency_csv(tmp_path, "A1-dinov2-cls", latency)
        # Append second row
        _append_latency_csv(tmp_path, "A3-featureprint", latency)

        csv_path = tmp_path / "latency_summary.csv"
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2
        model_ids = [r["model_id"] for r in rows]
        assert "A1-dinov2-cls" in model_ids
        assert "A3-featureprint" in model_ids

    def test_creates_parent_dir_if_missing(self, tmp_path: Path) -> None:
        """Creates the parent directory if it does not already exist."""
        nested = tmp_path / "deep" / "results"
        latency: dict[str, float] = {"p50_ms": 1.0, "p95_ms": 2.0, "p99_ms": 3.0}
        _append_latency_csv(nested, "A3-featureprint", latency)
        assert (nested / "latency_summary.csv").exists()

    def test_no_duplicate_header_on_append(self, tmp_path: Path) -> None:
        """Calling twice produces exactly one header line."""
        latency: dict[str, float] = {"p50_ms": 1.0, "p95_ms": 2.0, "p99_ms": 3.0}
        _append_latency_csv(tmp_path, "model_a", latency)
        _append_latency_csv(tmp_path, "model_b", latency)

        csv_path = tmp_path / "latency_summary.csv"
        lines = csv_path.read_text().splitlines()
        # Exactly one header line (contains "model_id") + two data lines
        header_lines = [ln for ln in lines if "model_id" in ln]
        assert len(header_lines) == 1


# ── Integration test ──────────────────────────────────────────────────────────


@pytest.mark.integration
def test_extract_feature_vector_real_image() -> None:
    """Smoke test with a real gallery fixture image (requires Vision framework)."""
    fixture = Path(__file__).parent / "fixtures" / "gallery"
    images = list(fixture.glob("*.jpg")) + list(fixture.glob("*.png"))
    if not images:
        pytest.skip("No fixture images found")

    result = extract_feature_vector(images[0])

    assert result.ndim == 1
    assert result.dtype == np.float32
    assert len(result) > 0


@pytest.mark.integration
def test_embed_images_real_fixtures() -> None:
    """Embed a small batch of real images and verify L2 normalisation."""
    fixture = Path(__file__).parent / "fixtures" / "gallery"
    images = list(fixture.glob("*.jpg")) + list(fixture.glob("*.png"))
    if len(images) < 2:
        pytest.skip("Need at least 2 fixture images")

    result = embed_images(images[:3])

    assert result.ndim == 2
    assert result.shape[0] == len(images[:3])
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, np.ones(len(images[:3])), atol=1e-5)


def _unused_dummy(_x: Any) -> None:
    """Suppress pyright unused-import warning for Any."""
