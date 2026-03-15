"""Tests for src/vinylid_ml/local_features.py.

Unit tests mock the heavy SuperPoint / LightGlue models so no network or GPU
is required.  Integration tests (marked ``@pytest.mark.integration``) load real
model weights from GitHub Releases and process actual fixture images.
"""

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportPrivateUsage=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vinylid_ml.local_features import (
    LOCAL_FEATURE_MODEL_ID,
    KeypointFeatures,
    LightGlueMatcher,
    LocalFeatureMatcher,
    MatchResult,
    SuperPointExtractor,
    _cache_path_for,
    _feats_to_keypoint_features,
    _keypoint_features_to_tensor,
    _load_cached_features,
    _save_cached_features,
    _to_image_tensor,
)

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "gallery"
_FIXTURE_PATHS = [
    _FIXTURE_DIR / "metallica_black_500.png",
    _FIXTURE_DIR / "metallica_black_2000.png",
    _FIXTURE_DIR / "10cc_donna_300.jpg",
    _FIXTURE_DIR / "alicecooper_detroit_1000.jpg",
    _FIXTURE_DIR / "bbking_best_354.jpg",
]


def _make_kp(n: int = 50, seed: int = 0) -> KeypointFeatures:
    """Return a ``KeypointFeatures`` with synthetic random data."""
    rng = np.random.default_rng(seed)
    return KeypointFeatures(
        keypoints=rng.random((n, 2)).astype(np.float32) * 640,
        descriptors=rng.random((n, 256)).astype(np.float32),
        scores=rng.random((n,)).astype(np.float32),
        image_size=(640, 480),
    )


def _make_lg_raw_output(num_matches: int) -> dict[str, object]:
    """Build a minimal LightGlue raw output dict with ``num_matches`` matches."""
    matches = torch.zeros((num_matches, 2), dtype=torch.int64)
    scores = torch.full((num_matches,), 0.9, dtype=torch.float32)
    # rbd() takes v[0] for list/tensor/ndarray values — wrap in list to match
    # what the real LightGlue model returns (batch dim = 1 → each value is a
    # list of one tensor or a 2-D batched tensor).
    return {
        "matches": [matches],
        "scores": [scores],
        "matches0": [torch.full((50,), -1, dtype=torch.int64)],
        "matches1": [torch.full((50,), -1, dtype=torch.int64)],
        "matching_scores0": [torch.zeros(50)],
        "matching_scores1": [torch.zeros(50)],
        "stop": 5,
        "prune0": [torch.ones(50, dtype=torch.int64)],
        "prune1": [torch.ones(50, dtype=torch.int64)],
    }


@pytest.fixture()
def mock_superpoint() -> MagicMock:
    """Return a MagicMock for the SuperPoint model."""
    return MagicMock()


@pytest.fixture()
def mock_lightglue() -> MagicMock:
    """Return a MagicMock for the LightGlue model."""
    return MagicMock()


@pytest.fixture()
def patched_extractor(mock_superpoint: MagicMock) -> SuperPointExtractor:
    """Return a ``SuperPointExtractor`` with the model mocked out."""
    with patch("vinylid_ml.local_features._load_superpoint", return_value=mock_superpoint):
        return SuperPointExtractor(device=torch.device("cpu"))


@pytest.fixture()
def patched_lg_matcher(mock_lightglue: MagicMock) -> LightGlueMatcher:
    """Return a ``LightGlueMatcher`` with the model mocked out."""
    with patch("vinylid_ml.local_features._load_lightglue", return_value=mock_lightglue):
        return LightGlueMatcher(device=torch.device("cpu"))


@pytest.fixture()
def patched_local_matcher(
    mock_superpoint: MagicMock, mock_lightglue: MagicMock
) -> LocalFeatureMatcher:
    """Return a ``LocalFeatureMatcher`` with both models mocked out."""
    with (
        patch("vinylid_ml.local_features._load_superpoint", return_value=mock_superpoint),
        patch("vinylid_ml.local_features._load_lightglue", return_value=mock_lightglue),
    ):
        return LocalFeatureMatcher(device=torch.device("cpu"))


@pytest.fixture(scope="session")
def integration_extractor_512() -> SuperPointExtractor:
    """Return a real CPU SuperPoint extractor reused across integration tests."""
    return SuperPointExtractor(max_num_keypoints=512, device=torch.device("cpu"))


@pytest.fixture(scope="session")
def integration_extractor_256() -> SuperPointExtractor:
    """Return a real CPU SuperPoint extractor reused across integration tests."""
    return SuperPointExtractor(max_num_keypoints=256, device=torch.device("cpu"))


@pytest.fixture(scope="session")
def integration_lightglue_matcher_cpu() -> LightGlueMatcher:
    """Return a real CPU LightGlue matcher reused across integration tests."""
    return LightGlueMatcher(device=torch.device("cpu"))


@pytest.fixture(scope="session")
def integration_local_matcher_256() -> LocalFeatureMatcher:
    """Return a real CPU LocalFeatureMatcher reused across integration tests."""
    return LocalFeatureMatcher(max_keypoints=256, device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# LOCAL_FEATURE_MODEL_ID
# ---------------------------------------------------------------------------


def test_model_id_constant() -> None:
    """LOCAL_FEATURE_MODEL_ID equals 'C2-superpoint-lightglue'."""
    assert LOCAL_FEATURE_MODEL_ID == "C2-superpoint-lightglue"


# ---------------------------------------------------------------------------
# KeypointFeatures
# ---------------------------------------------------------------------------


class TestKeypointFeatures:
    """Tests for the KeypointFeatures dataclass."""

    def test_fields_have_correct_shapes(self) -> None:
        """Verify keypoints (N,2), descriptors (N,256), scores (N,) shapes."""
        kp = _make_kp(30)
        assert kp.keypoints.shape == (30, 2)
        assert kp.descriptors.shape == (30, 256)
        assert kp.scores.shape == (30,)

    def test_image_size_stored_as_tuple(self) -> None:
        """image_size is a (width, height) tuple of ints."""
        kp = _make_kp()
        w, h = kp.image_size
        assert isinstance(w, int)
        assert isinstance(h, int)

    def test_dtypes_are_float32(self) -> None:
        """All numeric arrays are float32."""
        kp = _make_kp()
        assert kp.keypoints.dtype == np.float32
        assert kp.descriptors.dtype == np.float32
        assert kp.scores.dtype == np.float32


# ---------------------------------------------------------------------------
# MatchResult
# ---------------------------------------------------------------------------


class TestMatchResult:
    """Tests for the MatchResult dataclass."""

    def test_zero_matches(self) -> None:
        """MatchResult with zero matches has empty scores and confidence 0."""
        mr = MatchResult(
            num_matches=0,
            match_scores=np.empty(0, dtype=np.float32),
            confidence=0.0,
        )
        assert mr.num_matches == 0
        assert mr.match_scores.shape == (0,)
        assert mr.confidence == pytest.approx(0.0)

    def test_nonzero_matches(self) -> None:
        """MatchResult with matches correctly stores count and scores."""
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        mr = MatchResult(num_matches=3, match_scores=scores, confidence=float(scores.mean()))
        assert mr.num_matches == 3
        assert mr.match_scores.shape == (3,)
        assert mr.confidence == pytest.approx(0.8, abs=1e-5)

    def test_inlier_aliases_remain_available(self) -> None:
        """Backward-compatible inlier aliases reflect the renamed match fields."""
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        mr = MatchResult(num_matches=3, match_scores=scores, confidence=float(scores.mean()))
        assert mr.num_inliers == 3
        np.testing.assert_array_equal(mr.inlier_scores, scores)


# ---------------------------------------------------------------------------
# _to_image_tensor (private helper)
# ---------------------------------------------------------------------------


class TestToImageTensor:
    """Tests for _to_image_tensor."""

    def test_converts_pil_image_to_float32_tensor(self) -> None:
        """PIL Image → (3, H, W) float32 tensor in [0, 1]."""
        from PIL import Image

        pil = Image.new("RGB", (100, 80), color=(128, 0, 255))
        t = _to_image_tensor(pil, torch.device("cpu"))
        assert t.shape == (3, 80, 100)
        assert t.dtype == torch.float32
        assert float(t.min()) >= 0.0
        assert float(t.max()) <= 1.0

    def test_raises_file_not_found_for_missing_path(self, tmp_path: Path) -> None:
        """Non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Image not found"):
            _to_image_tensor(tmp_path / "missing.jpg", torch.device("cpu"))

    def test_loads_fixture_image_correctly(self, tmp_path: Path) -> None:
        """Path to a valid PNG on disk loads successfully and has 3 channels."""
        from PIL import Image

        img_path = tmp_path / "synthetic.png"
        Image.new("RGB", (64, 64), color=(100, 150, 200)).save(img_path)
        t = _to_image_tensor(img_path, torch.device("cpu"))
        assert t.dim() == 3
        assert t.shape[0] == 3


# ---------------------------------------------------------------------------
# _feats_to_keypoint_features / _keypoint_features_to_tensor (private helpers)
# ---------------------------------------------------------------------------


class TestFeatureConversionHelpers:
    """Round-trip tests for the feats <-> KeypointFeatures conversion helpers."""

    def test_feats_to_keypoint_features_removes_batch_dim(self) -> None:
        """SuperPoint extract() output (batch=1) is squeezed to (N, *) arrays."""
        n = 40
        feats: dict[str, torch.Tensor] = {
            "keypoints": torch.rand(1, n, 2),
            "descriptors": torch.rand(1, n, 256),
            "keypoint_scores": torch.rand(1, n),
            "image_size": torch.tensor([[320.0, 240.0]]),
        }
        kp = _feats_to_keypoint_features(feats)
        assert kp.keypoints.shape == (n, 2)
        assert kp.descriptors.shape == (n, 256)
        assert kp.scores.shape == (n,)
        assert kp.image_size == (320, 240)

    def test_keypoint_features_to_tensor_adds_batch_dim(self) -> None:
        """KeypointFeatures → dict with (1, N, *) tensors for LightGlue."""
        kp = _make_kp(20)
        d = _keypoint_features_to_tensor(kp, torch.device("cpu"))
        assert d["keypoints"].shape == (1, 20, 2)
        assert d["descriptors"].shape == (1, 20, 256)
        assert d["image_size"].shape == (1, 2)

    def test_roundtrip_preserves_values(self) -> None:
        """feats -> KeypointFeatures -> tensor dict recovers original values."""
        n = 15
        kp_arr = np.random.default_rng(7).random((n, 2)).astype(np.float32)
        feats: dict[str, torch.Tensor] = {
            "keypoints": torch.from_numpy(kp_arr).unsqueeze(0),
            "descriptors": torch.rand(1, n, 256),
            "keypoint_scores": torch.rand(1, n),
            "image_size": torch.tensor([[800.0, 600.0]]),
        }
        kp = _feats_to_keypoint_features(feats)
        d = _keypoint_features_to_tensor(kp, torch.device("cpu"))
        np.testing.assert_array_almost_equal(
            d["keypoints"][0].numpy(), kp_arr
        )


# ---------------------------------------------------------------------------
# Feature cache helpers (_cache_path_for, _save/load_cached_features)
# ---------------------------------------------------------------------------


class TestFeatureCache:
    """Tests for .npz feature caching helpers."""

    def test_cache_path_is_sha256_based(self, tmp_path: Path) -> None:
        """Cache filename is SHA-256 of the absolute image path and max_num_keypoints."""
        img_path = tmp_path / "img.jpg"
        cache_dir = tmp_path / "cache"
        max_kp = 512
        result = _cache_path_for(img_path, cache_dir, max_kp)
        expected_key = hashlib.sha256(
            f"{img_path.resolve()}|max_kp={max_kp}".encode()
        ).hexdigest()
        assert result == cache_dir / f"{expected_key}.npz"

    def test_cache_roundtrip_preserves_features(self, tmp_path: Path) -> None:
        """Save then load KeypointFeatures gives identical arrays."""
        kp = _make_kp(25, seed=99)
        cache_file = tmp_path / "test.npz"
        _save_cached_features(kp, cache_file)
        loaded = _load_cached_features(cache_file)

        np.testing.assert_array_equal(loaded.keypoints, kp.keypoints)
        np.testing.assert_array_equal(loaded.descriptors, kp.descriptors)
        np.testing.assert_array_equal(loaded.scores, kp.scores)
        assert loaded.image_size == kp.image_size

    def test_cache_roundtrip_dtypes_are_float32(self, tmp_path: Path) -> None:
        """Loaded arrays are always float32 regardless of saved dtype."""
        kp = _make_kp(10)
        cache_file = tmp_path / "typed.npz"
        _save_cached_features(kp, cache_file)
        loaded = _load_cached_features(cache_file)
        assert loaded.keypoints.dtype == np.float32
        assert loaded.descriptors.dtype == np.float32
        assert loaded.scores.dtype == np.float32


# ---------------------------------------------------------------------------
# SuperPointExtractor
# ---------------------------------------------------------------------------


class TestSuperPointExtractor:
    """Tests for SuperPointExtractor."""

    def test_extract_raises_file_not_found(
        self, patched_extractor: SuperPointExtractor, tmp_path: Path
    ) -> None:
        """extract() raises FileNotFoundError for a missing path."""
        with pytest.raises(FileNotFoundError, match="Image not found"):
            patched_extractor.extract(tmp_path / "nonexistent.jpg")

    def test_extract_batch_returns_one_per_path(
        self, patched_extractor: SuperPointExtractor, tmp_path: Path
    ) -> None:
        """extract_batch() returns one KeypointFeatures per input path."""
        n_kp = 30
        # Patch _to_image_tensor and the model's extract method
        dummy_kp = _make_kp(n_kp)
        with patch.object(patched_extractor, "extract", return_value=dummy_kp):
            results = patched_extractor.extract_batch(list(_FIXTURE_PATHS[:3]))
        assert len(results) == 3
        assert all(r.keypoints.shape == (n_kp, 2) for r in results)


# ---------------------------------------------------------------------------
# LightGlueMatcher
# ---------------------------------------------------------------------------


class TestLightGlueMatcher:
    """Tests for LightGlueMatcher."""

    def test_match_returns_zero_matches_when_no_matches(
        self, patched_lg_matcher: LightGlueMatcher
    ) -> None:
        """When LightGlue finds 0 matches, MatchResult has confidence 0.0."""
        patched_lg_matcher._model.return_value = _make_lg_raw_output(0)
        result = patched_lg_matcher.match(_make_kp(50), _make_kp(50))
        assert result.num_matches == 0
        assert result.confidence == pytest.approx(0.0)
        assert result.match_scores.shape == (0,)

    def test_match_returns_correct_match_count(
        self, patched_lg_matcher: LightGlueMatcher
    ) -> None:
        """num_matches and match_scores length match the mocked match count."""
        patched_lg_matcher._model.return_value = _make_lg_raw_output(15)
        result = patched_lg_matcher.match(_make_kp(50), _make_kp(50))
        assert result.num_matches == 15
        assert result.match_scores.shape == (15,)

    def test_match_confidence_is_mean_of_scores(
        self, patched_lg_matcher: LightGlueMatcher
    ) -> None:
        """confidence is the mean of match_scores."""
        patched_lg_matcher._model.return_value = _make_lg_raw_output(10)
        result = patched_lg_matcher.match(_make_kp(50), _make_kp(50))
        assert result.confidence == pytest.approx(float(result.match_scores.mean()), abs=1e-5)


# ---------------------------------------------------------------------------
# LocalFeatureMatcher
# ---------------------------------------------------------------------------


class TestLocalMatcherRankGallery:
    """Tests for LocalFeatureMatcher.rank_gallery."""

    def test_rank_gallery_sorted_descending_by_match_count(
        self, patched_local_matcher: LocalFeatureMatcher
    ) -> None:
        """rank_gallery returns indices sorted by descending match count."""
        gallery = [_make_kp(50, seed=i) for i in range(4)]
        match_count_sequence = [5, 20, 1, 15]
        with patch.object(
            patched_local_matcher._matcher,
            "count_matches_prepared",
            side_effect=match_count_sequence,
        ):
            ranked = patched_local_matcher.rank_gallery(_make_kp(50), gallery)

        # Expected order: index 1 (20), index 3 (15), index 0 (5), index 2 (1)
        assert list(ranked) == [1, 3, 0, 2]

    def test_rank_gallery_all_zero_matches_returns_all_indices(
        self, patched_local_matcher: LocalFeatureMatcher
    ) -> None:
        """When all match counts are 0, all gallery indices are still returned."""
        gallery = [_make_kp(50, seed=i) for i in range(3)]
        zeros = [0, 0, 0]
        with patch.object(
            patched_local_matcher._matcher,
            "count_matches_prepared",
            side_effect=zeros,
        ):
            ranked = patched_local_matcher.rank_gallery(_make_kp(50), gallery)
        assert set(ranked) == {0, 1, 2}


class TestLocalMatcherExtractFeatures:
    """Tests for LocalFeatureMatcher.extract_features with caching."""

    def test_extract_features_creates_cache_files(
        self, patched_local_matcher: LocalFeatureMatcher, tmp_path: Path
    ) -> None:
        """extract_features saves .npz cache files for each image."""
        dummy_kp = _make_kp(50)
        with patch.object(patched_local_matcher._extractor, "extract", return_value=dummy_kp):
            results = patched_local_matcher.extract_features(
                list(_FIXTURE_PATHS[:2]), cache_dir=tmp_path
            )
        assert len(results) == 2
        npz_files = list(tmp_path.glob("*.npz"))
        assert len(npz_files) == 2

    def test_extract_features_uses_cache_on_second_call(
        self, patched_local_matcher: LocalFeatureMatcher, tmp_path: Path
    ) -> None:
        """Second call with same paths loads from cache without re-extracting."""
        dummy_kp = _make_kp(50)
        mock_extract = MagicMock(return_value=dummy_kp)
        with patch.object(patched_local_matcher._extractor, "extract", mock_extract):
            # First call — should extract
            patched_local_matcher.extract_features(
                list(_FIXTURE_PATHS[:2]), cache_dir=tmp_path
            )
            first_call_count = mock_extract.call_count
            # Second call — should load from cache
            patched_local_matcher.extract_features(
                list(_FIXTURE_PATHS[:2]), cache_dir=tmp_path
            )
        assert first_call_count == 2
        assert mock_extract.call_count == 2  # no extra calls on second run


class TestLocalMatcherMeasureLatency:
    """Tests for LocalFeatureMatcher.measure_latency validation."""

    def test_raises_value_error_for_n_timed_zero(
        self, patched_local_matcher: LocalFeatureMatcher
    ) -> None:
        """n_timed=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_timed must be >= 1"):
            patched_local_matcher.measure_latency(list(_FIXTURE_PATHS[:2]), n_timed=0)

    def test_raises_value_error_for_negative_n_warmup(
        self, patched_local_matcher: LocalFeatureMatcher
    ) -> None:
        """n_warmup=-1 raises ValueError."""
        with pytest.raises(ValueError, match="n_warmup must be >= 0"):
            patched_local_matcher.measure_latency(list(_FIXTURE_PATHS[:2]), n_warmup=-1)

    def test_raises_value_error_for_single_path(
        self, patched_local_matcher: LocalFeatureMatcher
    ) -> None:
        """Providing only 1 path raises ValueError."""
        with pytest.raises(ValueError, match="At least 2 image_paths"):
            patched_local_matcher.measure_latency([_FIXTURE_PATHS[0]])

    def test_returns_latency_dict_with_percentile_keys(
        self, patched_local_matcher: LocalFeatureMatcher
    ) -> None:
        """Happy-path returns dict with p50_ms, p95_ms, p99_ms."""
        dummy_kp = _make_kp(20)
        with (
            patch.object(patched_local_matcher._extractor, "extract", return_value=dummy_kp),
            patch.object(
                patched_local_matcher._matcher,
                "match",
                return_value=MatchResult(5, np.empty(0, np.float32), 0.9),
            ),
        ):
            latency = patched_local_matcher.measure_latency(
                list(_FIXTURE_PATHS[:2]), n_warmup=1, n_timed=5
            )
        assert set(latency.keys()) == {"p50_ms", "p95_ms", "p99_ms"}
        assert all(v >= 0.0 for v in latency.values())


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIntegrationSuperPointExtractor:
    """Integration tests for SuperPointExtractor using real model weights."""

    def test_extract_fixture_image_produces_correct_shapes(
        self, integration_extractor_512: SuperPointExtractor
    ) -> None:
        """Extracted features have correct shapes and dtypes."""
        kp = integration_extractor_512.extract(_FIXTURE_PATHS[0])
        n = kp.keypoints.shape[0]
        assert n <= 512
        assert kp.keypoints.shape == (n, 2)
        assert kp.descriptors.shape == (n, 256)
        assert kp.scores.shape == (n,)
        assert all(isinstance(d, int) for d in kp.image_size)

    def test_extract_all_fixture_images(
        self, integration_extractor_256: SuperPointExtractor
    ) -> None:
        """All 5 fixture images can be extracted without error."""
        for path in _FIXTURE_PATHS:
            kp = integration_extractor_256.extract(path)
            assert kp.keypoints.shape[0] > 0, f"No keypoints detected for {path.name}"


@pytest.mark.integration
class TestIntegrationLightGlue:
    """Integration tests for full SuperPoint + LightGlue pipeline."""

    def test_same_album_has_more_matches_than_different_album(
        self,
        integration_extractor_512: SuperPointExtractor,
        integration_lightglue_matcher_cpu: LightGlueMatcher,
    ) -> None:
        """metallica_black_500 vs metallica_black_2000 has more matches than vs 10cc."""
        kp_query = integration_extractor_512.extract(_FIXTURE_DIR / "metallica_black_500.png")
        kp_same = integration_extractor_512.extract(_FIXTURE_DIR / "metallica_black_2000.png")
        kp_diff = integration_extractor_512.extract(_FIXTURE_DIR / "10cc_donna_300.jpg")

        result_same = integration_lightglue_matcher_cpu.match(kp_query, kp_same)
        result_diff = integration_lightglue_matcher_cpu.match(kp_query, kp_diff)

        assert result_same.num_matches > result_diff.num_matches, (
            f"Same-album matches ({result_same.num_matches}) should exceed "
            f"different-album matches ({result_diff.num_matches})"
        )

    def test_match_result_confidence_in_unit_interval(
        self,
        integration_extractor_256: SuperPointExtractor,
        integration_lightglue_matcher_cpu: LightGlueMatcher,
    ) -> None:
        """MatchResult confidence is always in [0, 1]."""
        kp0 = integration_extractor_256.extract(_FIXTURE_PATHS[0])
        kp1 = integration_extractor_256.extract(_FIXTURE_PATHS[2])
        result = integration_lightglue_matcher_cpu.match(kp0, kp1)
        assert 0.0 <= result.confidence <= 1.0


@pytest.mark.integration
class TestIntegrationLocalFeatureMatcher:
    """End-to-end integration test for LocalFeatureMatcher."""

    def test_end_to_end_rank_gallery_puts_same_album_first(
        self,
        integration_local_matcher_256: LocalFeatureMatcher,
        tmp_path: Path,
    ) -> None:
        """The same-album image (metallica_2000) is ranked #1 against metallica_500."""

        # Gallery: 5 images, only index 1 is the same album as the query
        gallery_feats = integration_local_matcher_256.extract_features(
            list(_FIXTURE_PATHS), cache_dir=tmp_path
        )
        query_feats = gallery_feats[0]  # metallica_black_500 is the query
        # Build gallery without the query itself
        gallery_without_query = gallery_feats[1:]  # metallica_2000 is index 0 of this sub-list

        ranked = integration_local_matcher_256.rank_gallery(query_feats, gallery_without_query)
        # metallica_black_2000.png (index 0 in gallery_without_query) should be first
        assert ranked[0] == 0
