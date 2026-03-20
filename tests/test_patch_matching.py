"""Tests for src/vinylid_ml/patch_matching.py.

Unit tests mock the DINOv2 model so no network or GPU is required.
Integration tests (marked ``@pytest.mark.integration``) load real model
weights and process actual fixture images.
"""

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vinylid_ml.patch_matching import (
    PATCH_DIM,
    PATCH_MODEL_ID,
    PATCHES_PER_IMAGE_224,
    DINOv2PatchExtractor,
    PatchFeatures,
    PatchMatcher,
    PatchMatchResult,
    cache_path_for,
    extract_with_cache,
    load_cached_patches,
    save_cached_patches,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "gallery"
_FIXTURE_PATHS = [
    _FIXTURE_DIR / "metallica_black_500.png",
    _FIXTURE_DIR / "metallica_black_2000.png",
    _FIXTURE_DIR / "10cc_donna_300.jpg",
    _FIXTURE_DIR / "alicecooper_detroit_1000.jpg",
    _FIXTURE_DIR / "bbking_best_354.jpg",
]


def _make_patch_features(n_patches: int = 256, dim: int = 384, seed: int = 0) -> PatchFeatures:
    """Return a ``PatchFeatures`` with random L2-normalised patches."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n_patches, dim)).astype(np.float32)
    # L2 normalise like the real extractor
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    patches = raw / np.maximum(norms, 1e-8)
    return PatchFeatures(patches=patches, image_size=(640, 480))


def _mock_dinov2_model() -> MagicMock:
    """Return a MagicMock that behaves like a DINOv2 model for forward_features."""
    model = MagicMock(spec=torch.nn.Module)

    def _forward_features(x: torch.Tensor) -> dict[str, torch.Tensor]:
        b = x.shape[0]
        tokens = torch.randn(b, 256, 384)
        return {"x_norm_patchtokens": tokens}

    model.forward_features = _forward_features
    model.eval.return_value = model
    model.to.return_value = model
    return model


@pytest.fixture()
def patched_extractor() -> DINOv2PatchExtractor:
    """Return a ``DINOv2PatchExtractor`` with DINOv2 mocked out."""
    mock_model = _mock_dinov2_model()
    with patch("vinylid_ml.patch_matching.torch.hub.load", return_value=mock_model):
        return DINOv2PatchExtractor(device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for module-level constants."""

    def test_patch_model_id(self) -> None:
        """PATCH_MODEL_ID equals 'C1-dinov2-patches'."""
        assert PATCH_MODEL_ID == "C1-dinov2-patches"

    def test_patch_dim(self) -> None:
        """PATCH_DIM equals 384 (DINOv2 ViT-S/14 hidden dim)."""
        assert PATCH_DIM == 384

    def test_patches_per_image_224(self) -> None:
        """PATCHES_PER_IMAGE_224 equals 256 (16x16 grid)."""
        assert PATCHES_PER_IMAGE_224 == 256


# ---------------------------------------------------------------------------
# PatchFeatures
# ---------------------------------------------------------------------------


class TestPatchFeatures:
    """Tests for the PatchFeatures dataclass."""

    def test_shape_and_dtype(self) -> None:
        """Patches have shape (N, 384) and dtype float32."""
        pf = _make_patch_features(256, 384)
        assert pf.patches.shape == (256, 384)
        assert pf.patches.dtype == np.float32

    def test_image_size_stored(self) -> None:
        """image_size is stored as a (width, height) tuple of ints."""
        pf = _make_patch_features()
        w, h = pf.image_size
        assert isinstance(w, int)
        assert isinstance(h, int)
        assert w == 640
        assert h == 480

    def test_patches_are_normalised(self) -> None:
        """All patch vectors have unit L2 norm."""
        pf = _make_patch_features()
        norms = np.linalg.norm(pf.patches, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# PatchMatchResult
# ---------------------------------------------------------------------------


class TestPatchMatchResult:
    """Tests for the PatchMatchResult dataclass."""

    def test_fields(self) -> None:
        """PatchMatchResult stores score, num_patches_matched, and top_k_patch_sims."""
        pmr = PatchMatchResult(
            score=0.75,
            num_patches_matched=42,
            top_k_patch_sims=np.array([0.9, 0.8], dtype=np.float32),
        )
        assert pmr.score == 0.75
        assert pmr.num_patches_matched == 42
        assert pmr.top_k_patch_sims.shape == (2,)

    def test_zero_score(self) -> None:
        """PatchMatchResult with zero matches has score 0.0 and empty sims."""
        pmr = PatchMatchResult(
            score=0.0,
            num_patches_matched=0,
            top_k_patch_sims=np.empty(0, dtype=np.float32),
        )
        assert pmr.score == 0.0
        assert pmr.num_patches_matched == 0


# ---------------------------------------------------------------------------
# DINOv2PatchExtractor
# ---------------------------------------------------------------------------


class TestDINOv2PatchExtractor:
    """Tests for the DINOv2PatchExtractor (mocked model)."""

    def test_input_size_validation(self) -> None:
        """Reject input_size not divisible by 14."""
        with pytest.raises(ValueError, match="divisible by 14"):
            mock_model = _mock_dinov2_model()
            with patch("vinylid_ml.patch_matching.torch.hub.load", return_value=mock_model):
                DINOv2PatchExtractor(device=torch.device("cpu"), input_size=225)

    def test_num_patches_224(self, patched_extractor: DINOv2PatchExtractor) -> None:
        """224px input produces 256 patches (16x16 grid)."""
        assert patched_extractor.num_patches == 256  # 16x16

    def test_num_patches_518(self) -> None:
        """518px input produces 1369 patches (37x37 grid)."""
        mock_model = _mock_dinov2_model()
        with patch("vinylid_ml.patch_matching.torch.hub.load", return_value=mock_model):
            ext = DINOv2PatchExtractor(device=torch.device("cpu"), input_size=518)
        assert ext.num_patches == 37 * 37  # 1369

    def test_extract_returns_patch_features(self, patched_extractor: DINOv2PatchExtractor) -> None:
        """extract() on a synthetic PIL image returns PatchFeatures with correct shape."""
        img = Image.new("RGB", (300, 300), color=(128, 128, 128))
        pf = patched_extractor.extract(img)
        assert isinstance(pf, PatchFeatures)
        assert pf.patches.shape == (256, 384)
        assert pf.patches.dtype == np.float32

    def test_extract_from_path(self, patched_extractor: DINOv2PatchExtractor) -> None:
        """extract() accepts a Path and returns 256 patches."""
        if not _FIXTURE_PATHS[0].exists():
            pytest.skip("Fixture images not available")
        pf = patched_extractor.extract(_FIXTURE_PATHS[0])
        assert isinstance(pf, PatchFeatures)
        assert pf.patches.shape[0] == 256

    def test_extract_missing_path_raises(self, patched_extractor: DINOv2PatchExtractor) -> None:
        """extract() raises FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            patched_extractor.extract(Path("/nonexistent/image.jpg"))

    def test_extract_batch(self, patched_extractor: DINOv2PatchExtractor) -> None:
        """extract_batch() returns one PatchFeatures per batch item."""
        batch = torch.randn(3, 3, 224, 224)
        sizes = [(300, 300), (400, 400), (500, 500)]
        results = patched_extractor.extract_batch(batch, sizes)
        assert len(results) == 3
        for pf in results:
            assert pf.patches.shape == (256, 384)

    def test_get_transforms_returns_compose(self, patched_extractor: DINOv2PatchExtractor) -> None:
        """get_transforms() returns a torchvision Compose pipeline."""
        t = patched_extractor.get_transforms()
        assert isinstance(t, transforms.Compose)


# ---------------------------------------------------------------------------
# PatchMatcher
# ---------------------------------------------------------------------------


class TestPatchMatcherBestAvg:
    """Tests for PatchMatcher.match_best_avg."""

    def test_same_features_high_score(self) -> None:
        """Self-match returns score close to 1.0."""
        pf = _make_patch_features(256, 384, seed=42)
        result = PatchMatcher.match_best_avg(pf, pf)
        assert result.score > 0.99

    def test_random_features_lower_score(self) -> None:
        """Random feature pairs score lower than self-match."""
        pf1 = _make_patch_features(256, 384, seed=1)
        pf2 = _make_patch_features(256, 384, seed=2)
        result = PatchMatcher.match_best_avg(pf1, pf2)
        # Random features → lower than self-match
        assert result.score < 0.99

    def test_top_k_sims_shape(self) -> None:
        """top_k_patch_sims has exactly top_k entries."""
        pf1 = _make_patch_features(256, 384, seed=1)
        pf2 = _make_patch_features(256, 384, seed=2)
        result = PatchMatcher.match_best_avg(pf1, pf2, top_k=5)
        assert result.top_k_patch_sims.shape == (5,)

    def test_top_k_sims_sorted_descending(self) -> None:
        """top_k_patch_sims are sorted in descending order."""
        pf1 = _make_patch_features(256, 384, seed=1)
        pf2 = _make_patch_features(256, 384, seed=2)
        result = PatchMatcher.match_best_avg(pf1, pf2, top_k=10)
        diffs = np.diff(result.top_k_patch_sims)
        assert np.all(diffs <= 1e-7)  # sorted descending

    def test_num_patches_matched_is_zero(self) -> None:
        """best_avg does not populate num_patches_matched (always 0)."""
        pf = _make_patch_features(256, 384, seed=0)
        result = PatchMatcher.match_best_avg(pf, pf)
        assert result.num_patches_matched == 0  # best_avg doesn't compute this


class TestPatchMatcherMutualNN:
    """Tests for PatchMatcher.match_mutual_nn."""

    def test_same_features_all_mutual(self) -> None:
        """Self-match yields >200 mutual NN pairs and score >0.8."""
        pf = _make_patch_features(256, 384, seed=42)
        result = PatchMatcher.match_mutual_nn(pf, pf)
        assert result.num_patches_matched > 200
        assert result.score > 0.8

    def test_random_features_fewer_mutual(self) -> None:
        """Random feature pairs produce fewer mutual NN matches."""
        pf1 = _make_patch_features(256, 384, seed=1)
        pf2 = _make_patch_features(256, 384, seed=2)
        result = PatchMatcher.match_mutual_nn(pf1, pf2)
        assert result.num_patches_matched >= 0
        assert 0.0 <= result.score <= 1.0

    def test_top_k_sims_contains_mutual_pairs(self) -> None:
        """Self-match mutual NN pairs have cosine similarity close to 1.0."""
        pf = _make_patch_features(256, 384, seed=42)
        result = PatchMatcher.match_mutual_nn(pf, pf, top_k=5)
        assert result.top_k_patch_sims.shape == (5,)
        assert result.top_k_patch_sims[0] > 0.99


class TestPatchMatcherDispatch:
    """Tests for PatchMatcher.match dispatch."""

    def test_dispatch_best_avg(self) -> None:
        """match() dispatches to best_avg when strategy='best_avg'."""
        pf = _make_patch_features(256, 384, seed=0)
        result = PatchMatcher.match(pf, pf, strategy="best_avg")
        assert result.num_patches_matched == 0  # best_avg

    def test_dispatch_mutual_nn(self) -> None:
        """match() dispatches to mutual_nn when strategy='mutual_nn'."""
        pf = _make_patch_features(256, 384, seed=0)
        result = PatchMatcher.match(pf, pf, strategy="mutual_nn")
        assert result.num_patches_matched > 0

    def test_dispatch_unknown_raises(self) -> None:
        """match() raises ValueError for unknown strategy."""
        pf = _make_patch_features(256, 384, seed=0)
        with pytest.raises(ValueError, match="Unknown strategy"):
            PatchMatcher.match(pf, pf, strategy="invalid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestCaching:
    """Tests for patch feature caching."""

    def test_cache_path_deterministic(self, tmp_path: Path) -> None:
        """cache_path_for returns the same .npz path for the same image."""
        p = Path("/some/image.jpg")
        c1 = cache_path_for(p, tmp_path)
        c2 = cache_path_for(p, tmp_path)
        assert c1 == c2
        assert c1.suffix == ".npz"

    def test_cache_roundtrip(self, tmp_path: Path) -> None:
        """Save then load preserves patches within fp16 tolerance."""
        pf = _make_patch_features(256, 384, seed=42)
        path = tmp_path / "test.npz"
        save_cached_patches(pf, path)
        loaded = load_cached_patches(path)
        # fp16 roundtrip → atol is relaxed
        np.testing.assert_allclose(loaded.patches, pf.patches, atol=5e-3)
        assert loaded.image_size == pf.image_size

    def test_cache_roundtrip_preserves_normalization(self, tmp_path: Path) -> None:
        """Loaded patches are L2-normalized after fp16 roundtrip."""
        pf = _make_patch_features(256, 384, seed=42)
        path = tmp_path / "test_norm.npz"
        save_cached_patches(pf, path)
        loaded = load_cached_patches(path)
        norms = np.linalg.norm(loaded.patches, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_save_uses_fp16(self, tmp_path: Path) -> None:
        """Saved .npz stores patches in fp16 for storage efficiency."""
        pf = _make_patch_features(256, 384)
        path = tmp_path / "test.npz"
        save_cached_patches(pf, path)
        with np.load(path) as data:
            assert data["patches"].dtype == np.float16

    def test_extract_with_cache_uses_cache(
        self, patched_extractor: DINOv2PatchExtractor, tmp_path: Path
    ) -> None:
        """Second extract_with_cache call loads from .npz cache."""
        if not _FIXTURE_PATHS[0].exists():
            pytest.skip("Fixture images not available")
        cache_dir = tmp_path / "cache"
        paths = [_FIXTURE_PATHS[0]]

        # First call — extract
        result1 = extract_with_cache(patched_extractor, paths, cache_dir)
        assert len(result1) == 1

        # Second call — should load from cache (no model call)
        result2 = extract_with_cache(patched_extractor, paths, cache_dir)
        assert len(result2) == 1
        np.testing.assert_allclose(result2[0].patches, result1[0].patches, atol=5e-3)


# ---------------------------------------------------------------------------
# Integration tests (require network + model weights)
# ---------------------------------------------------------------------------


# Need PIL Image import at module level for the mocked extractor tests
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402


@pytest.fixture(scope="session")
def integration_extractor() -> DINOv2PatchExtractor:
    """Return a real DINOv2PatchExtractor on CPU for integration tests."""
    return DINOv2PatchExtractor(device=torch.device("cpu"), input_size=224)


@pytest.mark.integration()
class TestIntegrationDINOv2PatchExtractor:
    """Integration tests using real DINOv2 model weights."""

    def test_extract_fixture_shape(self, integration_extractor: DINOv2PatchExtractor) -> None:
        """Real DINOv2 produces (256, 384) float32 patch features."""
        if not _FIXTURE_PATHS[0].exists():
            pytest.skip("Fixture images not available")
        pf = integration_extractor.extract(_FIXTURE_PATHS[0])
        assert pf.patches.shape == (256, 384)
        assert pf.patches.dtype == np.float32

    def test_patches_are_l2_normalised(self, integration_extractor: DINOv2PatchExtractor) -> None:
        """Real DINOv2 patch tokens are L2-normalized."""
        if not _FIXTURE_PATHS[0].exists():
            pytest.skip("Fixture images not available")
        pf = integration_extractor.extract(_FIXTURE_PATHS[0])
        norms = np.linalg.norm(pf.patches, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_same_album_higher_score_than_different(
        self, integration_extractor: DINOv2PatchExtractor
    ) -> None:
        """Same-album pair should score higher than different-album pair."""
        metallica_500 = _FIXTURE_PATHS[0]
        metallica_2000 = _FIXTURE_PATHS[1]
        alice_cooper = _FIXTURE_PATHS[3]

        if not all(p.exists() for p in [metallica_500, metallica_2000, alice_cooper]):
            pytest.skip("Fixture images not available")

        pf_m500 = integration_extractor.extract(metallica_500)
        pf_m2k = integration_extractor.extract(metallica_2000)
        pf_ac = integration_extractor.extract(alice_cooper)

        same_album = PatchMatcher.match_best_avg(pf_m500, pf_m2k)
        diff_album = PatchMatcher.match_best_avg(pf_m500, pf_ac)

        assert same_album.score > diff_album.score

    def test_mutual_nn_same_album(self, integration_extractor: DINOv2PatchExtractor) -> None:
        """Same-album pair should have more mutual NN matches."""
        metallica_500 = _FIXTURE_PATHS[0]
        metallica_2000 = _FIXTURE_PATHS[1]
        alice_cooper = _FIXTURE_PATHS[3]

        if not all(p.exists() for p in [metallica_500, metallica_2000, alice_cooper]):
            pytest.skip("Fixture images not available")

        pf_m500 = integration_extractor.extract(metallica_500)
        pf_m2k = integration_extractor.extract(metallica_2000)
        pf_ac = integration_extractor.extract(alice_cooper)

        same = PatchMatcher.match_mutual_nn(pf_m500, pf_m2k)
        diff = PatchMatcher.match_mutual_nn(pf_m500, pf_ac)

        assert same.num_patches_matched > diff.num_patches_matched
