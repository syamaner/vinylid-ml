"""Unit tests for scripts/evaluate_phone_photos.py helper functions.

Tests cover the pure helper logic:
- _select_gallery_for_phone_eval: gallery selection with optional extra albums
- _load_phone_photo_stripped: EXIF stripping and orientation correction
- _tta_aug: deterministic TTA augmentations
- _apply_alpha_qe: alpha query expansion on embedding matrices
- _embed_single_image: single PIL image embedding via EmbeddingModel
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_phone_photos import (
    _apply_alpha_qe,
    _embed_single_image,
    _load_phone_photo_stripped,
    _select_gallery_for_phone_eval,
    _tta_aug,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        }
    )


def _make_splits(
    album_ids: list[str],
    split: str = "test",
) -> dict[str, str]:
    """Assign all given album_ids to the given split."""
    return {aid: split for aid in album_ids}


# ---------------------------------------------------------------------------
# _select_gallery_for_phone_eval
# ---------------------------------------------------------------------------


class TestSelectGalleryForPhoneEval:
    """Tests for _select_gallery_for_phone_eval."""

    def test_test_split_only_single_image_albums(self) -> None:
        """Single-image test albums produce exactly one gallery entry each."""
        manifest = _make_manifest(
            ["img0.jpg", "img1.jpg", "img2.jpg"],
            ["a1", "a2", "a3"],
        )
        splits = _make_splits(["a1", "a2", "a3"])

        paths, album_ids = _select_gallery_for_phone_eval(manifest, splits)

        assert len(paths) == 3
        assert set(album_ids) == {"a1", "a2", "a3"}

    def test_multi_image_album_picks_highest_resolution(self) -> None:
        """For multi-image albums, the highest-resolution image is chosen."""
        manifest = _make_manifest(
            ["small.jpg", "large.jpg", "medium.jpg"],
            ["a1", "a1", "a1"],
            widths=[200, 1000, 500],
            heights=[200, 1000, 500],
        )
        splits = _make_splits(["a1"])

        paths, album_ids = _select_gallery_for_phone_eval(manifest, splits)

        assert len(paths) == 1
        assert paths[0] == "large.jpg"
        assert album_ids[0] == "a1"

    def test_non_test_albums_excluded_by_default(self) -> None:
        """Albums in train/val split are not included without extra_album_ids."""
        manifest = _make_manifest(
            ["test_img.jpg", "train_img.jpg"],
            ["a_test", "a_train"],
        )
        splits = {"a_test": "test", "a_train": "train"}

        paths, album_ids = _select_gallery_for_phone_eval(manifest, splits)

        assert len(paths) == 1
        assert album_ids[0] == "a_test"

    def test_extra_album_ids_expands_gallery(self) -> None:
        """extra_album_ids adds train/val albums to the gallery."""
        manifest = _make_manifest(
            ["test_img.jpg", "train_img.jpg", "val_img.jpg"],
            ["a_test", "a_train", "a_val"],
        )
        splits = {"a_test": "test", "a_train": "train", "a_val": "val"}

        paths, album_ids = _select_gallery_for_phone_eval(
            manifest, splits, extra_album_ids={"a_train", "a_val"}
        )

        assert len(paths) == 3
        assert set(album_ids) == {"a_test", "a_train", "a_val"}

    def test_extra_album_ids_no_duplicates(self) -> None:
        """An album_id already in test split is not duplicated via extra_album_ids."""
        manifest = _make_manifest(["img.jpg"], ["a1"])
        splits = {"a1": "test"}

        _paths, album_ids = _select_gallery_for_phone_eval(manifest, splits, extra_album_ids={"a1"})

        assert album_ids.count("a1") == 1

    def test_empty_manifest_returns_empty(self) -> None:
        """Empty manifest with test splits produces empty gallery."""
        manifest = _make_manifest([], [])
        splits = {"a1": "test"}

        paths, album_ids = _select_gallery_for_phone_eval(manifest, splits)

        assert paths == []
        assert album_ids == []

    def test_no_test_albums_returns_empty(self) -> None:
        """If all albums are in train, gallery is empty (no extra_album_ids)."""
        manifest = _make_manifest(["img.jpg"], ["a1"])
        splits = {"a1": "train"}

        paths, album_ids = _select_gallery_for_phone_eval(manifest, splits)

        assert paths == []
        assert album_ids == []

    def test_paths_and_album_ids_have_matching_lengths(self) -> None:
        """Returned paths and album_ids always have the same length."""
        manifest = _make_manifest(
            ["img0.jpg", "img1.jpg", "img2.jpg", "img3.jpg"],
            ["a1", "a1", "a2", "a3"],
            widths=[100, 800, 400, 200],
            heights=[100, 800, 400, 200],
        )
        splits = _make_splits(["a1", "a2", "a3"])

        paths, album_ids = _select_gallery_for_phone_eval(manifest, splits)

        assert len(paths) == len(album_ids)
        assert len(paths) == 3  # 3 albums → 3 canonical images


# ---------------------------------------------------------------------------
# _load_phone_photo_stripped
# ---------------------------------------------------------------------------


class TestLoadPhonePhotoStripped:
    """Tests for _load_phone_photo_stripped."""

    def test_returns_rgb_image(self, tmp_path: Path) -> None:
        """Output is always RGB regardless of input mode."""
        from PIL import Image

        # Create a small RGBA PNG (has alpha channel)
        img = Image.new("RGBA", (10, 10), color=(128, 64, 32, 200))
        path = tmp_path / "test.png"
        img.save(path)

        result = _load_phone_photo_stripped(path)
        assert result.mode == "RGB"

    def test_output_has_no_exif(self, tmp_path: Path) -> None:
        """Output image has no EXIF metadata (new blank canvas)."""
        from PIL import Image

        img = Image.new("RGB", (20, 20), color=(255, 0, 0))
        path = tmp_path / "test.jpg"
        img.save(path, format="JPEG")

        result = _load_phone_photo_stripped(path)
        exif = result.getexif()
        assert len(exif) == 0, f"Expected no EXIF, got {len(exif)} tags"

    def test_pixel_data_preserved(self, tmp_path: Path) -> None:
        """Pixel values are preserved during EXIF stripping."""
        from PIL import Image

        img = Image.new("RGB", (4, 4), color=(10, 20, 30))
        path = tmp_path / "test.png"
        img.save(path)

        result = _load_phone_photo_stripped(path)
        pixels = np.array(result)
        assert pixels.shape == (4, 4, 3)
        np.testing.assert_array_equal(pixels[0, 0], [10, 20, 30])

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """FileNotFoundError / OSError raised for non-existent file."""
        with pytest.raises(OSError):
            _load_phone_photo_stripped(tmp_path / "nonexistent.jpg")

    def test_exif_orientation_applied(self, tmp_path: Path) -> None:
        """EXIF orientation tag is applied before stripping (portrait iPhone photos)."""
        from PIL import Image

        # Create a landscape image (wider than tall) with EXIF rotation=6 (90° CW),
        # which a phone camera might produce for a portrait shot.
        # After applying EXIF orientation, the output should be taller than wide.
        img = Image.new("RGB", (40, 20), color=(0, 0, 0))

        # Craft minimal EXIF bytes with Orientation=6 (rotate 90° CW)
        # EXIF IFD with one tag: tag=0x0112 (274), type=SHORT, count=1, value=6
        exif_bytes = (
            b"Exif\x00\x00"
            b"II"  # little-endian
            b"\x2a\x00"  # magic 42
            b"\x08\x00\x00\x00"  # IFD offset
            b"\x01\x00"  # 1 IFD entry
            b"\x12\x01"  # tag 0x0112 (Orientation)
            b"\x03\x00"  # type SHORT
            b"\x01\x00\x00\x00"  # count 1
            b"\x06\x00\x00\x00"  # value: 6 (rotate 90° CW)
            b"\x00\x00\x00\x00"  # next IFD offset (none)
        )
        path = tmp_path / "rotated.jpg"
        img.save(path, format="JPEG", exif=exif_bytes)

        result = _load_phone_photo_stripped(path)
        # After 90 deg CW rotation: original 40x20 -> 20x40 (h > w)
        assert result.size[1] > result.size[0], (
            f"Expected portrait orientation after EXIF rotation, got {result.size}"
        )


# ---------------------------------------------------------------------------
# _tta_aug
# ---------------------------------------------------------------------------


class TestTtaAug:
    """Tests for the _tta_aug deterministic TTA augmentation helper."""

    def test_aug_idx_0_returns_same_object(self) -> None:
        """Index 0 (identity) returns the original image unchanged."""
        from PIL import Image

        img = Image.new("RGB", (32, 32), color=(128, 0, 0))
        result = _tta_aug(img, 0)
        assert result is img

    def test_aug_idx_1_hflip_changes_image(self) -> None:
        """Index 1 (horizontal flip) produces a different pixel layout."""
        from PIL import Image

        # Asymmetric: left column is red, right is blue
        img = Image.new("RGB", (4, 4), color=(255, 0, 0))
        img.paste(Image.new("RGB", (2, 4), color=(0, 0, 255)), (2, 0))

        result = _tta_aug(img, 1)

        assert result is not img
        assert np.array(result)[0, 0].tolist() == [0, 0, 255]  # blue flipped to left

    def test_aug_idx_2_rotates_image(self) -> None:
        """Index 2 (+5° rotation) produces a different image."""
        from PIL import Image

        img = Image.new("RGB", (32, 32), color=(200, 100, 50))
        result = _tta_aug(img, 2)

        assert result is not img
        assert result.size == img.size  # rotation preserves canvas size

    def test_aug_idx_3_rotates_opposite(self) -> None:
        """Index 3 (-5° rotation) produces a different result than index 2."""
        from PIL import Image

        img = Image.new("RGB", (32, 32), color=(200, 100, 50))
        result_2 = _tta_aug(img, 2)
        result_3 = _tta_aug(img, 3)

        # Different rotation angles should produce different pixel values
        assert not np.array_equal(np.array(result_2), np.array(result_3))

    def test_aug_idx_4_adjusts_brightness(self) -> None:
        """Index 4 (brightness +15%) makes the image brighter."""
        from PIL import Image

        img = Image.new("RGB", (8, 8), color=(100, 100, 100))
        result = _tta_aug(img, 4)

        orig_mean = float(np.array(img).mean())
        result_mean = float(np.array(result).mean())
        assert result_mean > orig_mean

    def test_out_of_range_index_returns_original(self) -> None:
        """Index >= 5 (not defined) falls back to returning the original image."""
        from PIL import Image

        img = Image.new("RGB", (8, 8), color=(10, 20, 30))
        result = _tta_aug(img, 99)
        assert result is img


# ---------------------------------------------------------------------------
# _apply_alpha_qe
# ---------------------------------------------------------------------------


class TestApplyAlphaQe:
    """Tests for the _apply_alpha_qe query expansion helper."""

    def _l2_normalize(self, x: np.ndarray) -> np.ndarray:
        """L2-normalize rows of a 2D array."""
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return (x / np.where(norms == 0, 1.0, norms)).astype(np.float32)

    def test_output_shape_matches_input(self) -> None:
        """Output query matrix has same shape as input."""
        rng = np.random.default_rng(42)
        q_mat = self._l2_normalize(rng.random((10, 64)).astype(np.float32))
        g_mat = self._l2_normalize(rng.random((50, 64)).astype(np.float32))

        result = _apply_alpha_qe(q_mat, g_mat, k=5, alpha=0.5)

        assert result.shape == q_mat.shape

    def test_output_rows_are_l2_normalized(self) -> None:
        """Each output row has unit L2 norm."""
        rng = np.random.default_rng(0)
        q_mat = self._l2_normalize(rng.random((8, 32)).astype(np.float32))
        g_mat = self._l2_normalize(rng.random((20, 32)).astype(np.float32))

        result = _apply_alpha_qe(q_mat, g_mat, k=3, alpha=0.5)

        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(q_mat)), atol=1e-5)

    def test_alpha_zero_is_identity(self) -> None:
        """Alpha=0 leaves query unchanged (gallery mean has zero weight)."""
        rng = np.random.default_rng(7)
        q_mat = self._l2_normalize(rng.random((5, 16)).astype(np.float32))
        g_mat = self._l2_normalize(rng.random((10, 16)).astype(np.float32))

        result = _apply_alpha_qe(q_mat, g_mat, k=3, alpha=0.0)

        # alpha=0 means blended = q + 0*gallery_mean = q (already normalized)
        np.testing.assert_allclose(result, q_mat, atol=1e-5)

    def test_larger_k_uses_more_neighbours(self) -> None:
        """K=1 and K=10 produce different expanded queries (different blend)."""
        rng = np.random.default_rng(13)
        q_mat = self._l2_normalize(rng.random((4, 32)).astype(np.float32))
        g_mat = self._l2_normalize(rng.random((30, 32)).astype(np.float32))

        result_k1 = _apply_alpha_qe(q_mat, g_mat, k=1, alpha=0.5)
        result_k10 = _apply_alpha_qe(q_mat, g_mat, k=10, alpha=0.5)

        assert not np.allclose(result_k1, result_k10, atol=1e-3)

    def test_output_dtype_is_float32(self) -> None:
        """Output matrix dtype is float32."""
        rng = np.random.default_rng(99)
        q_mat = self._l2_normalize(rng.random((3, 8)).astype(np.float32))
        g_mat = self._l2_normalize(rng.random((5, 8)).astype(np.float32))

        result = _apply_alpha_qe(q_mat, g_mat, k=2, alpha=0.3)

        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# _embed_single_image
# ---------------------------------------------------------------------------


class TestEmbedSingleImage:
    """Tests for _embed_single_image (tensor-model wrapper)."""

    def test_calls_model_embed_and_returns_float32(self) -> None:
        """_embed_single_image calls model.embed() and returns float32 array."""
        import torch
        from PIL import Image
        from torchvision import transforms

        # Minimal mock EmbeddingModel
        mock_model = MagicMock()
        mock_model.embed.return_value = torch.rand(1, 512)  # (B, D)

        tfm = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
            ]
        )

        img = Image.new("RGB", (64, 64), color=(255, 128, 0))
        result = _embed_single_image(img, mock_model, tfm)  # type: ignore[arg-type]

        mock_model.embed.assert_called_once()
        assert result.dtype == np.float32
        assert result.ndim == 1
        assert len(result) == 512
