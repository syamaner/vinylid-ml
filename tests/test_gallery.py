"""Tests for vinylid_ml.gallery — gallery embedding pipeline.

Organized into:
- Unit tests: save/load roundtrip, dataset structure, metadata (no model download)
- Integration tests: actual embedding extraction (downloads model on first run)
"""

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownLambdaType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from vinylid_ml.gallery import (
    EmbeddingResult,
    GalleryImageDataset,
    embed_dataset,
    load_embeddings,
    save_embeddings,
)

# ============================================================
# Unit tests — no model download required
# ============================================================


def _make_embedding_result(num_images: int = 10, embedding_dim: int = 384) -> EmbeddingResult:
    """Create a mock EmbeddingResult for testing."""
    rng = np.random.default_rng(42)
    return EmbeddingResult(
        embeddings=rng.standard_normal((num_images, embedding_dim)).astype(np.float16),
        image_paths=[f"artist/album/img_{i}.jpg" for i in range(num_images)],
        album_ids=[f"album_{i % 3}" for i in range(num_images)],
        model_id="A1-dinov2-cls",
        embedding_dim=embedding_dim,
    )


class TestSaveLoadRoundtrip:
    """Test saving and loading embeddings produces identical results."""

    def test_embeddings_roundtrip(self, tmp_path: Path) -> None:
        """Saved and loaded embeddings are identical (float16 exact match)."""
        original = _make_embedding_result()
        save_embeddings(original, tmp_path)
        loaded = load_embeddings(tmp_path, "A1-dinov2-cls")
        np.testing.assert_array_equal(loaded.embeddings, original.embeddings)

    def test_metadata_roundtrip(self, tmp_path: Path) -> None:
        """Saved and loaded metadata fields match exactly."""
        original = _make_embedding_result()
        save_embeddings(original, tmp_path)
        loaded = load_embeddings(tmp_path, "A1-dinov2-cls")
        assert loaded.image_paths == original.image_paths
        assert loaded.album_ids == original.album_ids
        assert loaded.model_id == original.model_id
        assert loaded.embedding_dim == original.embedding_dim

    def test_output_files_exist(self, tmp_path: Path) -> None:
        """save_embeddings creates the expected directory and files."""
        result = _make_embedding_result()
        save_embeddings(result, tmp_path)
        model_dir = tmp_path / "A1-dinov2-cls"
        assert model_dir.is_dir()
        assert (model_dir / "embeddings.npy").exists()
        assert (model_dir / "metadata.json").exists()

    def test_embeddings_dtype_is_float16(self, tmp_path: Path) -> None:
        """Saved embeddings are stored as float16."""
        result = _make_embedding_result()
        save_embeddings(result, tmp_path)
        loaded_npy = np.load(tmp_path / "A1-dinov2-cls" / "embeddings.npy")
        assert loaded_npy.dtype == np.float16


class TestMetadataJson:
    """Test metadata.json structure and required fields."""

    def test_required_fields_present(self, tmp_path: Path) -> None:
        """metadata.json contains all required fields."""
        result = _make_embedding_result()
        save_embeddings(result, tmp_path)
        with (tmp_path / "A1-dinov2-cls" / "metadata.json").open() as f:
            meta = json.load(f)
        required = {
            "model_id",
            "embedding_dim",
            "num_images",
            "image_paths",
            "album_ids",
            "timestamp",
        }
        assert required <= set(meta.keys())

    def test_num_images_matches(self, tmp_path: Path) -> None:
        """num_images in metadata matches actual embedding count."""
        result = _make_embedding_result(num_images=7)
        save_embeddings(result, tmp_path)
        with (tmp_path / "A1-dinov2-cls" / "metadata.json").open() as f:
            meta = json.load(f)
        assert meta["num_images"] == 7


class TestGalleryImageDataset:
    """Test the lightweight GalleryImageDataset."""

    @pytest.fixture
    def sample_gallery(self, tmp_path: Path) -> tuple[pd.DataFrame, Path]:
        """Create a small gallery of synthetic images with a manifest DataFrame."""
        gallery_root = tmp_path / "gallery"
        image_paths = []
        for i in range(4):
            artist_dir = gallery_root / f"Artist_{i}" / f"Album_{i}"
            artist_dir.mkdir(parents=True, exist_ok=True)
            img_path = artist_dir / f"cover_{i}.jpg"
            Image.new("RGB", (300, 300), color=(i * 60, 100, 200)).save(img_path)
            image_paths.append(str(img_path.relative_to(gallery_root)))

        manifest = pd.DataFrame(
            {
                "image_path": image_paths,
                "album_id": [f"album_{i}" for i in range(4)],
                "release_id": [f"release_{i}" for i in range(4)],
                "artist": [f"Artist {i}" for i in range(4)],
                "album": [f"Album {i}" for i in range(4)],
            }
        )
        return manifest, gallery_root

    def test_dataset_length(self, sample_gallery: tuple[pd.DataFrame, Path]) -> None:
        """Dataset length matches manifest row count."""
        manifest, gallery_root = sample_gallery
        from torchvision import transforms

        t = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
        ds = GalleryImageDataset(manifest, gallery_root, t)
        assert len(ds) == 4

    def test_item_structure(self, sample_gallery: tuple[pd.DataFrame, Path]) -> None:
        """Each item is a (tensor, image_path, album_id) tuple."""
        manifest, gallery_root = sample_gallery
        from torchvision import transforms

        t = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
        ds = GalleryImageDataset(manifest, gallery_root, t)
        tensor, path, album_id = ds[0]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
        assert isinstance(path, str)
        assert isinstance(album_id, str)


class TestEmbedDataset:
    """Test embed_dataset with a mock model (no real model download)."""

    def test_output_shape_and_dtype(self, tmp_path: Path) -> None:
        """embed_dataset produces correct shape and float16 dtype."""
        # Create synthetic gallery
        gallery_root = tmp_path / "gallery"
        image_paths = []
        for i in range(6):
            d = gallery_root / f"Artist_{i}" / f"Album_{i}"
            d.mkdir(parents=True, exist_ok=True)
            img_path = d / "cover.jpg"
            Image.new("RGB", (256, 256), color=(i * 40, 80, 160)).save(img_path)
            image_paths.append(str(img_path.relative_to(gallery_root)))

        manifest = pd.DataFrame(
            {
                "image_path": image_paths,
                "album_id": [f"album_{i}" for i in range(6)],
            }
        )

        from torchvision import transforms

        t = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

        # Mock model
        mock_model = MagicMock()
        mock_model.model_id = "test-model"
        mock_model.embedding_dim = 128
        mock_model.get_transforms.return_value = t
        mock_model.embed.side_effect = lambda imgs: torch.randn(imgs.shape[0], 128)

        ds = GalleryImageDataset(manifest, gallery_root, t)
        result = embed_dataset(mock_model, ds, batch_size=4, num_workers=0)

        assert result.embeddings.shape == (6, 128)
        assert result.embeddings.dtype == np.float16
        assert len(result.image_paths) == 6
        assert len(result.album_ids) == 6
        assert result.model_id == "test-model"

    def test_image_paths_preserved(self, tmp_path: Path) -> None:
        """embed_dataset preserves image paths in original order."""
        gallery_root = tmp_path / "gallery"
        image_paths = []
        for i in range(3):
            d = gallery_root / f"A_{i}" / f"B_{i}"
            d.mkdir(parents=True, exist_ok=True)
            img_path = d / "cover.jpg"
            Image.new("RGB", (100, 100)).save(img_path)
            image_paths.append(str(img_path.relative_to(gallery_root)))

        manifest = pd.DataFrame(
            {
                "image_path": image_paths,
                "album_id": [f"a_{i}" for i in range(3)],
            }
        )

        from torchvision import transforms

        t = transforms.Compose(
            [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]
        )
        mock_model = MagicMock()
        mock_model.model_id = "m"
        mock_model.embedding_dim = 8
        mock_model.get_transforms.return_value = t
        mock_model.embed.side_effect = lambda imgs: torch.randn(imgs.shape[0], 8)

        ds = GalleryImageDataset(manifest, gallery_root, t)
        result = embed_dataset(mock_model, ds, batch_size=2, num_workers=0)
        assert result.image_paths == image_paths


# ============================================================
# Integration tests — require model download
# ============================================================


@pytest.mark.integration
class TestEmbedGalleryIntegration:
    """Integration test: full pipeline with real DINOv2 model."""

    def test_embed_small_gallery(self, tmp_path: Path) -> None:
        """Embed 4 synthetic images with DINOv2 CLS, verify shapes and save/load."""
        from vinylid_ml.models import DINOv2Embedder

        model = DINOv2Embedder(pooling="cls")

        # Create synthetic gallery
        gallery_root = tmp_path / "gallery"
        image_paths = []
        for i in range(4):
            d = gallery_root / f"Artist_{i}" / f"Album_{i}"
            d.mkdir(parents=True, exist_ok=True)
            img_path = d / "cover.jpg"
            Image.new("RGB", (300, 300), color=(i * 50, 100, 200)).save(img_path)
            image_paths.append(str(img_path.relative_to(gallery_root)))

        manifest = pd.DataFrame(
            {
                "image_path": image_paths,
                "album_id": [f"album_{i}" for i in range(4)],
            }
        )

        ds = GalleryImageDataset(manifest, gallery_root, model.get_transforms())
        result = embed_dataset(model, ds, batch_size=2, num_workers=0)

        # Verify shapes
        assert result.embeddings.shape == (4, 384)
        assert result.embeddings.dtype == np.float16
        assert result.model_id == "A1-dinov2-cls"

        # Verify save/load roundtrip
        save_embeddings(result, tmp_path / "output")
        loaded = load_embeddings(tmp_path / "output", "A1-dinov2-cls")
        np.testing.assert_array_equal(loaded.embeddings, result.embeddings)
