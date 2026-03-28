"""Tests for scripts/embed_finetuned.py — fine-tuned embedding script.

Unit tests only: no model downloads required.
"""

# pyright: reportPrivateUsage=false
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision import transforms

from scripts.embed_finetuned import _collate, _SplitImageDataset, main

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_split_gallery(
    tmp_path: Path,
    albums: dict[str, str],  # album_id → split name
) -> tuple[pd.DataFrame, dict[str, str], Path]:
    """Create a minimal gallery of real images with a manifest and splits dict.

    Args:
        tmp_path: Pytest temp directory.
        albums: Mapping of album_id to split name (e.g. {"a1": "train"}).

    Returns:
        Tuple of (manifest, splits, gallery_root).
    """
    gallery_root = tmp_path / "gallery"
    rows: list[dict[str, str]] = []
    for album_id, _split in albums.items():
        img_dir = gallery_root / album_id
        img_dir.mkdir(parents=True, exist_ok=True)
        img_path = img_dir / "cover.jpg"
        Image.new("RGB", (64, 64), color=(100, 150, 200)).save(img_path)
        rows.append({"image_path": str(img_path.relative_to(gallery_root)), "album_id": album_id})

    manifest = pd.DataFrame(rows)
    splits = albums
    return manifest, splits, gallery_root


def _identity_transform() -> transforms.Compose:
    """Minimal transform for testing (resize to 32px)."""
    return transforms.Compose(
        [
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ]
    )


# ── _collate ──────────────────────────────────────────────────────────────────


class TestCollate:
    """Tests for the _collate helper."""

    def test_returns_three_items(self) -> None:
        """_collate returns (tensor, list[str], list[str])."""
        batch = [
            (torch.zeros(3, 32, 32), "path/a.jpg", "album_a"),
            (torch.zeros(3, 32, 32), "path/b.jpg", "album_b"),
        ]
        tensors, paths, album_ids = _collate(batch)
        assert isinstance(tensors, torch.Tensor)
        assert isinstance(paths, list)
        assert isinstance(album_ids, list)

    def test_tensor_shape(self) -> None:
        """Stacked tensor has shape (batch, C, H, W)."""
        batch = [(torch.zeros(3, 32, 32), f"p{i}.jpg", f"a{i}") for i in range(4)]
        tensors, _, _ = _collate(batch)
        assert tensors.shape == (4, 3, 32, 32)

    def test_paths_preserved(self) -> None:
        """Image paths are returned in original order."""
        paths_in = ["img/a.jpg", "img/b.jpg", "img/c.jpg"]
        batch = [(torch.zeros(3, 8, 8), p, "a") for p in paths_in]
        _, paths_out, _ = _collate(batch)
        assert paths_out == paths_in

    def test_album_ids_preserved(self) -> None:
        """Album IDs are returned in original order."""
        ids_in = ["album_x", "album_y", "album_z"]
        batch = [(torch.zeros(3, 8, 8), "p.jpg", aid) for aid in ids_in]
        _, _, ids_out = _collate(batch)
        assert ids_out == ids_in

    def test_single_item_batch(self) -> None:
        """_collate works correctly with a batch of 1."""
        batch = [(torch.ones(3, 16, 16), "single.jpg", "album_1")]
        tensors, paths, album_ids = _collate(batch)
        assert tensors.shape == (1, 3, 16, 16)
        assert paths == ["single.jpg"]
        assert album_ids == ["album_1"]


# ── _SplitImageDataset ────────────────────────────────────────────────────────


class TestSplitImageDataset:
    """Tests for _SplitImageDataset split filtering and item loading."""

    def test_filters_to_requested_split(self, tmp_path: Path) -> None:
        """Dataset only contains images from the requested split."""
        manifest, splits, gallery_root = _make_split_gallery(
            tmp_path,
            {"a1": "train", "a2": "train", "a3": "val", "a4": "test"},
        )
        ds = _SplitImageDataset(manifest, splits, "train", _identity_transform(), gallery_root)
        assert len(ds) == 2

    def test_val_split_count(self, tmp_path: Path) -> None:
        """Val split contains only val albums."""
        manifest, splits, gallery_root = _make_split_gallery(
            tmp_path,
            {"a1": "train", "a2": "val", "a3": "val"},
        )
        ds = _SplitImageDataset(manifest, splits, "val", _identity_transform(), gallery_root)
        assert len(ds) == 2

    def test_empty_result_for_unknown_split(self, tmp_path: Path) -> None:
        """Split with no matching albums produces an empty dataset."""
        manifest, splits, gallery_root = _make_split_gallery(
            tmp_path,
            {"a1": "train", "a2": "train"},
        )
        ds = _SplitImageDataset(manifest, splits, "test", _identity_transform(), gallery_root)
        assert len(ds) == 0

    def test_item_returns_tensor_path_album(self, tmp_path: Path) -> None:
        """Each item is a (tensor, str, str) tuple with correct types."""
        manifest, splits, gallery_root = _make_split_gallery(
            tmp_path,
            {"a1": "test"},
        )
        ds = _SplitImageDataset(manifest, splits, "test", _identity_transform(), gallery_root)
        tensor, path, album_id = ds[0]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 32, 32)
        assert isinstance(path, str)
        assert album_id == "a1"

    def test_all_split_remapping(self, tmp_path: Path) -> None:
        """'all' split remapping includes all albums mapped to 'combined'."""
        albums = {"a1": "train", "a2": "val", "a3": "test"}
        manifest, splits, gallery_root = _make_split_gallery(tmp_path, albums)

        # Replicate the main() remapping for split='all'
        combined_splits = {
            aid: "combined" for aid, s in splits.items() if s in {"train", "val", "test"}
        }
        ds = _SplitImageDataset(
            manifest, combined_splits, "combined", _identity_transform(), gallery_root
        )
        assert len(ds) == 3

    def test_album_ids_match_split(self, tmp_path: Path) -> None:
        """Returned album_ids belong to the requested split only."""
        manifest, splits, gallery_root = _make_split_gallery(
            tmp_path,
            {"a1": "train", "a2": "train", "a3": "test"},
        )
        ds = _SplitImageDataset(manifest, splits, "train", _identity_transform(), gallery_root)
        _, _, album_id = ds[0]
        assert album_id in {"a1", "a2"}


# ── main() error paths ────────────────────────────────────────────────────────


class TestMainErrorPaths:
    """Tests for main() exit behaviour on bad inputs."""

    def test_exits_when_config_not_found(self, tmp_path: Path) -> None:
        """main() exits with code 1 if config YAML does not exist."""
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--config",
                    str(tmp_path / "nonexistent.yaml"),
                    "--checkpoint",
                    str(tmp_path / "ckpt.pt"),
                    "--model-id",
                    "test-model",
                ]
            )
        assert exc_info.value.code == 1

    def test_exits_when_checkpoint_not_found(self, tmp_path: Path) -> None:
        """main() exits with code 1 if checkpoint file does not exist."""
        config_path = tmp_path / "dataset.yaml"
        config_path.write_text("paths:\n  gallery_root: gallery\n  output_dir: data\n")
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--config",
                    str(config_path),
                    "--checkpoint",
                    str(tmp_path / "missing_checkpoint.pt"),
                    "--model-id",
                    "test-model",
                ]
            )
        assert exc_info.value.code == 1

    def test_exits_when_train_config_not_found(self, tmp_path: Path) -> None:
        """main() exits with code 1 if config.json is missing from checkpoint dir."""
        config_path = tmp_path / "dataset.yaml"
        config_path.write_text("paths:\n  gallery_root: gallery\n  output_dir: data\n")
        # Write a dummy checkpoint file (no config.json alongside it)
        ckpt_path = tmp_path / "run" / "best_checkpoint.pt"
        ckpt_path.parent.mkdir(parents=True)
        ckpt_path.write_bytes(b"dummy")

        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "--config",
                    str(config_path),
                    "--checkpoint",
                    str(ckpt_path),
                    "--model-id",
                    "test-model",
                ]
            )
        assert exc_info.value.code == 1


# ── TrainingConfig patience field ────────────────────────────────────────────


class TestTrainingConfigPatience:
    """Verify patience is now a proper TrainingConfig field."""

    def test_patience_field_exists(self) -> None:
        """TrainingConfig has a patience field defaulting to None."""
        from vinylid_ml.training import TrainingConfig

        config = TrainingConfig()
        assert config.patience is None

    def test_patience_roundtrip(self, tmp_path: Path) -> None:
        """patience survives a save → load roundtrip."""
        from vinylid_ml.training import TrainingConfig

        original = TrainingConfig(patience=10)
        path = tmp_path / "config.json"
        original.save(path)
        loaded = TrainingConfig.load(path)
        assert loaded.patience == 10

    def test_patience_in_to_dict(self) -> None:
        """patience appears as a top-level key in to_dict()."""
        from vinylid_ml.training import TrainingConfig

        config = TrainingConfig(patience=5)
        d = config.to_dict()
        assert "patience" in d
        assert d["patience"] == 5
