"""Tests for vinylid_ml.training — training infrastructure.

Split into:
- Unit tests: MultiViewTransform, TrainingConfig, FineTuneModel properties (no model download)
- Integration tests: actual FineTuneModel forward pass (downloads models on first run)
"""

# pyright: reportPrivateUsage=false
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import transforms

from vinylid_ml.training import (
    BACKBONE_DIMS,
    FineTuneModel,
    MultiViewTransform,
    TrainingConfig,
)

# ============================================================
# Unit tests — no model download required
# ============================================================


class TestMultiViewTransform:
    """Tests for the multi-view augmentation wrapper."""

    @pytest.fixture
    def simple_transform(self) -> transforms.Compose:
        """A basic transform for testing (resize + to tensor)."""
        return transforms.Compose(
            [
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
            ]
        )

    @pytest.fixture
    def sample_image(self) -> Image.Image:
        """A small 64x64 RGB test image."""
        return Image.fromarray(torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8).numpy())

    def test_returns_correct_number_of_views(
        self, simple_transform: transforms.Compose, sample_image: Image.Image
    ) -> None:
        """MultiViewTransform produces n_views views."""
        mvt = MultiViewTransform(simple_transform, n_views=3)
        result = mvt(sample_image)
        assert result.shape[0] == 3

    def test_default_is_two_views(
        self, simple_transform: transforms.Compose, sample_image: Image.Image
    ) -> None:
        """Default n_views is 2."""
        mvt = MultiViewTransform(simple_transform)
        result = mvt(sample_image)
        assert result.shape[0] == 2

    def test_output_shape(
        self, simple_transform: transforms.Compose, sample_image: Image.Image
    ) -> None:
        """Output has shape (n_views, C, H, W)."""
        mvt = MultiViewTransform(simple_transform, n_views=2)
        result = mvt(sample_image)
        assert result.shape == (2, 3, 32, 32)

    def test_views_differ_deterministically(self, sample_image: Image.Image) -> None:
        """Each view receives an independent transform application."""

        class _CountingTransform:
            """Deterministic transform that encodes call order."""

            def __init__(self) -> None:
                self.call_count = 0
                self._base = transforms.Compose(
                    [transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]
                )

            def __call__(self, img: Image.Image) -> torch.Tensor:
                self.call_count += 1
                tensor: torch.Tensor = self._base(img)  # type: ignore[assignment]
                return tensor + float(self.call_count)  # offset by call index

        counting = _CountingTransform()
        mvt = MultiViewTransform(counting, n_views=2)
        result = mvt(sample_image)
        # Views should differ deterministically (offset by 1.0 vs 2.0)
        assert not torch.allclose(result[0], result[1])
        assert counting.call_count == 2

    def test_n_views_property(self, simple_transform: transforms.Compose) -> None:
        """n_views property reflects constructor arg."""
        mvt = MultiViewTransform(simple_transform, n_views=5)
        assert mvt.n_views == 5

    def test_invalid_n_views_zero(self, simple_transform: transforms.Compose) -> None:
        """n_views=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_views must be"):
            MultiViewTransform(simple_transform, n_views=0)

    def test_invalid_n_views_negative(self, simple_transform: transforms.Compose) -> None:
        """Negative n_views raises ValueError."""
        with pytest.raises(ValueError, match="n_views must be"):
            MultiViewTransform(simple_transform, n_views=-1)


class TestTrainingConfig:
    """Tests for the TrainingConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config has reasonable values."""
        config = TrainingConfig()
        assert config.backbone == "dinov2"
        assert config.loss == "arcface"
        assert config.epochs == 10
        assert config.seed == 42

    def test_to_dict(self) -> None:
        """to_dict produces a plain dictionary with all fields."""
        config = TrainingConfig(backbone="mobilenet_v3_small", epochs=50)
        d = config.to_dict()
        assert d["backbone"] == "mobilenet_v3_small"
        assert d["epochs"] == 50
        assert "seed" in d

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """Config survives a save → load roundtrip."""
        original = TrainingConfig(
            backbone="sscd",
            loss="proxy-anchor",
            lr=3e-5,
            epochs=20,
            subset_albums=2000,
            git_sha="abc123",
        )
        path = tmp_path / "config.json"
        original.save(path)
        loaded = TrainingConfig.load(path)

        assert loaded.backbone == "sscd"
        assert loaded.loss == "proxy-anchor"
        assert loaded.lr == pytest.approx(3e-5)
        assert loaded.epochs == 20
        assert loaded.subset_albums == 2000
        assert loaded.git_sha == "abc123"

    def test_load_ignores_unknown_fields(self, tmp_path: Path) -> None:
        """Loading a config with extra fields doesn't crash."""
        import json

        path = tmp_path / "config.json"
        data = {"backbone": "dinov2", "unknown_field": "should_be_ignored", "epochs": 5}
        with path.open("w") as f:
            json.dump(data, f)

        loaded = TrainingConfig.load(path)
        assert loaded.backbone == "dinov2"
        assert loaded.epochs == 5

    def test_extra_metadata(self) -> None:
        """Extra field accepts arbitrary metadata."""
        config = TrainingConfig(extra={"torch_version": "2.2.0", "notes": "test run"})
        d = config.to_dict()
        assert d["extra"]["torch_version"] == "2.2.0"  # type: ignore[index]


class TestBackboneDims:
    """Tests for the BACKBONE_DIMS constant."""

    def test_dinov2_dim(self) -> None:
        assert BACKBONE_DIMS["dinov2"] == 384

    def test_mobilenet_dim(self) -> None:
        assert BACKBONE_DIMS["mobilenet_v3_small"] == 576

    def test_sscd_dim(self) -> None:
        assert BACKBONE_DIMS["sscd"] == 512


class TestFineTuneModelProperties:
    """Test FineTuneModel properties without loading actual models."""

    def test_unknown_backbone_raises(self) -> None:
        """ValueError for unknown backbone name."""
        with pytest.raises(ValueError, match="Unknown backbone"):
            FineTuneModel(backbone_name="unknown", projection_dim=256)  # type: ignore[arg-type]


# ============================================================
# Integration tests — require model download (cached)
# ============================================================


@pytest.fixture(scope="session")
def dinov2_finetune_model() -> FineTuneModel:
    """Session-scoped DINOv2 FineTuneModel for integration tests."""
    return FineTuneModel(
        backbone_name="dinov2",
        projection_dim=384,
        device=torch.device("cpu"),
        freeze_backbone=True,
    )


@pytest.mark.integration
class TestFineTuneModelDINOv2Integration:
    """Integration tests for DINOv2 FineTuneModel."""

    def test_output_shape(self, dinov2_finetune_model: FineTuneModel) -> None:
        """Forward produces (B, projection_dim) output."""
        images = torch.rand(2, 3, 224, 224)
        embeddings = dinov2_finetune_model(images)
        assert embeddings.shape == (2, 384)

    def test_output_is_l2_normalized(self, dinov2_finetune_model: FineTuneModel) -> None:
        """Output embeddings have unit L2 norm."""
        images = torch.rand(2, 3, 224, 224)
        embeddings = dinov2_finetune_model(images)
        norms = torch.norm(embeddings, p=2, dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-4, rtol=0.0)

    def test_output_on_cpu(self, dinov2_finetune_model: FineTuneModel) -> None:
        """Output is on CPU."""
        images = torch.rand(2, 3, 224, 224)
        embeddings = dinov2_finetune_model(images)
        assert embeddings.device.type == "cpu"

    def test_freeze_and_unfreeze(self, dinov2_finetune_model: FineTuneModel) -> None:
        """Freeze/unfreeze toggles backbone requires_grad."""
        assert dinov2_finetune_model.is_backbone_frozen()

        dinov2_finetune_model.unfreeze_backbone()
        assert not dinov2_finetune_model.is_backbone_frozen()

        dinov2_finetune_model.freeze_backbone()
        assert dinov2_finetune_model.is_backbone_frozen()

    def test_projection_gradients_when_frozen(self, dinov2_finetune_model: FineTuneModel) -> None:
        """Projection head has gradients even when backbone is frozen."""
        dinov2_finetune_model.freeze_backbone()
        images = torch.rand(2, 3, 224, 224)
        embeddings = dinov2_finetune_model(images)
        loss = embeddings.sum()
        loss.backward()

        proj_linear = dinov2_finetune_model.projection[0]
        assert isinstance(proj_linear, torch.nn.Linear)
        assert proj_linear.weight.grad is not None

    def test_backbone_name_property(self, dinov2_finetune_model: FineTuneModel) -> None:
        """backbone_name property returns correct value."""
        assert dinov2_finetune_model.backbone_name == "dinov2"

    def test_projection_dim_property(self, dinov2_finetune_model: FineTuneModel) -> None:
        """projection_dim property returns correct value."""
        assert dinov2_finetune_model.projection_dim == 384
