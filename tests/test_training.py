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
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from scripts.train import _build_llrd_param_groups
from vinylid_ml.training import (
    BACKBONE_DIMS,
    FineTuneModel,
    MultiViewTransform,
    TrainingConfig,
)

# ── Mock ViT-like backbone for unit-testing LLRD and partial unfreeze ──────────


class _MockViTBackbone(nn.Module):
    """Minimal ViT-like backbone with patch_embed, blocks, and norm.

    Mimics DINOv2 named-parameter structure without a real model download.
    """

    def __init__(self, num_blocks: int = 4) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(3, 8)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 8))
        self.blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return x


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


class TestTrainingConfigLLRDFields:
    """Tests for the new LLRD-related TrainingConfig fields."""

    def test_defaults(self) -> None:
        """backbone_lr_mult, llrd_decay, unfreeze_blocks have correct defaults."""
        config = TrainingConfig()
        assert config.backbone_lr_mult == pytest.approx(0.1)
        assert config.llrd_decay is None
        assert config.unfreeze_blocks is None

    def test_llrd_fields_roundtrip(self, tmp_path: Path) -> None:
        """LLRD fields survive a save → load roundtrip."""
        original = TrainingConfig(backbone_lr_mult=0.01, llrd_decay=0.9, unfreeze_blocks=4)
        path = tmp_path / "config.json"
        original.save(path)
        loaded = TrainingConfig.load(path)
        assert loaded.backbone_lr_mult == pytest.approx(0.01)
        assert loaded.llrd_decay == pytest.approx(0.9)
        assert loaded.unfreeze_blocks == 4

    def test_llrd_fields_in_to_dict(self) -> None:
        """LLRD fields appear as top-level keys in to_dict()."""
        config = TrainingConfig(backbone_lr_mult=0.01, llrd_decay=0.9)
        d = config.to_dict()
        assert "backbone_lr_mult" in d
        assert "llrd_decay" in d
        assert "unfreeze_blocks" in d


class TestBuildLLRDParamGroups:
    """Unit tests for _build_llrd_param_groups using a mock ViT backbone."""

    def test_non_dinov2_returns_single_group(self) -> None:
        """Non-DINOv2 backbone returns a single uniform LR group."""
        backbone = _MockViTBackbone(num_blocks=4)
        groups = _build_llrd_param_groups(
            backbone, "mobilenet_v3_small", backbone_lr=1e-5, decay=0.9
        )
        assert len(groups) == 1
        assert groups[0]["lr"] == pytest.approx(1e-5)

    def test_dinov2_produces_multiple_groups(self) -> None:
        """DINOv2 LLRD produces more groups than a single uniform group."""
        backbone = _MockViTBackbone(num_blocks=4)
        groups = _build_llrd_param_groups(backbone, "dinov2", backbone_lr=1e-6, decay=0.9)
        assert len(groups) > 1

    def test_outermost_block_highest_lr(self) -> None:
        """Last (outermost) block gets full backbone_lr; inner blocks get less."""
        backbone = _MockViTBackbone(num_blocks=4)
        groups = _build_llrd_param_groups(backbone, "dinov2", backbone_lr=1e-6, decay=0.9)
        lrs = [g["lr"] for g in groups]
        assert max(lrs) == pytest.approx(1e-6)  # outermost block or norm

    def test_lr_monotonically_increases_with_depth(self) -> None:
        """Group LRs from _build_llrd_param_groups are in ascending order."""
        backbone = _MockViTBackbone(num_blocks=4)
        groups = _build_llrd_param_groups(backbone, "dinov2", backbone_lr=1e-6, decay=0.9)
        lrs = [g["lr"] for g in groups]  # original order returned by the function
        assert lrs == sorted(lrs), f"Expected ascending LRs, got {lrs}"

    def test_no_frozen_params_in_groups(self) -> None:
        """No parameter with requires_grad=False appears in any group."""
        backbone = _MockViTBackbone(num_blocks=4)
        # Freeze everything except last block
        for name, param in backbone.named_parameters():
            param.requires_grad = name.startswith("blocks.3")
        groups = _build_llrd_param_groups(backbone, "dinov2", backbone_lr=1e-6, decay=0.9)
        for group in groups:
            for p in group["params"]:
                assert p.requires_grad

    def test_all_trainable_params_covered(self) -> None:
        """Every trainable param appears in exactly one group."""
        backbone = _MockViTBackbone(num_blocks=4)
        groups = _build_llrd_param_groups(backbone, "dinov2", backbone_lr=1e-6, decay=0.9)
        all_group_ids = [id(p) for g in groups for p in g["params"]]
        trainable_ids = [id(p) for p in backbone.parameters() if p.requires_grad]
        assert sorted(all_group_ids) == sorted(trainable_ids)


class TestPartialUnfreezeBackbone:
    """Unit tests for FineTuneModel.partial_unfreeze_backbone using a mock backbone."""

    def _make_model_with_mock_backbone(self, num_blocks: int = 6) -> FineTuneModel:
        """Build a FineTuneModel instance with a mock ViT backbone (no download)."""
        model = FineTuneModel.__new__(FineTuneModel)
        nn.Module.__init__(model)  # must call Module.__init__ before assigning submodules
        mock_backbone = _MockViTBackbone(num_blocks=num_blocks)
        # Freeze all backbone params first
        for p in mock_backbone.parameters():
            p.requires_grad = False
        model.backbone = mock_backbone
        model._backbone_name = "dinov2"  # type: ignore[attr-defined]
        model._projection_dim = 8  # type: ignore[attr-defined]
        model.projection = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
        return model

    def test_last_n_blocks_unfrozen(self) -> None:
        """After partial unfreeze, only the last n_blocks blocks are trainable."""
        model = self._make_model_with_mock_backbone(num_blocks=6)
        model.partial_unfreeze_backbone(n_blocks=2)
        backbone = model.backbone
        for name, param in backbone.named_parameters():
            if name.startswith("blocks."):
                block_idx = int(name.split(".")[1])
                expected = block_idx >= 4  # last 2 of 6
                assert param.requires_grad == expected, f"{name}: expected {expected}"

    def test_norm_always_unfrozen(self) -> None:
        """The final norm layer is always unfrozen in partial unfreeze."""
        model = self._make_model_with_mock_backbone(num_blocks=6)
        model.partial_unfreeze_backbone(n_blocks=1)
        for name, param in model.backbone.named_parameters():
            if name.startswith("norm"):
                assert param.requires_grad

    def test_patch_embed_stays_frozen(self) -> None:
        """patch_embed remains frozen after partial unfreeze."""
        model = self._make_model_with_mock_backbone(num_blocks=6)
        model.partial_unfreeze_backbone(n_blocks=3)
        for name, param in model.backbone.named_parameters():
            if "patch_embed" in name or "cls_token" in name:
                assert not param.requires_grad

    def test_n_blocks_ge_total_unfreezes_all_blocks(self) -> None:
        """n_blocks >= num_blocks unfreezes all transformer blocks."""
        model = self._make_model_with_mock_backbone(num_blocks=4)
        model.partial_unfreeze_backbone(n_blocks=10)  # more than 4 blocks
        for name, param in model.backbone.named_parameters():
            if name.startswith("blocks."):
                assert param.requires_grad

    def test_non_dinov2_falls_back_to_full_unfreeze(self) -> None:
        """Non-DINOv2 backbone fully unfreezes (with a warning)."""
        model = self._make_model_with_mock_backbone(num_blocks=4)
        model._backbone_name = "mobilenet_v3_small"  # type: ignore[attr-defined]
        model.partial_unfreeze_backbone(n_blocks=2)
        # Full unfreeze: all backbone params trainable
        assert all(p.requires_grad for p in model.backbone.parameters())

    def test_n_blocks_zero_raises(self) -> None:
        """n_blocks=0 raises ValueError."""
        model = self._make_model_with_mock_backbone(num_blocks=4)
        with pytest.raises(ValueError, match="n_blocks must be >= 1"):
            model.partial_unfreeze_backbone(n_blocks=0)

    def test_n_blocks_negative_raises(self) -> None:
        """Negative n_blocks raises ValueError."""
        model = self._make_model_with_mock_backbone(num_blocks=4)
        with pytest.raises(ValueError, match="n_blocks must be >= 1"):
            model.partial_unfreeze_backbone(n_blocks=-1)


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
