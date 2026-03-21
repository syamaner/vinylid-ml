"""Tests for vinylid_ml.losses — metric learning loss functions.

Unit tests only — no model downloads or GPU required.
"""

from __future__ import annotations

import pytest
import torch

from vinylid_ml.losses import ArcFaceLoss, ProxyAnchorLoss, SupConLoss

# ============================================================
# Helpers
# ============================================================


def _random_l2_normalized(batch: int, dim: int) -> torch.Tensor:
    """Create random L2-normalized embeddings with gradients enabled."""
    x = torch.randn(batch, dim, requires_grad=True)
    with torch.no_grad():
        x_normed = torch.nn.functional.normalize(x, p=2, dim=-1)
    return x_normed.detach().requires_grad_(True)


# ============================================================
# ArcFaceLoss
# ============================================================


class TestArcFaceLoss:
    """Tests for ArcFace angular margin classification loss."""

    def test_output_is_scalar(self) -> None:
        """Loss returns a 0-d tensor."""
        loss_fn = ArcFaceLoss(embedding_dim=64, num_classes=10)
        emb = _random_l2_normalized(8, 64)
        labels = torch.randint(0, 10, (8,))
        loss = loss_fn(emb, labels)
        assert loss.dim() == 0

    def test_output_is_positive(self) -> None:
        """Cross-entropy-based loss should be non-negative."""
        loss_fn = ArcFaceLoss(embedding_dim=64, num_classes=10)
        emb = _random_l2_normalized(8, 64)
        labels = torch.randint(0, 10, (8,))
        loss = loss_fn(emb, labels)
        assert loss.item() > 0

    def test_gradient_flows(self) -> None:
        """Gradients propagate to both embeddings and loss weights."""
        loss_fn = ArcFaceLoss(embedding_dim=32, num_classes=5)
        emb = _random_l2_normalized(4, 32)
        labels = torch.randint(0, 5, (4,))
        loss = loss_fn(emb, labels)
        loss.backward()
        assert emb.grad is not None
        assert loss_fn.weight.grad is not None

    def test_margin_reduces_target_logit(self) -> None:
        """Higher margin produces higher loss (target logit more suppressed)."""
        emb = _random_l2_normalized(4, 32)
        labels = torch.randint(0, 5, (4,))

        loss_low_margin = ArcFaceLoss(embedding_dim=32, num_classes=5, margin=0.1, scale=1.0)
        loss_high_margin = ArcFaceLoss(embedding_dim=32, num_classes=5, margin=0.5, scale=1.0)

        # Copy weights so they're identical
        with torch.no_grad():
            loss_high_margin.weight.copy_(loss_low_margin.weight)

        l_low = loss_low_margin(emb, labels)
        l_high = loss_high_margin(emb, labels)
        # Higher margin increases the loss (target logit is more suppressed)
        assert l_high.item() > l_low.item()

    def test_scale_amplifies_loss(self) -> None:
        """Higher scale should produce larger loss magnitude."""
        emb = _random_l2_normalized(4, 32)
        labels = torch.randint(0, 5, (4,))

        loss_low = ArcFaceLoss(embedding_dim=32, num_classes=5, margin=0.1, scale=1.0)
        loss_high = ArcFaceLoss(embedding_dim=32, num_classes=5, margin=0.1, scale=64.0)

        with torch.no_grad():
            loss_high.weight.copy_(loss_low.weight)

        l_low = loss_low(emb, labels)
        l_high = loss_high(emb, labels)
        assert l_high.item() > l_low.item()

    def test_num_classes_property(self) -> None:
        """num_classes property reflects constructor arg."""
        loss_fn = ArcFaceLoss(embedding_dim=64, num_classes=100)
        assert loss_fn.num_classes == 100

    def test_weight_shape(self) -> None:
        """Weight matrix has shape (num_classes, embedding_dim)."""
        loss_fn = ArcFaceLoss(embedding_dim=64, num_classes=100)
        assert loss_fn.weight.shape == (100, 64)

    def test_single_sample_batch(self) -> None:
        """Works with batch size 1."""
        loss_fn = ArcFaceLoss(embedding_dim=32, num_classes=5)
        emb = _random_l2_normalized(1, 32)
        labels = torch.tensor([2])
        loss = loss_fn(emb, labels)
        assert torch.isfinite(loss)

    def test_invalid_margin_zero(self) -> None:
        """margin=0 raises ValueError."""
        with pytest.raises(ValueError, match="margin must be in"):
            ArcFaceLoss(embedding_dim=32, num_classes=5, margin=0.0)

    def test_invalid_margin_too_large(self) -> None:
        """margin >= pi raises ValueError."""
        import math

        with pytest.raises(ValueError, match="margin must be in"):
            ArcFaceLoss(embedding_dim=32, num_classes=5, margin=math.pi)

    def test_invalid_scale_zero(self) -> None:
        """scale=0 raises ValueError."""
        with pytest.raises(ValueError, match="scale must be"):
            ArcFaceLoss(embedding_dim=32, num_classes=5, scale=0.0)


# ============================================================
# ProxyAnchorLoss
# ============================================================


class TestProxyAnchorLoss:
    """Tests for Proxy-Anchor metric learning loss."""

    def test_output_is_scalar(self) -> None:
        """Loss returns a 0-d tensor."""
        loss_fn = ProxyAnchorLoss(embedding_dim=64, num_classes=10)
        emb = _random_l2_normalized(8, 64)
        labels = torch.randint(0, 10, (8,))
        loss = loss_fn(emb, labels)
        assert loss.dim() == 0

    def test_output_is_positive(self) -> None:
        """Loss should be positive (log(1 + exp(...)) > 0)."""
        loss_fn = ProxyAnchorLoss(embedding_dim=64, num_classes=10)
        emb = _random_l2_normalized(8, 64)
        labels = torch.randint(0, 10, (8,))
        loss = loss_fn(emb, labels)
        assert loss.item() > 0

    def test_gradient_flows(self) -> None:
        """Gradients propagate to both embeddings and proxy parameters."""
        loss_fn = ProxyAnchorLoss(embedding_dim=32, num_classes=5)
        emb = _random_l2_normalized(4, 32)
        labels = torch.randint(0, 5, (4,))
        loss = loss_fn(emb, labels)
        loss.backward()
        assert emb.grad is not None
        assert loss_fn.proxies.grad is not None

    def test_handles_subset_of_classes_in_batch(self) -> None:
        """Works when only a subset of classes appear in the batch."""
        loss_fn = ProxyAnchorLoss(embedding_dim=32, num_classes=100)
        emb = _random_l2_normalized(4, 32)
        # Only classes 0 and 1 present out of 100
        labels = torch.tensor([0, 0, 1, 1])
        loss = loss_fn(emb, labels)
        assert torch.isfinite(loss)

    def test_num_classes_property(self) -> None:
        """num_classes property reflects constructor arg."""
        loss_fn = ProxyAnchorLoss(embedding_dim=64, num_classes=50)
        assert loss_fn.num_classes == 50

    def test_proxy_shape(self) -> None:
        """Proxy matrix has shape (num_classes, embedding_dim)."""
        loss_fn = ProxyAnchorLoss(embedding_dim=64, num_classes=50)
        assert loss_fn.proxies.shape == (50, 64)

    def test_single_sample_batch(self) -> None:
        """Works with batch size 1."""
        loss_fn = ProxyAnchorLoss(embedding_dim=32, num_classes=5)
        emb = _random_l2_normalized(1, 32)
        labels = torch.tensor([3])
        loss = loss_fn(emb, labels)
        assert torch.isfinite(loss)

    def test_higher_alpha_increases_loss(self) -> None:
        """Higher alpha amplifies the exponential terms, increasing loss."""
        emb = _random_l2_normalized(8, 32)
        labels = torch.randint(0, 5, (8,))

        loss_low = ProxyAnchorLoss(embedding_dim=32, num_classes=5, alpha=1.0)
        loss_high = ProxyAnchorLoss(embedding_dim=32, num_classes=5, alpha=32.0)

        with torch.no_grad():
            loss_high.proxies.copy_(loss_low.proxies)

        l_low = loss_low(emb, labels)
        l_high = loss_high(emb, labels)
        assert l_high.item() > l_low.item()


# ============================================================
# SupConLoss
# ============================================================


class TestSupConLoss:
    """Tests for Supervised Contrastive Loss."""

    def test_output_is_scalar_2d(self) -> None:
        """Loss returns a 0-d tensor for 2-D input."""
        loss_fn = SupConLoss()
        features = _random_l2_normalized(8, 64)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = loss_fn(features, labels)
        assert loss.dim() == 0

    def test_output_is_scalar_3d(self) -> None:
        """Loss returns a 0-d tensor for 3-D (multi-view) input."""
        loss_fn = SupConLoss()
        # 4 samples, 2 views each, 64-dim
        features = torch.randn(4, 2, 64)
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
        labels = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(features, labels)
        assert loss.dim() == 0

    def test_output_is_positive(self) -> None:
        """SupCon loss should be positive for random embeddings."""
        loss_fn = SupConLoss()
        features = _random_l2_normalized(8, 64)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = loss_fn(features, labels)
        assert loss.item() > 0

    def test_gradient_flows(self) -> None:
        """Gradients propagate to input features."""
        loss_fn = SupConLoss()
        features = _random_l2_normalized(8, 64)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = loss_fn(features, labels)
        loss.backward()
        assert features.grad is not None

    def test_aligned_same_class_has_lower_loss(self) -> None:
        """Loss is lower when same-class features are identical vs random."""
        loss_fn = SupConLoss(temperature=0.5)

        # Aligned: same-class features are identical
        base = torch.randn(4, 64)
        base = torch.nn.functional.normalize(base, dim=-1)
        aligned = torch.stack([base, base], dim=1)  # (4, 2, 64)
        labels = torch.arange(4)

        # Random: all features are independent
        random_feats = torch.randn(4, 2, 64)
        random_feats = torch.nn.functional.normalize(random_feats, dim=-1)

        l_aligned = loss_fn(aligned, labels)
        l_random = loss_fn(random_feats, labels)
        assert l_aligned.item() < l_random.item()

    def test_three_views(self) -> None:
        """Works with 3 augmented views per sample."""
        loss_fn = SupConLoss()
        features = torch.randn(4, 3, 64)
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
        labels = torch.tensor([0, 1, 0, 1])
        loss = loss_fn(features, labels)
        assert torch.isfinite(loss)

    def test_all_same_label_does_not_crash(self) -> None:
        """Gracefully handles a batch where all samples share one label."""
        loss_fn = SupConLoss()
        features = _random_l2_normalized(6, 32)
        labels = torch.zeros(6, dtype=torch.long)
        loss = loss_fn(features, labels)
        assert torch.isfinite(loss)

    def test_no_positives_for_some_samples(self) -> None:
        """Handles samples that have no positive pairs (unique labels)."""
        loss_fn = SupConLoss()
        features = _random_l2_normalized(4, 32)
        # Each sample has a unique label — no positives
        labels = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(features, labels)
        assert torch.isfinite(loss)

    def test_invalid_temperature_zero(self) -> None:
        """temperature=0 raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be"):
            SupConLoss(temperature=0.0)

    def test_invalid_temperature_negative(self) -> None:
        """Negative temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature must be"):
            SupConLoss(temperature=-0.1)
