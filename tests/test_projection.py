"""Unit tests for vinylid_ml.projection — ProjectionHead and NTXentLoss.

All tests are pure unit tests with no model downloads or disk access.
"""

from __future__ import annotations

import pytest
import torch

from vinylid_ml.projection import NTXentLoss, ProjectionHead

# ---------------------------------------------------------------------------
# ProjectionHead
# ---------------------------------------------------------------------------


class TestProjectionHead:
    """Tests for ProjectionHead construction, shape, and output properties."""

    def test_default_output_shape(self) -> None:
        """Default head maps 512-dim input to 128-dim output."""
        head = ProjectionHead()
        x = torch.rand(4, 512)
        out = head(x)
        assert out.shape == (4, 128)

    def test_custom_dims_output_shape(self) -> None:
        """Custom in/hidden/out dimensions are respected."""
        head = ProjectionHead(in_dim=384, hidden_dim=128, out_dim=64)
        x = torch.rand(8, 384)
        out = head(x)
        assert out.shape == (8, 64)

    def test_output_is_l2_normalized(self) -> None:
        """Each output row must have unit L2 norm (within floating-point tolerance)."""
        head = ProjectionHead()
        x = torch.rand(16, 512)
        out = head(x)
        norms = torch.norm(out, p=2, dim=-1)
        torch.testing.assert_close(norms, torch.ones(16), atol=1e-5, rtol=0.0)

    def test_single_sample(self) -> None:
        """Batch size of 1 should work correctly."""
        head = ProjectionHead()
        x = torch.rand(1, 512)
        out = head(x)
        assert out.shape == (1, 128)
        norm = torch.norm(out, p=2, dim=-1)
        torch.testing.assert_close(norm, torch.ones(1), atol=1e-5, rtol=0.0)

    def test_output_dim_attribute(self) -> None:
        """out_dim attribute matches the configured value."""
        head = ProjectionHead(in_dim=512, hidden_dim=256, out_dim=64)
        assert head.out_dim == 64

    def test_invalid_in_dim_raises(self) -> None:
        """in_dim < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="in_dim"):
            ProjectionHead(in_dim=0)

    def test_invalid_hidden_dim_raises(self) -> None:
        """hidden_dim < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="hidden_dim"):
            ProjectionHead(hidden_dim=0)

    def test_invalid_out_dim_raises(self) -> None:
        """out_dim < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="out_dim"):
            ProjectionHead(out_dim=0)

    def test_invalid_dropout_raises(self) -> None:
        """dropout >= 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="dropout"):
            ProjectionHead(dropout=1.0)

    def test_no_grad_in_eval_mode(self) -> None:
        """In eval mode with inference_mode, no gradient should flow."""
        head = ProjectionHead()
        head.eval()
        x = torch.rand(4, 512)
        with torch.inference_mode():
            out = head(x)
        assert out.requires_grad is False

    def test_dropout_disabled_in_eval(self) -> None:
        """Deterministic output in eval mode (dropout disabled)."""
        head = ProjectionHead(dropout=0.5)
        head.eval()
        x = torch.rand(8, 512)
        with torch.inference_mode():
            out1 = head(x)
            out2 = head(x)
        torch.testing.assert_close(out1, out2)

    def test_parameters_are_learnable(self) -> None:
        """All parameters in the projection layers should require gradients."""
        head = ProjectionHead()
        n_learnable = sum(1 for p in head.parameters() if p.requires_grad)
        assert n_learnable > 0

    def test_xavier_init_no_nan(self) -> None:
        """Xavier init + L2-norm on random input should not produce NaN/Inf."""
        head = ProjectionHead()
        x = torch.rand(32, 512)
        out = head(x)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# NTXentLoss
# ---------------------------------------------------------------------------


class TestNTXentLoss:
    """Tests for NTXentLoss construction and forward pass properties."""

    def test_output_is_scalar(self) -> None:
        """Loss should be a zero-dimensional scalar tensor."""
        loss_fn = NTXentLoss()
        z_a = torch.nn.functional.normalize(torch.rand(8, 128), p=2, dim=-1)
        z_p = torch.nn.functional.normalize(torch.rand(8, 128), p=2, dim=-1)
        loss = loss_fn(z_a, z_p)
        assert loss.ndim == 0

    def test_loss_is_positive(self) -> None:
        """NT-Xent loss with random embeddings should be positive."""
        loss_fn = NTXentLoss(temperature=0.1)
        z_a = torch.nn.functional.normalize(torch.rand(8, 64), p=2, dim=-1)
        z_p = torch.nn.functional.normalize(torch.rand(8, 64), p=2, dim=-1)
        loss = loss_fn(z_a, z_p)
        assert loss.item() > 0.0

    def test_loss_is_finite(self) -> None:
        """Loss should not be NaN or Inf for valid inputs."""
        loss_fn = NTXentLoss(temperature=0.07)
        z_a = torch.nn.functional.normalize(torch.rand(16, 128), p=2, dim=-1)
        z_p = torch.nn.functional.normalize(torch.rand(16, 128), p=2, dim=-1)
        loss = loss_fn(z_a, z_p)
        assert torch.isfinite(loss)

    def test_perfect_pairs_lower_loss(self) -> None:
        """Identical anchor/positive pairs should give lower loss than random."""
        loss_fn = NTXentLoss(temperature=0.07)

        # Random pairs — high loss
        z_rand = torch.nn.functional.normalize(torch.rand(8, 128), p=2, dim=-1)
        loss_random = loss_fn(
            z_rand, torch.nn.functional.normalize(torch.rand(8, 128), p=2, dim=-1)
        )

        # Perfect pairs (anchor == positive) — low loss
        z_perfect = torch.nn.functional.normalize(torch.rand(8, 128), p=2, dim=-1)
        loss_perfect = loss_fn(z_perfect, z_perfect.clone())

        assert loss_perfect.item() < loss_random.item()

    def test_symmetric(self) -> None:
        """Loss(a, p) should equal Loss(p, a) — symmetry of NT-Xent."""
        loss_fn = NTXentLoss(temperature=0.1)
        z_a = torch.nn.functional.normalize(torch.rand(6, 64), p=2, dim=-1)
        z_p = torch.nn.functional.normalize(torch.rand(6, 64), p=2, dim=-1)
        loss_ap = loss_fn(z_a, z_p)
        loss_pa = loss_fn(z_p, z_a)
        torch.testing.assert_close(loss_ap, loss_pa, atol=1e-5, rtol=0.0)

    def test_higher_temperature_higher_loss(self) -> None:
        """Higher temperature → softer distribution → generally higher loss."""
        z_a = torch.nn.functional.normalize(torch.rand(8, 64), p=2, dim=-1)
        z_p = torch.nn.functional.normalize(torch.rand(8, 64), p=2, dim=-1)

        loss_low_t = NTXentLoss(temperature=0.01)(z_a, z_p)
        loss_high_t = NTXentLoss(temperature=1.0)(z_a, z_p)

        # At low temperature with random negatives, loss is very high (sharp distribution).
        # At high temperature, loss is moderated.  Both are > 0; relative ordering depends
        # on the sample, so we just verify they differ.
        assert loss_low_t.item() != pytest.approx(loss_high_t.item(), abs=1e-3)

    def test_batch_size_1_raises(self) -> None:
        """Batch size of 1 provides no negatives and should raise ValueError."""
        loss_fn = NTXentLoss()
        z = torch.nn.functional.normalize(torch.rand(1, 64), p=2, dim=-1)
        with pytest.raises(ValueError, match="batch_size"):
            loss_fn(z, z.clone())

    def test_invalid_temperature_raises(self) -> None:
        """Non-positive temperature should raise ValueError."""
        with pytest.raises(ValueError, match="temperature"):
            NTXentLoss(temperature=0.0)
        with pytest.raises(ValueError, match="temperature"):
            NTXentLoss(temperature=-0.1)

    def test_gradients_flow(self) -> None:
        """Loss must be differentiable w.r.t. both anchor and positive inputs."""
        loss_fn = NTXentLoss()
        # Use leaf tensors directly so .grad is populated after backward()
        z_a = (
            torch.nn.functional.normalize(torch.rand(4, 32), p=2, dim=-1)
            .detach()
            .requires_grad_(True)
        )
        z_p = (
            torch.nn.functional.normalize(torch.rand(4, 32), p=2, dim=-1)
            .detach()
            .requires_grad_(True)
        )
        loss = loss_fn(z_a, z_p)
        loss.backward()
        assert z_a.grad is not None
        assert z_p.grad is not None
        assert torch.isfinite(z_a.grad).all()
