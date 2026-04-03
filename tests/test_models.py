"""Tests for vinylid_ml.models — embedding model wrappers.

Organized into:
- Unit tests: gem_pool math, model properties (no model download)
- Integration tests: actual embedding extraction (downloads models on first run)
"""

# pyright: reportPrivateUsage=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import pytest
import torch

from vinylid_ml.models import (
    DINOv2Embedder,
    EmbeddingModel,
    OpenCLIPEmbedder,
    SSCDEmbedder,
    gem_pool,
    get_device,
)

# ============================================================
# Unit tests — no model download required
# ============================================================


class TestGemPool:
    """Tests for the standalone GeM pooling function."""

    def test_output_shape(self) -> None:
        """GeM pool over token dim produces (B, D) from (B, N, D)."""
        x = torch.rand(2, 256, 384)  # batch=2, 256 patches, 384-dim
        result = gem_pool(x, p=3.0)
        assert result.shape == (2, 384)

    def test_p1_equals_mean_pool(self) -> None:
        """GeM with p=1.0 is equivalent to average pooling."""
        x = torch.rand(4, 16, 64)
        gem_result = gem_pool(x, p=1.0)
        mean_result = x.mean(dim=1)
        torch.testing.assert_close(gem_result, mean_result, atol=1e-5, rtol=1e-5)

    def test_higher_p_upweights_large_values(self) -> None:
        """Higher p should produce values closer to max than to mean."""
        x = torch.tensor([[[0.1, 0.1], [0.1, 0.9]]])  # (1, 2, 2)
        low_p = gem_pool(x, p=1.0)
        high_p = gem_pool(x, p=5.0)
        # For the second dim: values are [0.1, 0.9], high_p result should be > low_p
        assert high_p[0, 1] > low_p[0, 1]

    def test_single_token_is_identity(self) -> None:
        """GeM of a single token is the token itself (any p)."""
        x = torch.rand(1, 1, 128)
        result = gem_pool(x, p=3.0)
        torch.testing.assert_close(result, x.squeeze(1), atol=1e-5, rtol=1e-5)

    def test_handles_near_zero_values(self) -> None:
        """GeM should not produce NaN/Inf for very small values."""
        x = torch.full((1, 4, 8), 1e-8)
        result = gem_pool(x, p=3.0)
        assert torch.isfinite(result).all()


class TestGetDevice:
    """Tests for the get_device() helper."""

    def test_returns_torch_device(self) -> None:
        """get_device() returns a valid torch.device."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_prefers_cuda_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CUDA should take precedence over MPS and CPU when available."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

        device = get_device()

        assert device.type == "cuda"

    def test_falls_back_to_mps_when_cuda_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MPS should be used when CUDA is unavailable and MPS is available."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

        device = get_device()

        assert device.type == "mps"

    def test_falls_back_to_cpu_when_no_accelerator_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CPU should be used when neither CUDA nor MPS is available."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

        device = get_device()

        assert device.type == "cpu"


class TestModelProperties:
    """Test model wrapper properties without loading actual models."""

    def test_dinov2_cls_model_id(self) -> None:
        """DINOv2 CLS wrapper at default 224px reports correct model ID."""
        model = DINOv2Embedder.__new__(DINOv2Embedder)
        model._pooling = "cls"
        model._input_size = 224
        assert model.model_id == "A1-dinov2-cls"

    def test_dinov2_gem_model_id(self) -> None:
        """DINOv2 GeM wrapper at default 224px reports correct model ID."""
        model = DINOv2Embedder.__new__(DINOv2Embedder)
        model._pooling = "gem"
        model._input_size = 224
        assert model.model_id == "A1-dinov2-gem"

    def test_dinov2_518_model_id(self) -> None:
        """DINOv2 CLS at 518px includes resolution in model ID."""
        model = DINOv2Embedder.__new__(DINOv2Embedder)
        model._pooling = "cls"
        model._input_size = 518
        assert model.model_id == "A1-dinov2-cls-518"

    def test_dinov2_embedding_dim(self) -> None:
        """DINOv2 wrapper reports 384-dim embeddings."""
        model = DINOv2Embedder.__new__(DINOv2Embedder)
        assert model.embedding_dim == 384

    def test_openclip_model_id(self) -> None:
        """OpenCLIP wrapper reports correct model ID."""
        model = OpenCLIPEmbedder.__new__(OpenCLIPEmbedder)
        assert model.model_id == "A2-openclip"

    def test_openclip_embedding_dim(self) -> None:
        """OpenCLIP wrapper reports 512-dim embeddings."""
        model = OpenCLIPEmbedder.__new__(OpenCLIPEmbedder)
        assert model.embedding_dim == 512

    def test_sscd_model_id(self) -> None:
        """SSCD wrapper reports correct model ID."""
        model = SSCDEmbedder.__new__(SSCDEmbedder)
        assert model.model_id == "A4-sscd"

    def test_sscd_embedding_dim(self) -> None:
        """SSCD wrapper reports 512-dim embeddings."""
        model = SSCDEmbedder.__new__(SSCDEmbedder)
        assert model.embedding_dim == 512

    def test_all_are_embedding_models(self) -> None:
        """All wrappers are subclasses of EmbeddingModel."""
        assert issubclass(DINOv2Embedder, EmbeddingModel)
        assert issubclass(OpenCLIPEmbedder, EmbeddingModel)
        assert issubclass(SSCDEmbedder, EmbeddingModel)


# ============================================================
# Integration tests — require model download (cached by torch.hub)
# ============================================================


def _assert_l2_normalized(embeddings: torch.Tensor, atol: float = 1e-4) -> None:
    """Assert that all embedding vectors have unit L2 norm."""
    norms = torch.norm(embeddings, p=2, dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=atol, rtol=0.0)


@pytest.fixture(scope="session")
def dinov2_cls() -> DINOv2Embedder:
    """Session-scoped DINOv2 CLS embedder (downloaded once, cached by torch.hub)."""
    return DINOv2Embedder(pooling="cls")


@pytest.fixture(scope="session")
def dinov2_gem() -> DINOv2Embedder:
    """Session-scoped DINOv2 GeM embedder (same backbone, different pooling)."""
    return DINOv2Embedder(pooling="gem")


@pytest.fixture(scope="session")
def openclip_model() -> OpenCLIPEmbedder:
    """Session-scoped OpenCLIP embedder."""
    return OpenCLIPEmbedder()


@pytest.fixture(scope="session")
def sscd_model() -> SSCDEmbedder:
    """Session-scoped SSCD embedder."""
    return SSCDEmbedder()


@pytest.fixture
def random_batch_224() -> torch.Tensor:
    """Random unnormalized batch (4 x 224x224) for smoke-testing embed() shape/dtype."""
    return torch.rand(4, 3, 224, 224)


@pytest.fixture
def random_batch_518() -> torch.Tensor:
    """Random unnormalized batch (2 x 518x518) for DINOv2 native resolution smoke test."""
    return torch.rand(2, 3, 518, 518)


@pytest.mark.integration
class TestDINOv2EmbedderIntegration:
    """Integration tests for DINOv2 embedding extraction."""

    def test_cls_output_shape(
        self, dinov2_cls: DINOv2Embedder, random_batch_224: torch.Tensor
    ) -> None:
        """CLS embedding produces (B, 384) output."""
        embeddings = dinov2_cls.embed(random_batch_224)
        assert embeddings.shape == (4, 384)

    def test_cls_output_is_l2_normalized(
        self, dinov2_cls: DINOv2Embedder, random_batch_224: torch.Tensor
    ) -> None:
        """CLS embeddings are L2-normalized to unit length."""
        embeddings = dinov2_cls.embed(random_batch_224)
        _assert_l2_normalized(embeddings)

    def test_gem_output_shape(
        self, dinov2_gem: DINOv2Embedder, random_batch_224: torch.Tensor
    ) -> None:
        """GeM embedding produces (B, 384) output."""
        embeddings = dinov2_gem.embed(random_batch_224)
        assert embeddings.shape == (4, 384)

    def test_gem_output_is_l2_normalized(
        self, dinov2_gem: DINOv2Embedder, random_batch_224: torch.Tensor
    ) -> None:
        """GeM embeddings are L2-normalized."""
        embeddings = dinov2_gem.embed(random_batch_224)
        _assert_l2_normalized(embeddings)

    def test_cls_and_gem_produce_different_embeddings(
        self,
        dinov2_cls: DINOv2Embedder,
        dinov2_gem: DINOv2Embedder,
        random_batch_224: torch.Tensor,
    ) -> None:
        """CLS and GeM pooling should produce different embeddings."""
        cls_emb = dinov2_cls.embed(random_batch_224)
        gem_emb = dinov2_gem.embed(random_batch_224)
        assert not torch.allclose(cls_emb, gem_emb, atol=1e-3)

    def test_518_resolution(
        self, dinov2_cls: DINOv2Embedder, random_batch_518: torch.Tensor
    ) -> None:
        """DINOv2 supports 518x518 input (native training resolution)."""
        embeddings = dinov2_cls.embed(random_batch_518)
        assert embeddings.shape == (2, 384)
        _assert_l2_normalized(embeddings)

    def test_single_image(self, dinov2_cls: DINOv2Embedder) -> None:
        """Single image (batch size 1) works correctly."""
        single = torch.rand(1, 3, 224, 224)
        embeddings = dinov2_cls.embed(single)
        assert embeddings.shape == (1, 384)
        _assert_l2_normalized(embeddings)

    def test_embeddings_on_cpu(self, dinov2_cls: DINOv2Embedder) -> None:
        """Output embeddings are always returned on CPU."""
        images = torch.rand(2, 3, 224, 224)
        embeddings = dinov2_cls.embed(images)
        assert embeddings.device == torch.device("cpu")

    def test_deterministic_output(self, dinov2_cls: DINOv2Embedder) -> None:
        """Same input produces identical embeddings (inference mode)."""
        images = torch.rand(2, 3, 224, 224)
        emb1 = dinov2_cls.embed(images)
        emb2 = dinov2_cls.embed(images)
        torch.testing.assert_close(emb1, emb2)


@pytest.mark.integration
class TestOpenCLIPEmbedderIntegration:
    """Integration tests for OpenCLIP embedding extraction."""

    def test_output_shape(
        self, openclip_model: OpenCLIPEmbedder, random_batch_224: torch.Tensor
    ) -> None:
        """OpenCLIP produces (B, 512) output."""
        embeddings = openclip_model.embed(random_batch_224)
        assert embeddings.shape == (4, 512)

    def test_output_is_l2_normalized(
        self, openclip_model: OpenCLIPEmbedder, random_batch_224: torch.Tensor
    ) -> None:
        """OpenCLIP embeddings are L2-normalized."""
        embeddings = openclip_model.embed(random_batch_224)
        _assert_l2_normalized(embeddings)

    def test_single_image(self, openclip_model: OpenCLIPEmbedder) -> None:
        """Single image works correctly."""
        single = torch.rand(1, 3, 224, 224)
        embeddings = openclip_model.embed(single)
        assert embeddings.shape == (1, 512)

    def test_embeddings_on_cpu(self, openclip_model: OpenCLIPEmbedder) -> None:
        """Output embeddings are on CPU."""
        images = torch.rand(2, 3, 224, 224)
        embeddings = openclip_model.embed(images)
        assert embeddings.device == torch.device("cpu")


@pytest.mark.integration
class TestSSCDEmbedderIntegration:
    """Integration tests for SSCD embedding extraction."""

    def test_output_shape(self, sscd_model: SSCDEmbedder, random_batch_224: torch.Tensor) -> None:
        """SSCD produces (B, 512) output."""
        embeddings = sscd_model.embed(random_batch_224)
        assert embeddings.shape == (4, 512)

    def test_output_is_l2_normalized(
        self, sscd_model: SSCDEmbedder, random_batch_224: torch.Tensor
    ) -> None:
        """SSCD embeddings are L2-normalized."""
        embeddings = sscd_model.embed(random_batch_224)
        _assert_l2_normalized(embeddings)

    def test_single_image(self, sscd_model: SSCDEmbedder) -> None:
        """Single image works correctly."""
        single = torch.rand(1, 3, 224, 224)
        embeddings = sscd_model.embed(single)
        assert embeddings.shape == (1, 512)

    def test_embeddings_on_cpu(self, sscd_model: SSCDEmbedder) -> None:
        """Output embeddings are on CPU."""
        images = torch.rand(2, 3, 224, 224)
        embeddings = sscd_model.embed(images)
        assert embeddings.device == torch.device("cpu")
