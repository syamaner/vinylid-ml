"""Embedding model wrappers for album cover recognition.

Provides a unified interface for extracting L2-normalized embeddings from images
using different pretrained models:
- A1/A1-GeM: DINOv2 ViT-S/14 (CLS token or GeM pooling)
- A2: OpenCLIP ViT-B/32
- A4: SSCD ResNet50 (copy detection)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import structlog
import torch
import torch.nn.functional as F  # noqa: N812
from torchvision import transforms

__all__ = [
    "ALL_MODEL_IDS",
    "DINOv2Embedder",
    "EmbeddingModel",
    "OpenCLIPEmbedder",
    "SSCDEmbedder",
    "create_model",
    "gem_pool",
    "get_device",
]

#: Ordered tuple of all supported zero-shot model IDs.
ALL_MODEL_IDS: tuple[str, ...] = (
    "A1-dinov2-cls",
    "A1-dinov2-gem",
    "A1-dinov2-cls-518",
    "A1-dinov2-gem-518",
    "A2-openclip",
    "A4-sscd",
)

logger = structlog.get_logger()


def get_device() -> torch.device:
    """Get the best available torch device (MPS on macOS, else CPU).

    Returns:
        torch.device for MPS if available, otherwise CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ImageNet normalization (used by DINOv2, SSCD, MobileNetV3)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# OpenCLIP normalization (CLIP-specific, slightly different from ImageNet)
OPENCLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENCLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def gem_pool(x: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    """Generalized Mean (GeM) pooling over the token/spatial dimension.

    GeM(x, p) = (mean(x^p))^(1/p)

    With p=1 this is average pooling. Higher p up-weights discriminative
    (larger-valued) features while suppressing background, which typically
    improves retrieval performance.

    Note: Inputs are clamped to ``eps`` before exponentiation. For ViT patch
    tokens (which are layer-normed and may contain negatives), this intentionally
    suppresses negative features — they are near-zero after layer norm and less
    discriminative. This matches the standard approach for applying GeM to
    transformer features in retrieval literature.

    Args:
        x: Input tensor of shape (B, N, D) where N is the token/spatial dimension.
            Values should be non-negative or layer-normed (negatives will be clamped).
        p: GeM exponent. Default 3.0 (standard for retrieval).
        eps: Small value to clamp inputs before exponentiation (avoids NaN for zeros).

    Returns:
        Pooled tensor of shape (B, D).
    """
    return x.clamp(min=eps).pow(p).mean(dim=1).pow(1.0 / p)


class EmbeddingModel(ABC):
    """Abstract base class for all embedding extraction models.

    All subclasses must implement embed() which takes preprocessed image tensors
    and returns L2-normalized embedding vectors on CPU.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique identifier for this model (e.g., 'A1-dinov2-cls')."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding vectors."""

    @property
    @abstractmethod
    def input_size(self) -> int:
        """Expected input image resolution (e.g., 224 or 518)."""

    @abstractmethod
    def get_transforms(self) -> transforms.Compose:
        """Get the preprocessing transforms for this model.

        Returns the correct normalization and resizing for this specific model.
        Callers should apply these transforms to PIL images before calling embed().

        Returns:
            A torchvision Compose transform: Resize -> CenterCrop -> ToTensor -> Normalize.
        """

    @abstractmethod
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalized embeddings from a batch of preprocessed images.

        Args:
            images: Preprocessed tensor of shape (B, 3, H, W). Must be
                normalized using this model's get_transforms(). Different
                models require different normalization (e.g., ImageNet vs CLIP).

        Returns:
            L2-normalized embeddings of shape (B, D) on CPU.
        """


class DINOv2Embedder(EmbeddingModel):
    """DINOv2 ViT-S/14 wrapper supporting CLS token and GeM pooling.

    Loads the model from torch.hub (facebookresearch/dinov2). Supports both
    224x224 and 518x518 input resolutions (input must be divisible by 14).

    Args:
        pooling: Embedding extraction method — 'cls' for [CLS] token,
            'gem' for Generalized Mean pooling over patch tokens.
        gem_p: GeM exponent (only used when pooling='gem'). Default 3.0.
        device: Torch device. If None, auto-detects MPS or CPU.
    """

    def __init__(
        self,
        pooling: Literal["cls", "gem"] = "cls",
        gem_p: float = 3.0,
        device: torch.device | None = None,
        input_size: int = 224,
    ) -> None:
        self._pooling = pooling
        self._gem_p = gem_p
        self._device = device or get_device()
        self._input_size = input_size

        logger.info(
            "loading_model",
            model_id=self.model_id,
            device=str(self._device),
        )
        hub_result = torch.hub.load(  # type: ignore[reportUnknownMemberType]
            "facebookresearch/dinov2", "dinov2_vits14", verbose=False
        )
        if not isinstance(hub_result, torch.nn.Module):
            raise TypeError(f"Expected nn.Module from torch.hub, got {type(hub_result)}")
        self._model: torch.nn.Module = hub_result
        self._model.to(self._device)
        self._model.eval()
        logger.info("model_loaded", model_id=self.model_id)

    @property
    def model_id(self) -> str:
        base = f"A1-dinov2-{self._pooling}"
        if self._input_size != 224:
            return f"{base}-{self._input_size}"
        return base

    @property
    def embedding_dim(self) -> int:
        return 384

    @property
    def input_size(self) -> int:
        return self._input_size

    def get_transforms(self) -> transforms.Compose:
        """ImageNet-normalized transforms for DINOv2."""
        return transforms.Compose(
            [
                transforms.Resize(
                    self._input_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(self._input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Extract embeddings using CLS token or GeM-pooled patch tokens.

        Args:
            images: Tensor of shape (B, 3, H, W), ImageNet-normalized.
                H and W must be divisible by 14.

        Returns:
            L2-normalized embeddings of shape (B, 384) on CPU.
        """
        images = images.to(self._device)

        with torch.inference_mode():
            if self._pooling == "cls":
                # forward() returns the [CLS] token embedding directly
                embeddings = torch.as_tensor(self._model(images))
            else:
                # forward_features() returns dict with patch tokens
                features = self._model.forward_features(images)  # type: ignore[operator]
                patch_tokens = torch.as_tensor(features["x_norm_patchtokens"])
                embeddings = gem_pool(patch_tokens, p=self._gem_p)

        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings.cpu()


class OpenCLIPEmbedder(EmbeddingModel):
    """OpenCLIP ViT-B/32 image embedding wrapper.

    Produces 512-dim image embeddings. Also supports text embeddings
    (not exposed here — for future text-to-image search).

    Args:
        model_name: OpenCLIP model name. Default 'ViT-B-32'.
        pretrained: Pretrained weights tag. Default 'laion2b_s34b_b79k'.
        device: Torch device. If None, auto-detects MPS or CPU.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: torch.device | None = None,
    ) -> None:
        self._device = device or get_device()

        logger.info(
            "loading_model",
            model_id=self.model_id,
            model_name=model_name,
            pretrained=pretrained,
            device=str(self._device),
        )
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(  # type: ignore[reportUnknownMemberType]
            model_name, pretrained=pretrained, device=self._device
        )
        self._model: torch.nn.Module = model
        self._preprocess = preprocess
        self._model.eval()
        logger.info("model_loaded", model_id=self.model_id)

    @property
    def model_id(self) -> str:
        return "A2-openclip"

    @property
    def embedding_dim(self) -> int:
        return 512

    @property
    def input_size(self) -> int:
        return 224

    def get_transforms(self) -> transforms.Compose:
        """CLIP-normalized transforms for OpenCLIP.

        Note: Uses CLIP normalization (slightly different from ImageNet).
        """
        return transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=OPENCLIP_MEAN, std=OPENCLIP_STD),
            ]
        )

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Extract image embeddings via OpenCLIP's image encoder.

        Args:
            images: Tensor of shape (B, 3, H, W), preprocessed.

        Returns:
            L2-normalized embeddings of shape (B, 512) on CPU.
        """
        images = images.to(self._device)

        with torch.inference_mode():
            embeddings = torch.as_tensor(self._model.encode_image(images))  # type: ignore[operator]

        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings.cpu().float()


#: SSCD TorchScript model URLs from Meta's CDN.
#: Shared with ``training.py`` for backbone loading.
SSCD_URLS: dict[str, str] = {
    "sscd_disc_mixup": (
        "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
    ),
    "sscd_disc_large": (
        "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_large.torchscript.pt"
    ),
    "sscd_disc_blur": (
        "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_blur.torchscript.pt"
    ),
}


class SSCDEmbedder(EmbeddingModel):
    """SSCD (Self-Supervised Copy Detection) ResNet50 wrapper.

    Meta's copy detection model trained with heavy augmentations (crops, overlays,
    color jitter) to be invariant to the transformations we see in practice:
    clean digital art → phone photo of physical cover.

    Downloads the pretrained TorchScript model from Meta's CDN (cached locally
    in the torch hub cache directory).

    Args:
        variant: SSCD model variant. Default 'sscd_disc_mixup'.
        device: Torch device. If None, auto-detects MPS or CPU.
    """

    def __init__(
        self,
        variant: str = "sscd_disc_mixup",
        device: torch.device | None = None,
    ) -> None:
        self._device = device or get_device()

        if variant not in SSCD_URLS:
            raise ValueError(
                f"Unknown SSCD variant '{variant}'. Available: {list(SSCD_URLS.keys())}"
            )

        logger.info(
            "loading_model",
            model_id=self.model_id,
            variant=variant,
            device=str(self._device),
        )

        # Download TorchScript model to torch hub cache
        cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
        cache_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{variant}.torchscript.pt"
        cached_path = cache_dir / filename

        if not cached_path.exists():
            logger.info("downloading_sscd", url=SSCD_URLS[variant])
            torch.hub.download_url_to_file(SSCD_URLS[variant], str(cached_path))

        jit_model = torch.jit.load(  # type: ignore[reportUnknownMemberType]
            str(cached_path), map_location=self._device
        )
        if not isinstance(jit_model, torch.nn.Module):
            raise TypeError("Expected nn.Module from torch.jit.load")
        self._model: torch.nn.Module = jit_model
        self._model.eval()
        logger.info("model_loaded", model_id=self.model_id)

    @property
    def model_id(self) -> str:
        return "A4-sscd"

    @property
    def embedding_dim(self) -> int:
        return 512

    @property
    def input_size(self) -> int:
        return 224

    def get_transforms(self) -> transforms.Compose:
        """ImageNet-normalized transforms for SSCD."""
        return transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Extract copy-detection embeddings.

        Args:
            images: Tensor of shape (B, 3, H, W), ImageNet-normalized.

        Returns:
            L2-normalized embeddings of shape (B, 512) on CPU.
        """
        images = images.to(self._device)

        with torch.inference_mode():
            embeddings: torch.Tensor = self._model(images)

        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings.cpu()


def create_model(model_id: str) -> EmbeddingModel:
    """Instantiate an EmbeddingModel from a model ID string.

    Factory function shared by all scripts that need to load a model.
    Supports all IDs in ALL_MODEL_IDS.

    Args:
        model_id: Model identifier string (e.g., ``"A1-dinov2-cls"``).

    Returns:
        A configured, ready-to-use EmbeddingModel instance.

    Raises:
        ValueError: If model_id is not recognised.
    """
    match model_id:
        case "A1-dinov2-cls":
            return DINOv2Embedder(pooling="cls")
        case "A1-dinov2-gem":
            return DINOv2Embedder(pooling="gem")
        case "A1-dinov2-cls-518":
            return DINOv2Embedder(pooling="cls", input_size=518)
        case "A1-dinov2-gem-518":
            return DINOv2Embedder(pooling="gem", input_size=518)
        case "A2-openclip":
            return OpenCLIPEmbedder()
        case "A4-sscd":
            return SSCDEmbedder()
        case _:
            raise ValueError(
                f"Unknown model_id: '{model_id}'. Available: {', '.join(ALL_MODEL_IDS)}"
            )
