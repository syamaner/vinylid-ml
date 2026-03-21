"""Training infrastructure for fine-tuning embedding models.

Provides:
- **FineTuneModel**: Wraps a pretrained backbone with a projection head.
- **MultiViewTransform**: Produces multiple augmented views per image for SupCon.
- **TrainingConfig**: Reproducibility-focused hyperparameter container.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torchvision import transforms

__all__ = [
    "FineTuneModel",
    "MultiViewTransform",
    "TrainingConfig",
]

logger = structlog.get_logger()

#: Mapping of backbone name → feature dimensionality before projection.
BACKBONE_DIMS: dict[str, int] = {
    "dinov2": 384,
    "mobilenet_v3_small": 576,
    "sscd": 512,
}

#: Supported backbone names.
BackboneName = Literal["dinov2", "mobilenet_v3_small", "sscd"]


def _load_dinov2_backbone(device: torch.device) -> nn.Module:
    """Load DINOv2 ViT-S/14 backbone from torch.hub.

    Returns:
        The DINOv2 model (call ``forward()`` to get CLS token).
    """
    logger.info("loading_backbone", backbone="dinov2", device=str(device))
    model = torch.hub.load(  # type: ignore[reportUnknownMemberType]
        "facebookresearch/dinov2", "dinov2_vits14", verbose=False
    )
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module from torch.hub, got {type(model)}")
    model.to(device)
    return model


def _load_mobilenet_backbone(device: torch.device) -> nn.Module:
    """Load MobileNetV3-Small backbone from torchvision.

    Removes the classifier head; ``forward()`` returns 576-dim features.

    Returns:
        MobileNetV3-Small feature extractor.
    """
    logger.info("loading_backbone", backbone="mobilenet_v3_small", device=str(device))
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # Remove the classifier — keep features (AdaptiveAvgPool → flatten → 576-dim)
    model.classifier = nn.Identity()  # type: ignore[assignment]
    model.to(device)
    return model


def _load_sscd_backbone(device: torch.device) -> nn.Module:
    """Load SSCD ResNet50 backbone from Meta's CDN.

    Reuses the URL and caching logic from :data:`vinylid_ml.models.SSCD_URLS`
    to avoid duplication.  SSCD is distributed as a TorchScript model —
    gradient flow works through ``torch.jit`` modules.

    Returns:
        SSCD TorchScript model producing 512-dim embeddings.
    """
    from vinylid_ml.models import SSCD_URLS

    variant = "sscd_disc_mixup"
    logger.info("loading_backbone", backbone="sscd", device=str(device))
    cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_path = cache_dir / f"{variant}.torchscript.pt"

    if not cached_path.exists():
        url = SSCD_URLS[variant]
        logger.info("downloading_sscd", url=url)
        torch.hub.download_url_to_file(url, str(cached_path))

    model = torch.jit.load(str(cached_path), map_location=device)  # type: ignore[reportUnknownMemberType]
    if not isinstance(model, nn.Module):
        raise TypeError("Expected nn.Module from torch.jit.load")
    return model


_BackboneLoader = Callable[[torch.device], nn.Module]

_BACKBONE_LOADERS: dict[str, _BackboneLoader] = {
    "dinov2": _load_dinov2_backbone,
    "mobilenet_v3_small": _load_mobilenet_backbone,
    "sscd": _load_sscd_backbone,
}


class FineTuneModel(nn.Module):
    """Pretrained backbone + projection head for metric learning.

    Wraps a frozen or trainable backbone with a ``Linear → BN1d``
    projection head.  The ``forward()`` method returns **L2-normalised**
    embeddings suitable for all three loss functions in ``losses.py``.

    Args:
        backbone_name: One of ``"dinov2"``, ``"mobilenet_v3_small"``,
            ``"sscd"``.
        projection_dim: Output embedding dimensionality after projection.
        device: Torch device.  ``None`` auto-detects MPS / CPU.
        freeze_backbone: If ``True``, the backbone starts with all
            parameters frozen (``requires_grad=False``).
    """

    def __init__(
        self,
        backbone_name: BackboneName,
        projection_dim: int,
        device: torch.device | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self._backbone_name = backbone_name
        self._projection_dim = projection_dim

        if device is None:
            device = (
                torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
            )
        self._device = device

        if backbone_name not in BACKBONE_DIMS:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. Supported: {sorted(BACKBONE_DIMS.keys())}"
            )

        backbone_dim = BACKBONE_DIMS[backbone_name]

        # Load backbone
        loader = _BACKBONE_LOADERS[backbone_name]
        self.backbone: nn.Module = loader(device)

        # Projection head: Linear → BN1d
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
        ).to(device)

        if freeze_backbone:
            self.freeze_backbone()

        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "finetune_model_created",
            backbone=backbone_name,
            projection_dim=projection_dim,
            total_params=total_params,
            trainable_params=trainable,
            frozen=freeze_backbone,
        )

    @property
    def backbone_name(self) -> str:
        """Name of the backbone architecture."""
        return self._backbone_name

    @property
    def projection_dim(self) -> int:
        """Output embedding dimensionality."""
        return self._projection_dim

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters and set to eval mode.

        Sets ``requires_grad=False`` on all backbone parameters and puts
        the backbone into eval mode so BatchNorm running stats are not
        updated during training.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        logger.info("backbone_frozen", backbone=self._backbone_name)

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters and restore train mode."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.train()
        logger.info("backbone_unfrozen", backbone=self._backbone_name)

    def is_backbone_frozen(self) -> bool:
        """Check whether the backbone parameters are frozen."""
        return not any(p.requires_grad for p in self.backbone.parameters())

    def train(self, mode: bool = True) -> FineTuneModel:
        """Override to keep backbone in eval mode while frozen.

        When the backbone is frozen, calling ``model.train()`` keeps the
        backbone in eval mode so BatchNorm running stats are not updated.
        """
        super().train(mode)
        if mode and self.is_backbone_frozen():
            self.backbone.eval()
        return self

    @property
    def device(self) -> torch.device:
        """The device this model is on."""
        return self._device

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalised embeddings.

        Args:
            images: Preprocessed batch of shape ``(B, 3, H, W)``.

        Returns:
            L2-normalised embeddings of shape ``(B, projection_dim)``.
        """
        images = images.to(self._device)

        # Extract backbone features
        if self._backbone_name == "dinov2":
            # DINOv2 forward() returns the [CLS] token (384-dim)
            features = torch.as_tensor(self.backbone(images))
        elif self._backbone_name == "mobilenet_v3_small":
            # MobileNetV3 features → (B, 576, 1, 1) before classifier;
            # we replaced classifier with Identity so output is (B, 576)
            features = torch.as_tensor(self.backbone(images))
            if features.dim() == 4:
                features = features.flatten(1)
        else:
            # SSCD TorchScript model → 512-dim
            features = torch.as_tensor(self.backbone(images))

        # Project and normalise
        embeddings: torch.Tensor = self.projection(features)
        return F.normalize(embeddings, p=2, dim=-1)


class MultiViewTransform:
    """Wrapper that applies a base transform N times to produce multiple views.

    Used with ``SupConLoss`` which requires multiple augmented views per image.

    Args:
        base_transform: The augmentation transform to apply (e.g.,
            ``get_train_transforms()``).
        n_views: Number of augmented views to produce per image.
    """

    def __init__(self, base_transform: transforms.Compose, n_views: int = 2) -> None:
        if n_views < 1:
            msg = f"n_views must be >= 1, got {n_views}"
            raise ValueError(msg)
        self._base_transform = base_transform
        self._n_views = n_views

    @property
    def n_views(self) -> int:
        """Number of views produced per image."""
        return self._n_views

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Apply the base transform N times.

        Args:
            img: PIL Image to augment.

        Returns:
            Tensor of shape ``(N_views, C, H, W)``.
        """
        views = [
            torch.as_tensor(self._base_transform(img))  # type: ignore[misc]
            for _ in range(self._n_views)
        ]
        return torch.stack(views)


@dataclass
class TrainingConfig:
    """Complete training configuration for reproducibility logging.

    All fields are serialised to ``config.json`` alongside each training run.

    Attributes:
        backbone: Backbone architecture name.
        loss: Loss function name.
        projection_dim: Embedding dimension after projection.
        num_classes: Number of album classes.
        lr: Base learning rate.
        weight_decay: AdamW weight decay.
        epochs: Total training epochs.
        batch_size: Training batch size.
        seed: Random seed for reproducibility.
        subset_albums: If set, train on only this many albums.
        freeze_epochs: Number of initial epochs with backbone frozen.
        margin: ArcFace/ProxyAnchor margin parameter.
        scale: ArcFace scale parameter.
        alpha: ProxyAnchor alpha parameter.
        temperature: SupCon temperature parameter.
        n_views: Number of augmented views for SupCon.
        input_size: Model input resolution.
        device: Torch device string.
        git_sha: Git commit SHA at training time.
        manifest_hash: Hash of the dataset manifest for provenance.
        extra: Any additional metadata.
    """

    backbone: str = "dinov2"
    loss: str = "arcface"
    projection_dim: int = 384
    num_classes: int = 0
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    batch_size: int = 32
    seed: int = 42
    subset_albums: int | None = None
    freeze_epochs: int = 0
    margin: float = 0.5
    scale: float = 64.0
    alpha: float = 32.0
    temperature: float = 0.07
    n_views: int = 2
    input_size: int = 224
    device: str = "cpu"
    git_sha: str | None = None
    manifest_hash: str | None = None
    extra: dict[str, object] = field(default_factory=lambda: dict[str, object]())

    def to_dict(self) -> dict[str, object]:
        """Convert to a JSON-serialisable dictionary."""
        return asdict(self)

    def save(self, path: Path) -> None:
        """Write config to a JSON file.

        Args:
            path: Output file path.
        """
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> TrainingConfig:
        """Load config from a JSON file.

        Args:
            path: Input file path.

        Returns:
            Restored TrainingConfig instance.
        """
        with path.open() as f:
            data = json.load(f)
        # Filter to known fields only
        known_fields = {fld.name for fld in cls.__dataclass_fields__.values()}  # type: ignore[reportUnknownMemberType]
        filtered = {k: v for k, v in data.items() if k in known_fields}  # type: ignore[reportUnknownVariableType]
        return cls(**filtered)  # type: ignore[reportUnknownArgumentType]
