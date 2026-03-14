"""VinylID ML — Album cover recognition via embedding-based retrieval."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vinylid_ml.models import (
        DINOv2Embedder as DINOv2Embedder,
    )
    from vinylid_ml.models import (
        EmbeddingModel as EmbeddingModel,
    )
    from vinylid_ml.models import (
        OpenCLIPEmbedder as OpenCLIPEmbedder,
    )
    from vinylid_ml.models import (
        SSCDEmbedder as SSCDEmbedder,
    )
    from vinylid_ml.models import (
        gem_pool as gem_pool,
    )
    from vinylid_ml.models import (
        get_device as get_device,
    )

__all__ = [
    "DINOv2Embedder",
    "EmbeddingModel",
    "OpenCLIPEmbedder",
    "SSCDEmbedder",
    "VinylIDError",
    "gem_pool",
    "get_device",
]

_MODELS_ATTRS = frozenset(__all__) - {"VinylIDError"}


def __getattr__(name: str) -> object:
    if name in _MODELS_ATTRS:
        models = importlib.import_module("vinylid_ml.models")
        return getattr(models, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


class VinylIDError(Exception):
    """Base exception for all VinylID ML errors."""
