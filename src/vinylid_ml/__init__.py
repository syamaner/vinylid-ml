"""VinylID ML — Album cover recognition via embedding-based retrieval."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vinylid_ml.gallery import (
        EmbeddingResult as EmbeddingResult,
    )
    from vinylid_ml.gallery import (
        GalleryImageDataset as GalleryImageDataset,
    )
    from vinylid_ml.gallery import (
        embed_dataset as embed_dataset,
    )
    from vinylid_ml.gallery import (
        load_embeddings as load_embeddings,
    )
    from vinylid_ml.gallery import (
        save_embeddings as save_embeddings,
    )
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
    "EmbeddingResult",
    "GalleryImageDataset",
    "OpenCLIPEmbedder",
    "SSCDEmbedder",
    "VinylIDError",
    "embed_dataset",
    "gem_pool",
    "get_device",
    "load_embeddings",
    "save_embeddings",
]

_LAZY_MODULES: dict[str, str] = {
    "DINOv2Embedder": "vinylid_ml.models",
    "EmbeddingModel": "vinylid_ml.models",
    "OpenCLIPEmbedder": "vinylid_ml.models",
    "SSCDEmbedder": "vinylid_ml.models",
    "gem_pool": "vinylid_ml.models",
    "get_device": "vinylid_ml.models",
    "EmbeddingResult": "vinylid_ml.gallery",
    "GalleryImageDataset": "vinylid_ml.gallery",
    "embed_dataset": "vinylid_ml.gallery",
    "load_embeddings": "vinylid_ml.gallery",
    "save_embeddings": "vinylid_ml.gallery",
}


def __getattr__(name: str) -> object:
    module_name = _LAZY_MODULES.get(name)
    if module_name is not None:
        module = importlib.import_module(module_name)
        return getattr(module, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


class VinylIDError(Exception):
    """Base exception for all VinylID ML errors."""
