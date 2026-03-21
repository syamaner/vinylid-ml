"""VinylID ML — Album cover recognition via embedding-based retrieval."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vinylid_ml.apple_featureprint import (
        FEATUREPRINT_MODEL_ID as FEATUREPRINT_MODEL_ID,
    )
    from vinylid_ml.apple_featureprint import (
        embed_images as embed_images,
    )
    from vinylid_ml.apple_featureprint import (
        extract_feature_vector as extract_feature_vector,
    )
    from vinylid_ml.apple_featureprint import (
        measure_featureprint_latency as measure_featureprint_latency,
    )
    from vinylid_ml.eval_metrics import (
        CalibrationResult as CalibrationResult,
    )
    from vinylid_ml.eval_metrics import (
        NNAmbiguityResult as NNAmbiguityResult,
    )
    from vinylid_ml.eval_metrics import (
        compute_confidence_calibration as compute_confidence_calibration,
    )
    from vinylid_ml.eval_metrics import (
        compute_nn_ambiguity as compute_nn_ambiguity,
    )
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
    from vinylid_ml.local_features import (
        LOCAL_FEATURE_MODEL_ID as LOCAL_FEATURE_MODEL_ID,
    )
    from vinylid_ml.local_features import (
        KeypointFeatures as KeypointFeatures,
    )
    from vinylid_ml.local_features import (
        LightGlueMatcher as LightGlueMatcher,
    )
    from vinylid_ml.local_features import (
        LocalFeatureMatcher as LocalFeatureMatcher,
    )
    from vinylid_ml.local_features import (
        MatchResult as MatchResult,
    )
    from vinylid_ml.local_features import (
        SuperPointExtractor as SuperPointExtractor,
    )
    from vinylid_ml.losses import (
        ArcFaceLoss as ArcFaceLoss,
    )
    from vinylid_ml.losses import (
        ProxyAnchorLoss as ProxyAnchorLoss,
    )
    from vinylid_ml.losses import (
        SupConLoss as SupConLoss,
    )
    from vinylid_ml.models import (
        ALL_MODEL_IDS as ALL_MODEL_IDS,
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
        create_model as create_model,
    )
    from vinylid_ml.models import (
        gem_pool as gem_pool,
    )
    from vinylid_ml.models import (
        get_device as get_device,
    )
    from vinylid_ml.report import (
        generate_report as generate_report,
    )
    from vinylid_ml.training import (
        FineTuneModel as FineTuneModel,
    )
    from vinylid_ml.training import (
        MultiViewTransform as MultiViewTransform,
    )
    from vinylid_ml.training import (
        TrainingConfig as TrainingConfig,
    )

__all__ = [
    "ALL_MODEL_IDS",
    "FEATUREPRINT_MODEL_ID",
    "LOCAL_FEATURE_MODEL_ID",
    "ArcFaceLoss",
    "CalibrationResult",
    "DINOv2Embedder",
    "EmbeddingModel",
    "EmbeddingResult",
    "FineTuneModel",
    "GalleryImageDataset",
    "KeypointFeatures",
    "LightGlueMatcher",
    "LocalFeatureMatcher",
    "MatchResult",
    "MultiViewTransform",
    "NNAmbiguityResult",
    "OpenCLIPEmbedder",
    "ProxyAnchorLoss",
    "SSCDEmbedder",
    "SupConLoss",
    "SuperPointExtractor",
    "TrainingConfig",
    "VinylIDError",
    "compute_confidence_calibration",
    "compute_nn_ambiguity",
    "create_model",
    "embed_dataset",
    "embed_images",
    "extract_feature_vector",
    "gem_pool",
    "generate_report",
    "get_device",
    "load_embeddings",
    "measure_featureprint_latency",
    "save_embeddings",
]

_LAZY_MODULES: dict[str, str] = {
    "ArcFaceLoss": "vinylid_ml.losses",
    "ProxyAnchorLoss": "vinylid_ml.losses",
    "SupConLoss": "vinylid_ml.losses",
    "FineTuneModel": "vinylid_ml.training",
    "MultiViewTransform": "vinylid_ml.training",
    "TrainingConfig": "vinylid_ml.training",
    "LOCAL_FEATURE_MODEL_ID": "vinylid_ml.local_features",
    "KeypointFeatures": "vinylid_ml.local_features",
    "LightGlueMatcher": "vinylid_ml.local_features",
    "LocalFeatureMatcher": "vinylid_ml.local_features",
    "MatchResult": "vinylid_ml.local_features",
    "SuperPointExtractor": "vinylid_ml.local_features",
    "FEATUREPRINT_MODEL_ID": "vinylid_ml.apple_featureprint",
    "embed_images": "vinylid_ml.apple_featureprint",
    "extract_feature_vector": "vinylid_ml.apple_featureprint",
    "measure_featureprint_latency": "vinylid_ml.apple_featureprint",
    "CalibrationResult": "vinylid_ml.eval_metrics",
    "NNAmbiguityResult": "vinylid_ml.eval_metrics",
    "compute_confidence_calibration": "vinylid_ml.eval_metrics",
    "compute_nn_ambiguity": "vinylid_ml.eval_metrics",
    "ALL_MODEL_IDS": "vinylid_ml.models",
    "create_model": "vinylid_ml.models",
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
    "generate_report": "vinylid_ml.report",
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
