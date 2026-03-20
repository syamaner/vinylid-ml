"""DINOv2 patch-level matching for C1 retrieval evaluation.

Extracts all 256 patch tokens from DINOv2 ViT-S/14 (16x16 grid at 224px) per
image instead of pooling to a single vector.  Match quality is computed via
patch-to-patch cosine similarity between query and gallery images.

This is the dense learned-feature analog of SIFT — it captures local structure
(text, logos, geometric patterns) that global embeddings may average away.
Evaluated as an offline oracle on the test split; per-image storage (~196KB fp16)
is too large for on-device deployment at 23k gallery scale (~4.5GB).

Typical usage::

    from vinylid_ml.patch_matching import DINOv2PatchExtractor, PatchMatcher

    extractor = DINOv2PatchExtractor()
    pf_query = extractor.extract(query_path)
    pf_gallery = extractor.extract(gallery_path)
    result = PatchMatcher.match_best_avg(pf_query, pf_gallery)
    score = result.score
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import structlog
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from PIL import Image
from torchvision import transforms

__all__ = [
    "PATCHES_PER_IMAGE_224",
    "PATCH_DIM",
    "PATCH_MODEL_ID",
    "DINOv2PatchExtractor",
    "PatchFeatures",
    "PatchMatchResult",
    "PatchMatcher",
    "cache_path_for",
    "extract_with_cache",
    "load_cached_patches",
    "save_cached_patches",
]

#: Model identifier used throughout the evaluation pipeline.
PATCH_MODEL_ID: str = "C1-dinov2-patches"

#: Dimensionality of each DINOv2 ViT-S/14 patch token.
PATCH_DIM: int = 384

#: Number of patch tokens for 224x224 input (16x16 grid, patch size 14).
PATCHES_PER_IMAGE_224: int = 256

# ImageNet normalization (same as DINOv2Embedder in models.py)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass
class PatchFeatures:
    """Patch token features extracted from a single image by DINOv2.

    Attributes:
        patches: L2-normalised patch token embeddings of shape
            ``(N_patches, 384)`` where N_patches is typically 256 for 224px input.
        image_size: Original image dimensions as ``(width, height)`` in pixels.
    """

    patches: NDArray[np.float32]
    image_size: tuple[int, int]  # (W, H)


@dataclass
class PatchMatchResult:
    """Result of matching two images via patch-level cosine similarity.

    Attributes:
        score: Aggregated patch-matching score based on cosine similarity.
            Higher = better match.

            * For ``best_avg``: mean of per-query-patch maximum cosine
              similarities (one max per query patch, then averaged).
              Range ``[-1, 1]``.
            * For ``mutual_nn``: count of mutual nearest-neighbor patch pairs
              normalised by ``min(N_query, N_gallery)``.  Range ``[0, 1]``.
        num_patches_matched: Number of mutual nearest-neighbor patch pairs
            (only populated for ``mutual_nn`` strategy; ``0`` for ``best_avg``).
        top_k_patch_sims: Top-K patch-level cosine similarities, shape ``(K,)``.

            * For ``best_avg``: top-K largest per-query-patch maximum cosine
              similarities.
            * For ``mutual_nn``: top-K cosine similarities of mutual
              nearest-neighbor patch pairs.

            Useful for diagnostic analysis of the match-quality distribution.
    """

    score: float
    num_patches_matched: int
    top_k_patch_sims: NDArray[np.float32]


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class DINOv2PatchExtractor:
    """Extract all patch tokens from DINOv2 ViT-S/14.

    Loads the same model as ``DINOv2Embedder`` (A1) but returns the full
    ``(N_patches, 384)`` patch token matrix instead of pooling to a single vector.

    Args:
        device: Torch device.  ``None`` = auto-detect (MPS on Apple Silicon, else CPU).
        input_size: Input image resolution.  Must be divisible by 14.  Default 224.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        input_size: int = 224,
    ) -> None:
        if input_size % 14 != 0:
            raise ValueError(f"input_size must be divisible by 14, got {input_size}")

        self._device = device or _get_device()
        self._input_size = input_size

        logger.info(
            "loading_dinov2_patch_extractor",
            device=str(self._device),
            input_size=input_size,
        )
        hub_result = torch.hub.load(  # type: ignore[reportUnknownMemberType]
            "facebookresearch/dinov2", "dinov2_vits14", verbose=False
        )
        if not isinstance(hub_result, torch.nn.Module):
            raise TypeError(f"Expected nn.Module from torch.hub, got {type(hub_result)}")
        self._model: torch.nn.Module = hub_result
        self._model.to(self._device)
        self._model.eval()
        logger.info("dinov2_patch_extractor_loaded")

    @property
    def input_size(self) -> int:
        """Expected input image resolution."""
        return self._input_size

    @property
    def num_patches(self) -> int:
        """Number of patch tokens for the configured input size."""
        grid = self._input_size // 14
        return grid * grid

    def get_transforms(self) -> transforms.Compose:
        """ImageNet-normalized transforms matching DINOv2Embedder.

        Returns:
            Resize → CenterCrop → ToTensor → Normalize pipeline.
        """
        return transforms.Compose(
            [
                transforms.Resize(
                    self._input_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(self._input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )

    def extract(self, image: Image.Image | Path) -> PatchFeatures:
        """Extract L2-normalised patch tokens from a single image.

        Args:
            image: A PIL ``Image`` or ``Path`` to an image file.

        Returns:
            ``PatchFeatures`` with ``(N_patches, 384)`` float32 patches.

        Raises:
            FileNotFoundError: If ``image`` is a ``Path`` that does not exist.
            RuntimeError: If the image file cannot be opened.
        """
        pil_img, img_size = _load_image(image)
        tensor: torch.Tensor = self.get_transforms()(pil_img)  # type: ignore[assignment]
        batch = tensor.unsqueeze(0).to(self._device)

        with torch.inference_mode():
            features = self._model.forward_features(batch)  # type: ignore[operator]
            patch_tokens = torch.as_tensor(features["x_norm_patchtokens"])  # (1, N, 384)
            patch_tokens = F.normalize(patch_tokens, p=2, dim=-1)

        patches: NDArray[np.float32] = patch_tokens[0].cpu().numpy().astype(np.float32)
        return PatchFeatures(patches=patches, image_size=img_size)

    def extract_batch(
        self,
        images: torch.Tensor,
        image_sizes: list[tuple[int, int]],
    ) -> list[PatchFeatures]:
        """Extract patch tokens from a pre-processed batch.

        Args:
            images: Tensor of shape ``(B, 3, H, W)``, ImageNet-normalized.
            image_sizes: Original ``(width, height)`` per image in the batch.

        Returns:
            List of ``PatchFeatures``, one per batch item.
        """
        images = images.to(self._device)

        with torch.inference_mode():
            features = self._model.forward_features(images)  # type: ignore[operator]
            patch_tokens = torch.as_tensor(features["x_norm_patchtokens"])
            patch_tokens = F.normalize(patch_tokens, p=2, dim=-1)

        results: list[PatchFeatures] = []
        for i in range(patch_tokens.shape[0]):
            patches: NDArray[np.float32] = patch_tokens[i].cpu().numpy().astype(np.float32)
            results.append(PatchFeatures(patches=patches, image_size=image_sizes[i]))
        return results


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------


class PatchMatcher:
    """Stateless patch-level matching strategies.

    All methods are static — no state to initialise.  Two aggregation
    strategies are provided:

    * ``best_avg``: for each query patch, find the gallery patch with highest
      cosine similarity; return the mean of these per-patch maxima.
    * ``mutual_nn``: count bidirectional nearest-neighbor matches; return the
      count normalised by ``min(N_query, N_gallery)``.
    """

    @staticmethod
    def match_best_avg(
        query: PatchFeatures,
        gallery: PatchFeatures,
        top_k: int = 10,
    ) -> PatchMatchResult:
        """Best-match average: mean of per-query-patch maximum similarities.

        For each query patch, finds the highest cosine similarity to any gallery
        patch.  Returns the mean of these maxima.  Simple and fast — no spatial
        consistency check.

        Args:
            query: Query image patch features.
            gallery: Gallery image patch features.
            top_k: Number of top per-patch similarities to store in the result.

        Returns:
            ``PatchMatchResult`` with aggregated score.
        """
        # (N_q, 384) @ (384, N_g) → (N_q, N_g) cosine similarity
        sim = query.patches @ gallery.patches.T
        per_query_max = sim.max(axis=1)  # (N_q,)
        score = float(per_query_max.mean())
        top_k_sims = np.sort(per_query_max)[::-1][:top_k].astype(np.float32)
        return PatchMatchResult(
            score=score,
            num_patches_matched=0,
            top_k_patch_sims=top_k_sims,
        )

    @staticmethod
    def match_mutual_nn(
        query: PatchFeatures,
        gallery: PatchFeatures,
        top_k: int = 10,
    ) -> PatchMatchResult:
        """Mutual nearest-neighbor matching between patch sets.

        A pair ``(q_i, g_j)`` is a mutual NN if ``g_j`` is the nearest gallery
        patch to ``q_i`` AND ``q_i`` is the nearest query patch to ``g_j``.
        The score is the count of mutual NN pairs normalised by
        ``min(N_query, N_gallery)``.

        Args:
            query: Query image patch features.
            gallery: Gallery image patch features.
            top_k: Number of top mutual-NN similarities to store in the result.

        Returns:
            ``PatchMatchResult`` with normalised mutual-NN count as score.
        """
        sim = query.patches @ gallery.patches.T  # (N_q, N_g)
        nn_q2g = sim.argmax(axis=1)  # (N_q,) — nearest gallery for each query
        nn_g2q = sim.argmax(axis=0)  # (N_g,) — nearest query for each gallery

        # Mutual: q_i's nearest is g_j, and g_j's nearest is q_i
        n_q = query.patches.shape[0]
        query_indices = np.arange(n_q, dtype=np.intp)
        roundtrip = np.array(nn_g2q[nn_q2g], dtype=np.intp)
        mutual_mask = roundtrip == query_indices
        num_mutual = int(mutual_mask.sum())

        min_patches = min(query.patches.shape[0], gallery.patches.shape[0])
        score = num_mutual / min_patches if min_patches > 0 else 0.0

        # Collect similarities for mutual pairs
        mutual_sims = sim[query_indices[mutual_mask], nn_q2g[mutual_mask]]
        top_k_sims = np.sort(mutual_sims)[::-1][:top_k].astype(np.float32)

        return PatchMatchResult(
            score=score,
            num_patches_matched=num_mutual,
            top_k_patch_sims=top_k_sims,
        )

    @staticmethod
    def match(
        query: PatchFeatures,
        gallery: PatchFeatures,
        strategy: Literal["best_avg", "mutual_nn"] = "best_avg",
        top_k: int = 10,
    ) -> PatchMatchResult:
        """Match two patch feature sets using the specified strategy.

        Args:
            query: Query image patch features.
            gallery: Gallery image patch features.
            strategy: Aggregation strategy — ``\"best_avg\"`` or ``\"mutual_nn\"``.
            top_k: Number of top similarities to store.

        Returns:
            ``PatchMatchResult`` from the chosen strategy.
        """
        if strategy == "best_avg":
            return PatchMatcher.match_best_avg(query, gallery, top_k=top_k)
        if strategy == "mutual_nn":
            return PatchMatcher.match_mutual_nn(query, gallery, top_k=top_k)
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'best_avg' or 'mutual_nn'.")


# ---------------------------------------------------------------------------
# Feature caching
# ---------------------------------------------------------------------------


def cache_path_for(image_path: Path, cache_dir: Path) -> Path:
    """Compute the ``.npz`` cache path for an image.

    The filename is the hex-encoded SHA-256 digest of the absolute resolved path.

    Args:
        image_path: Path to the image.
        cache_dir: Directory that will contain cached ``.npz`` files.

    Returns:
        ``cache_dir / "<sha256>.npz"``.
    """
    key = hashlib.sha256(str(image_path.resolve()).encode()).hexdigest()
    return cache_dir / f"{key}.npz"


def save_cached_patches(pf: PatchFeatures, path: Path) -> None:
    """Save ``PatchFeatures`` to a ``.npz`` file (fp16 for storage efficiency).

    Args:
        pf: Features to save.
        path: Destination ``.npz`` file path.
    """
    np.savez(
        path,
        patches=pf.patches.astype(np.float16),
        image_size=np.array(pf.image_size, dtype=np.int32),
    )


def load_cached_patches(path: Path) -> PatchFeatures:
    """Load ``PatchFeatures`` from a ``.npz`` cache file.

    Args:
        path: Path to a ``.npz`` file written by ``save_cached_patches``.

    Returns:
        ``PatchFeatures`` with float32 numpy arrays.
    """
    with np.load(path) as data:
        patches = data["patches"].astype(np.float32)
        # Re-normalize to restore L2-unit-norm invariant after fp16 roundtrip.
        norms = np.linalg.norm(patches, axis=-1, keepdims=True)
        patches = patches / np.clip(norms, 1e-12, None)
        return PatchFeatures(
            patches=patches,
            image_size=(int(data["image_size"][0]), int(data["image_size"][1])),
        )


def extract_with_cache(
    extractor: DINOv2PatchExtractor,
    image_paths: list[Path],
    cache_dir: Path,
) -> list[PatchFeatures]:
    """Extract patch features for a list of images, using a file cache.

    On the first call for each image, features are extracted and saved to
    ``cache_dir`` as ``<sha256>.npz``.  Subsequent calls load from cache.

    Args:
        extractor: Initialised ``DINOv2PatchExtractor``.
        image_paths: Images to process.
        cache_dir: Directory for ``.npz`` feature cache.

    Returns:
        List of ``PatchFeatures`` aligned with ``image_paths``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[PatchFeatures | None] = [None] * len(image_paths)
    missing_indices: list[int] = []

    for idx, path in enumerate(image_paths):
        cp = cache_path_for(path, cache_dir)
        if cp.exists():
            results[idx] = load_cached_patches(cp)
        else:
            missing_indices.append(idx)

    if missing_indices:
        logger.info(
            "extracting_patch_features",
            cached=len(image_paths) - len(missing_indices),
            to_extract=len(missing_indices),
        )
        for idx in missing_indices:
            pf = extractor.extract(image_paths[idx])
            cp = cache_path_for(image_paths[idx], cache_dir)
            save_cached_patches(pf, cp)
            results[idx] = pf

    none_paths = [image_paths[i] for i, r in enumerate(results) if r is None]
    if none_paths:
        raise RuntimeError(f"Feature extraction failed for {len(none_paths)} path(s): {none_paths}")
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_device() -> torch.device:
    """Return the best available torch device (MPS on Apple Silicon, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_image(image: Image.Image | Path) -> tuple[Image.Image, tuple[int, int]]:
    """Load a PIL image and return ``(rgb_image, (width, height))``.

    Args:
        image: PIL ``Image`` or ``Path`` to an image file.

    Returns:
        Tuple of ``(PIL.Image in RGB mode, (width, height))``.

    Raises:
        FileNotFoundError: If ``image`` is a ``Path`` that does not exist.
        RuntimeError: If the image file cannot be opened.
    """
    if isinstance(image, Path):
        if not image.exists():
            raise FileNotFoundError(f"Image not found: {image}")
        try:
            with Image.open(image) as opened:
                pil_img = opened.convert("RGB")
                return pil_img, (pil_img.width, pil_img.height)
        except Exception as exc:
            raise RuntimeError(f"Could not open image {image}: {exc}") from exc
    else:
        rgb = image.convert("RGB")
        return rgb, (rgb.width, rgb.height)
