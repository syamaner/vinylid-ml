"""SuperPoint + LightGlue local feature wrapper for C2 pairwise retrieval.

Implements neural local feature extraction (SuperPoint) and matching (LightGlue)
for album cover retrieval.  Unlike the embedding-based models (A1-A4), local
feature matching is inherently pairwise — there is no single embedding vector.
Model C2 is therefore evaluated via explicit gallery matching, not nearest-neighbour
search in an embedding space.

Typical usage::

    from pathlib import Path
    from vinylid_ml.local_features import LocalFeatureMatcher

    matcher = LocalFeatureMatcher()
    gallery_feats = matcher.extract_features(gallery_paths, cache_dir=Path("data/cache"))
    query_feat = matcher.extract_features([query_path])[0]
    ranked = matcher.rank_gallery(query_feat, gallery_feats)   # indices, best-first

Weights (~5 MB SuperPoint, ~45 MB LightGlue) are downloaded from GitHub Releases on
first use and cached in ``~/.cache/torch/hub/checkpoints/``.
"""

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import hashlib
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import torch
from numpy.typing import NDArray
from PIL import Image

from vinylid_ml.models import get_device

if TYPE_CHECKING:
    from lightglue import LightGlue as _LightGlueT
    from lightglue import SuperPoint as _SuperPointT

__all__ = [
    "LOCAL_FEATURE_MODEL_ID",
    "KeypointFeatures",
    "LightGlueMatcher",
    "LocalFeatureMatcher",
    "MatchResult",
    "SuperPointExtractor",
]

#: Model identifier used throughout the evaluation pipeline.
LOCAL_FEATURE_MODEL_ID: str = "C2-superpoint-lightglue"

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass
class KeypointFeatures:
    """Keypoint features extracted from a single image by SuperPoint.

    Attributes:
        keypoints: Keypoint positions of shape ``(N, 2)`` in ``(x, y)`` image coordinates.
        descriptors: L2-normalised 256-dim descriptors of shape ``(N, 256)``.
        scores: Per-keypoint detection scores of shape ``(N,)``, in ``[0, 1]``.
        image_size: Original image dimensions as ``(width, height)`` in pixels.
    """

    keypoints: NDArray[np.float32]
    descriptors: NDArray[np.float32]
    scores: NDArray[np.float32]
    image_size: tuple[int, int]  # (W, H)


@dataclass
class MatchResult:
    """Result of matching two ``KeypointFeatures`` images with LightGlue.

    Attributes:
        num_matches: Number of retained correspondences after LightGlue's
            internal pruning / thresholding. This is not a geometrically
            verified inlier count from RANSAC / homography fitting.
        match_scores: Per-correspondence confidence scores of shape
            ``(num_matches,)``, in ``[0, 1]``. Empty array when
            ``num_matches == 0``.
        confidence: Mean match score in ``[0, 1]``. Zero if no matches found.
    """

    num_matches: int
    match_scores: NDArray[np.float32]
    confidence: float

    @property
    def num_inliers(self) -> int:
        """Backward-compatible alias for ``num_matches``."""
        return self.num_matches

    @property
    def inlier_scores(self) -> NDArray[np.float32]:
        """Backward-compatible alias for ``match_scores``."""
        return self.match_scores


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class SuperPointExtractor:
    """SuperPoint keypoint and descriptor extractor.

    Wraps the ``cvg/LightGlue`` SuperPoint model.  Accepts PIL images or file
    paths.  Images are loaded as float32 RGB ``[0, 1]`` tensors; SuperPoint
    converts to grayscale internally before running the convolutional backbone.

    Args:
        max_num_keypoints: Maximum keypoints to detect per image.  Default 2048.
        device: Torch device.  ``None`` = auto-detect (CUDA, then MPS, then CPU).

    Raises:
        ImportError: If ``lightglue`` is not installed.
    """

    def __init__(
        self,
        max_num_keypoints: int = 2048,
        device: torch.device | None = None,
    ) -> None:
        self._device = device or get_device()
        self._max_kp = max_num_keypoints
        self._model: _SuperPointT = _load_superpoint(max_num_keypoints, self._device)

    @property
    def max_num_keypoints(self) -> int:
        """Maximum number of keypoints to detect per image."""
        return self._max_kp

    def extract(self, image: Image.Image | Path) -> KeypointFeatures:
        """Extract keypoints and descriptors from a single image.

        Args:
            image: A PIL ``Image`` (any mode, converted to RGB internally) or a
                ``Path`` to an image file.

        Returns:
            ``KeypointFeatures`` with ``N`` detected keypoints.

        Raises:
            FileNotFoundError: If ``image`` is a ``Path`` that does not exist.
            RuntimeError: If the image cannot be read.
        """
        img_tensor = _to_image_tensor(image, self._device)
        with torch.inference_mode():
            raw: dict[str, Any] = self._model.extract(img_tensor)  # type: ignore[attr-defined]
        return _feats_to_keypoint_features(raw)

    def extract_batch(
        self,
        paths: list[Path],
        batch_size: int = 8,
    ) -> list[KeypointFeatures]:
        """Extract features for a list of image paths.

        Processes images sequentially (SuperPoint's ``extract`` handles one image
        at a time).  The ``batch_size`` parameter is accepted for API consistency
        but does not affect extraction order.

        Args:
            paths: Image paths to process.
            batch_size: Hint for future batched extraction support.  Currently unused.

        Returns:
            List of ``KeypointFeatures`` aligned with ``paths``.
        """
        _ = batch_size  # reserved for future batched extraction
        return [self.extract(p) for p in paths]


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------


class LightGlueMatcher:
    """LightGlue feature matcher.

    Matches two ``KeypointFeatures`` using LightGlue's cross-attention
    transformer architecture. LightGlue's built-in adaptive point pruning and
    confidence thresholding reject weak correspondences. No explicit geometric
    verification step (for example RANSAC / homography estimation) is applied
    here, so reported counts are retained LightGlue correspondences rather than
    geometrically verified inliers.

    Args:
        device: Torch device.  ``None`` = auto-detect (CUDA, then MPS, then CPU).

    Raises:
        ImportError: If ``lightglue`` is not installed.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        self._device = device or get_device()
        self._model: _LightGlueT = _load_lightglue(self._device)

    def prepare_features(self, kp: KeypointFeatures) -> dict[str, torch.Tensor]:
        """Convert ``KeypointFeatures`` to LightGlue input tensors on matcher device.

        Args:
            kp: Unbatched keypoint features for a single image.

        Returns:
            Dict with ``keypoints``, ``descriptors``, ``image_size`` tensors.
        """
        return _keypoint_features_to_tensor(kp, self._device)

    def match_prepared(
        self,
        feats0: dict[str, torch.Tensor],
        feats1: dict[str, torch.Tensor],
    ) -> MatchResult:
        """Match two pre-converted LightGlue feature dicts.

        Args:
            feats0: Prepared features for image 0.
            feats1: Prepared features for image 1.

        Returns:
            ``MatchResult`` with retained match count and per-match confidence
            scores.
        """
        with torch.inference_mode():
            raw: dict[str, Any] = self._model({"image0": feats0, "image1": feats1})

        matches_tensor = _extract_matches_tensor(raw)
        scores_tensor = _extract_scores_tensor(raw)
        num_matches = int(matches_tensor.shape[0])
        match_scores = scores_tensor.detach().cpu().numpy().astype(np.float32)
        confidence = float(match_scores.mean()) if num_matches > 0 else 0.0
        return MatchResult(
            num_matches=num_matches,
            match_scores=match_scores,
            confidence=confidence,
        )

    def count_matches_prepared(
        self,
        feats0: dict[str, torch.Tensor],
        feats1: dict[str, torch.Tensor],
    ) -> int:
        """Return only the retained match count for two prepared feature dicts.

        This avoids score extraction and CPU transfers in hot evaluation loops.

        Args:
            feats0: Prepared features for image 0.
            feats1: Prepared features for image 1.

        Returns:
            Number of retained correspondences returned by LightGlue.
        """
        with torch.inference_mode():
            raw: dict[str, Any] = self._model({"image0": feats0, "image1": feats1})
        matches_tensor = _extract_matches_tensor(raw)
        return int(matches_tensor.shape[0])

    def match_num_inliers_prepared(
        self,
        feats0: dict[str, torch.Tensor],
        feats1: dict[str, torch.Tensor],
    ) -> int:
        """Backward-compatible alias for ``count_matches_prepared``."""
        return self.count_matches_prepared(feats0, feats1)

    def match(self, kp0: KeypointFeatures, kp1: KeypointFeatures) -> MatchResult:
        """Match two ``KeypointFeatures`` and return match statistics.

        Args:
            kp0: Features for the first (e.g. query) image.
            kp1: Features for the second (e.g. gallery) image.

        Returns:
            ``MatchResult`` with retained match count and per-match confidence
            scores.
        """
        feats0 = self.prepare_features(kp0)
        feats1 = self.prepare_features(kp1)
        return self.match_prepared(feats0, feats1)

    def count_matches(self, kp0: KeypointFeatures, kp1: KeypointFeatures) -> int:
        """Return only the retained match count for two images.

        Args:
            kp0: Features for the first (e.g. query) image.
            kp1: Features for the second (e.g. gallery) image.

        Returns:
            Number of retained correspondences returned by LightGlue.
        """
        feats0 = self.prepare_features(kp0)
        feats1 = self.prepare_features(kp1)
        return self.count_matches_prepared(feats0, feats1)

    def match_num_inliers(self, kp0: KeypointFeatures, kp1: KeypointFeatures) -> int:
        """Backward-compatible alias for ``count_matches``."""
        return self.count_matches(kp0, kp1)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class LocalFeatureMatcher:
    """Orchestrates SuperPoint extraction + LightGlue matching with feature caching.

    Suitable for both sample evaluation (full pairwise) and test-split evaluation
    (LightGlue re-ranking of A4-sscd top-K candidates).

    Feature cache: ``.npz`` files saved under ``cache_dir``, keyed by
    ``SHA-256(absolute_path)``.  Re-running with the same images skips extraction.

    Args:
        max_keypoints: Maximum keypoints per image.  Default 2048.
        device: Torch device.  ``None`` = auto-detect (CUDA, then MPS, then CPU).
    """

    def __init__(
        self,
        max_keypoints: int = 2048,
        device: torch.device | None = None,
    ) -> None:
        _device = device or get_device()
        self._extractor = SuperPointExtractor(max_num_keypoints=max_keypoints, device=_device)
        self._matcher = LightGlueMatcher(device=_device)

    def extract_feature(self, image: Image.Image | Path) -> KeypointFeatures:
        """Extract keypoint features for a single image or image path."""
        return self._extractor.extract(image)

    def prepare_features(self, kp: KeypointFeatures) -> dict[str, torch.Tensor]:
        """Convert one ``KeypointFeatures`` to LightGlue-ready tensors."""
        return self._matcher.prepare_features(kp)

    def match_prepared(
        self,
        feats0: dict[str, torch.Tensor],
        feats1: dict[str, torch.Tensor],
    ) -> MatchResult:
        """Match two prepared feature dicts."""
        return self._matcher.match_prepared(feats0, feats1)

    def count_matches_prepared(
        self,
        feats0: dict[str, torch.Tensor],
        feats1: dict[str, torch.Tensor],
    ) -> int:
        """Count retained LightGlue correspondences for prepared feature dicts."""
        return self._matcher.count_matches_prepared(feats0, feats1)

    def match(self, kp0: KeypointFeatures, kp1: KeypointFeatures) -> MatchResult:
        """Match two extracted feature sets."""
        return self._matcher.match(kp0, kp1)

    def count_matches(self, kp0: KeypointFeatures, kp1: KeypointFeatures) -> int:
        """Count retained LightGlue correspondences for two extracted feature sets."""
        return self._matcher.count_matches(kp0, kp1)

    def extract_features(
        self,
        image_paths: list[Path],
        batch_size: int = 8,
        cache_dir: Path | None = None,
    ) -> list[KeypointFeatures]:
        """Extract and cache keypoint features for a list of images.

        On the first call for each image, features are extracted and saved to
        ``cache_dir`` as ``<sha256>.npz``.  Subsequent calls load from the cache,
        making repeated evaluation runs fast.

        Args:
            image_paths: Images to process.
            batch_size: Passed through to ``SuperPointExtractor.extract_batch``.
            cache_dir: Directory for ``.npz`` feature cache.  Defaults to
                ``~/.cache/vinylid_ml/local_features/C2-superpoint-lightglue/``.

        Returns:
            List of ``KeypointFeatures`` aligned with ``image_paths``.
        """
        resolved_cache = (
            cache_dir
            if cache_dir is not None
            else Path.home() / ".cache" / "vinylid_ml" / "local_features" / LOCAL_FEATURE_MODEL_ID
        )
        resolved_cache.mkdir(parents=True, exist_ok=True)
        results: list[KeypointFeatures | None] = [None] * len(image_paths)
        missing_paths: list[Path] = []
        missing_indices: list[int] = []

        max_kp = self._extractor.max_num_keypoints
        for idx, path in enumerate(image_paths):
            cache_path = _cache_path_for(path, resolved_cache, max_kp)
            if cache_path.exists():
                results[idx] = _load_cached_features(cache_path)
            else:
                missing_paths.append(path)
                missing_indices.append(idx)

        if missing_paths:
            extracted = self._extractor.extract_batch(missing_paths, batch_size=batch_size)
            for idx, path, kp in zip(missing_indices, missing_paths, extracted, strict=True):
                cache_path = _cache_path_for(path, resolved_cache, max_kp)
                _save_cached_features(kp, cache_path)
                results[idx] = kp

        none_paths = [image_paths[i] for i, r in enumerate(results) if r is None]
        if none_paths:
            raise RuntimeError(
                f"Feature extraction failed for {len(none_paths)} path(s): {none_paths}"
            )
        return [r for r in results if r is not None]

    def rank_gallery(
        self,
        query: KeypointFeatures,
        gallery: list[KeypointFeatures],
    ) -> NDArray[np.intp]:
        """Rank gallery images by match quality against a query.

        Matches the query against every gallery image and sorts by descending
        retained LightGlue match count.

        Args:
            query: Query image keypoint features.
            gallery: Gallery image keypoint features (one per album).

        Returns:
            Array of gallery indices sorted by descending match count (best
            match first), shape ``(len(gallery),)``.
        """
        query_prepared = self.prepare_features(query)
        gallery_prepared = [self.prepare_features(g) for g in gallery]
        match_counts = np.array(
            [
                self.count_matches_prepared(query_prepared, g_prepared)
                for g_prepared in gallery_prepared
            ],
            dtype=np.int64,
        )
        return np.argsort(-match_counts).astype(np.intp)

    def measure_latency(
        self,
        image_paths: list[Path],
        n_warmup: int = 5,
        n_timed: int = 50,
    ) -> dict[str, float]:
        """Measure end-to-end extraction + 1-vs-1 matching latency.

        Times the cost of extracting features from one image and matching against
        a second.  Uses two fixed images to keep variance low.

        Args:
            image_paths: At least 2 image paths for warmup and timed runs.
            n_warmup: Warmup iterations (not included in reported statistics).
            n_timed: Timed iterations.  Default 50.

        Returns:
            Dict with keys ``"p50_ms"``, ``"p95_ms"``, ``"p99_ms"`` (milliseconds).

        Raises:
            ValueError: If ``n_timed < 1``, ``n_warmup < 0``, or fewer than 2 paths
                are provided.
        """
        if n_timed < 1:
            raise ValueError(f"n_timed must be >= 1, got {n_timed}")
        if n_warmup < 0:
            raise ValueError(f"n_warmup must be >= 0, got {n_warmup}")
        if len(image_paths) < 2:
            raise ValueError(f"At least 2 image_paths required, got {len(image_paths)}")

        path0, path1 = image_paths[0], image_paths[1]

        for _ in range(n_warmup):
            kp0 = self.extract_feature(path0)
            kp1 = self.extract_feature(path1)
            self.match(kp0, kp1)

        times_ms: list[float] = []
        for _ in range(n_timed):
            t0 = time.perf_counter()
            kp0 = self.extract_feature(path0)
            kp1 = self.extract_feature(path1)
            self.match(kp0, kp1)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

        times_ms.sort()
        n = len(times_ms)
        return {
            "p50_ms": times_ms[n // 2],
            "p95_ms": times_ms[max(0, int(n * 0.95) - 1)],
            "p99_ms": times_ms[max(0, int(n * 0.99) - 1)],
        }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_superpoint(max_num_keypoints: int, device: torch.device) -> _SuperPointT:
    """Load and return an eval-mode SuperPoint model on ``device``.

    Args:
        max_num_keypoints: Maximum keypoints per image.
        device: Target torch device.

    Returns:
        SuperPoint model in eval mode.
    """
    from lightglue import SuperPoint  # type: ignore[import-untyped]

    logger.info(
        "loading_superpoint",
        max_num_keypoints=max_num_keypoints,
        device=str(device),
    )
    model: _SuperPointT = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
    logger.info("superpoint_loaded")
    return model


def _load_lightglue(device: torch.device) -> _LightGlueT:
    """Load and return an eval-mode LightGlue matcher on ``device``.

    Flash attention (``flash=True``) is CUDA-only in LightGlue's CrossBlock.
    On MPS we instead patch ``CrossBlock.forward`` to call
    ``F.scaled_dot_product_attention`` directly, which uses Apple's Metal
    memory-efficient attention kernel (~2x faster than the default einsum path).

    Args:
        device: Target torch device.

    Returns:
        LightGlue model in eval mode, configured for SuperPoint descriptors.
    """
    from lightglue import LightGlue  # type: ignore[import-untyped]

    use_flash = device.type == "cuda"
    logger.info("loading_lightglue", device=str(device), flash=use_flash)
    model: _LightGlueT = LightGlue(features="superpoint", flash=use_flash).eval().to(device)

    # On MPS: patch CrossBlock to use SDPA for cross-attention (~2x faster).
    if device.type == "mps":
        _patch_cross_attention_sdpa(model)
        logger.info("lightglue_cross_attn_sdpa_patched")

    logger.info("lightglue_loaded")
    return model


def _patch_cross_attention_sdpa(model: Any) -> None:
    """Patch all CrossBlock instances in ``model`` to use SDPA for cross-attention.

    LightGlue's ``CrossBlock`` uses manual einsum-based attention on non-CUDA
    devices.  ``F.scaled_dot_product_attention`` maps to Metal's fused
    memory-efficient kernel on MPS (PyTorch 2.x), giving ~2x speedup.

    The patch is mathematically equivalent: SDPA applies ``scale=1/sqrt(d_k)``
    internally, which matches the original path that pre-scales q and k each
    by ``scale**0.5`` before the dot-product.

    Args:
        model: A LightGlue instance whose TransformerLayer.cross_attn modules
            will be patched in-place.
    """
    import torch.nn.functional as F  # noqa: N812

    def _sdpa_forward(
        self: Any,
        x0: torch.Tensor,
        x1: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1),
        )
        # SDPA applies scale=1/sqrt(head_dim) — equivalent to the manual path
        # that pre-scales qk0 and qk1 each by self.scale**0.5.
        m0 = F.scaled_dot_product_attention(
            qk0.contiguous(), qk1.contiguous(), v1.contiguous(), attn_mask=mask
        )
        m1 = F.scaled_dot_product_attention(
            qk1.contiguous(),
            qk0.contiguous(),
            v0.contiguous(),
            attn_mask=mask.transpose(-1, -2) if mask is not None else None,
        )
        if mask is not None:
            m0, m1 = m0.nan_to_num(), m1.nan_to_num()

        def _reshape(t: torch.Tensor) -> torch.Tensor:
            return t.transpose(1, 2).flatten(start_dim=-2)

        m0, m1 = self.map_(_reshape, m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1

    for layer in model.transformers:  # type: ignore[union-attr]
        layer.cross_attn.forward = types.MethodType(_sdpa_forward, layer.cross_attn)


def _to_image_tensor(image: Image.Image | Path, device: torch.device) -> torch.Tensor:
    """Convert a PIL Image or file path to a float32 ``[0, 1]`` RGB tensor.

    Args:
        image: PIL ``Image`` (converted to RGB) or a ``Path`` to an image file.
        device: Target torch device for the output tensor.

    Returns:
        Float32 tensor of shape ``(3, H, W)`` in ``[0, 1]``.

    Raises:
        FileNotFoundError: If ``image`` is a ``Path`` that does not exist.
        RuntimeError: If the image file cannot be opened.
    """
    if isinstance(image, Path):
        if not image.exists():
            raise FileNotFoundError(f"Image not found: {image}")
        try:
            with Image.open(image) as _opened:
                pil_img = _opened.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Could not open image {image}: {exc}") from exc
    else:
        pil_img = image.convert("RGB")

    arr = np.array(pil_img, dtype=np.uint8)  # (H, W, 3)
    # Normalize to [0, 1] and reorder to (C, H, W) — matching lightglue.utils.numpy_image_to_torch
    tensor = torch.from_numpy(arr).permute(2, 0, 1).to(device=device, dtype=torch.float32)
    tensor = tensor.div_(255.0)
    return tensor


def _feats_to_keypoint_features(feats: dict[str, Any]) -> KeypointFeatures:
    """Convert SuperPoint ``extract()`` output dict to ``KeypointFeatures``.

    Removes the batch dimension (``B=1``) and moves tensors to CPU numpy arrays.

    Args:
        feats: Dict returned by ``SuperPoint.extract()``.  Keys: ``"keypoints"``
            ``(1, N, 2)``, ``"descriptors"`` ``(1, N, 256)``,
            ``"keypoint_scores"`` ``(1, N)``, ``"image_size"`` ``(1, 2)``.

    Returns:
        ``KeypointFeatures`` with arrays of shape ``(N, *)`` (no batch dim).
    """
    keypoints: NDArray[np.float32] = feats["keypoints"][0].cpu().numpy().astype(np.float32)
    descriptors: NDArray[np.float32] = feats["descriptors"][0].cpu().numpy().astype(np.float32)
    scores: NDArray[np.float32] = feats["keypoint_scores"][0].cpu().numpy().astype(np.float32)
    img_size_arr: NDArray[np.float32] = feats["image_size"][0].cpu().numpy()
    image_size = (int(img_size_arr[0]), int(img_size_arr[1]))  # (W, H)
    return KeypointFeatures(
        keypoints=keypoints,
        descriptors=descriptors,
        scores=scores,
        image_size=image_size,
    )


def _keypoint_features_to_tensor(
    kp: KeypointFeatures,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Convert ``KeypointFeatures`` to a batched ``(B=1)`` dict for LightGlue.

    LightGlue's ``forward`` expects each image dict to have ``keypoints``
    ``[B, M, 2]``, ``descriptors`` ``[B, M, D]``, and ``image_size`` ``[B, 2]``.

    Args:
        kp: ``KeypointFeatures`` with unbatched arrays.
        device: Target torch device.

    Returns:
        Dict with tensors on ``device``, all with a leading batch dimension of 1.
    """
    return {
        "keypoints": torch.from_numpy(kp.keypoints).unsqueeze(0).to(device),
        "descriptors": torch.from_numpy(kp.descriptors).unsqueeze(0).to(device),
        "image_size": torch.tensor([[kp.image_size[0], kp.image_size[1]]], dtype=torch.float32).to(
            device
        ),
    }


def _extract_matches_tensor(raw: dict[str, Any]) -> torch.Tensor:
    """Extract ``matches`` tensor (shape ``(M, 2)``) from LightGlue raw output."""
    matches = raw["matches"]
    matches_tensor = matches[0] if isinstance(matches, list) else matches
    if matches_tensor.ndim == 3 and matches_tensor.shape[0] == 1:
        matches_tensor = matches_tensor[0]
    return matches_tensor


def _extract_scores_tensor(raw: dict[str, Any]) -> torch.Tensor:
    """Extract ``scores`` tensor (shape ``(M,)``) from LightGlue raw output."""
    scores = raw["scores"]
    scores_tensor = scores[0] if isinstance(scores, list) else scores
    if scores_tensor.ndim == 2 and scores_tensor.shape[0] == 1:
        scores_tensor = scores_tensor[0]
    return scores_tensor


def _cache_path_for(image_path: Path, cache_dir: Path, max_num_keypoints: int) -> Path:
    """Compute the ``.npz`` cache path for an image.

    The filename is the hex-encoded SHA-256 digest of the absolute resolved path
    and ``max_num_keypoints``, so caches from different extractor configurations
    never collide.  Changing ``max_num_keypoints`` automatically produces cache
    misses rather than silently reusing stale features.

    Args:
        image_path: Path to the image.
        cache_dir: Directory that will contain cached ``.npz`` files.
        max_num_keypoints: Extractor keypoint limit — included in the key so
            runs with different limits use separate cache entries.

    Returns:
        ``cache_dir / "<sha256>.npz"``.
    """
    key_input = f"{image_path.resolve()}|max_kp={max_num_keypoints}"
    key = hashlib.sha256(key_input.encode()).hexdigest()
    return cache_dir / f"{key}.npz"


def _save_cached_features(kp: KeypointFeatures, path: Path) -> None:
    """Save ``KeypointFeatures`` to a ``.npz`` file.

    Args:
        kp: Features to save.
        path: Destination ``.npz`` file path.
    """
    np.savez(
        path,
        keypoints=kp.keypoints,
        descriptors=kp.descriptors,
        scores=kp.scores,
        image_size=np.array(kp.image_size, dtype=np.int32),
    )


def _load_cached_features(path: Path) -> KeypointFeatures:
    """Load ``KeypointFeatures`` from a ``.npz`` cache file.

    Args:
        path: Path to a ``.npz`` file written by ``_save_cached_features``.

    Returns:
        ``KeypointFeatures`` with float32 numpy arrays.
    """
    with np.load(path) as data:
        return KeypointFeatures(
            keypoints=data["keypoints"].astype(np.float32),
            descriptors=data["descriptors"].astype(np.float32),
            scores=data["scores"].astype(np.float32),
            image_size=(int(data["image_size"][0]), int(data["image_size"][1])),
        )
