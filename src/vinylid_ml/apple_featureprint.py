"""Apple Vision FeaturePrint wrapper for A3 zero-shot evaluation.

Uses ``VNGenerateImageFeaturePrintRequest`` via ``pyobjc-framework-Vision``
to extract raw float vectors from images.  The Vision framework is macOS-only.

``VNFeaturePrintObservation`` exposes the raw feature bytes via ``.data``, so
we can produce the same ``(N, D)`` float32 embedding matrix as A1/A2/A4 and
plug directly into ``evaluate.py`` without any changes to the retrieval pipeline.

Typical usage::

    from pathlib import Path
    from vinylid_ml.apple_featureprint import embed_images, FEATUREPRINT_MODEL_ID

    paths = [Path("cover1.jpg"), Path("cover2.jpg")]
    embeddings = embed_images(paths)  # shape (2, D), L2-normalised float32
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from numpy.typing import NDArray

__all__ = [
    "FEATUREPRINT_MODEL_ID",
    "embed_images",
    "extract_feature_vector",
    "measure_featureprint_latency",
]

#: Model identifier used throughout the evaluation pipeline.
FEATUREPRINT_MODEL_ID: str = "A3-featureprint"

logger = structlog.get_logger()

# VNElementType enum values from the Vision framework
_VN_ELEMENT_TYPE_FLOAT: int = 1
_VN_ELEMENT_TYPE_DOUBLE: int = 2


def _import_vision() -> tuple[Any, Any]:
    """Import Vision and Foundation modules from pyobjc.

    Returns:
        Tuple of ``(Vision, Foundation)`` modules.

    Raises:
        ImportError: If ``pyobjc-framework-Vision`` is not installed.
    """
    try:
        import Foundation  # type: ignore[import-not-found]
        import Vision  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "pyobjc-framework-Vision is required for A3 evaluation. "
            "Install with: pip install pyobjc-framework-Vision"
        ) from exc
    return Vision, Foundation


def extract_feature_vector(image_path: Path) -> NDArray[np.float32]:
    """Extract a raw FeaturePrint vector for a single image.

    Runs ``VNGenerateImageFeaturePrintRequest`` and reads the raw float bytes
    from ``VNFeaturePrintObservation.data``.  The returned vector is **not**
    L2-normalised — callers should normalise before computing cosine
    similarities (``embed_images`` does this automatically).

    Args:
        image_path: Path to the image file (JPEG, PNG, etc.).

    Returns:
        1D float32 numpy array of length ``elementCount``
        (typically 2048 on macOS 14; may vary by OS version).

    Raises:
        ImportError: If ``pyobjc-framework-Vision`` is not installed.
        FileNotFoundError: If ``image_path`` does not exist.
        RuntimeError: If the Vision request fails or returns no results.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    vision, foundation = _import_vision()

    url = foundation.NSURL.fileURLWithPath_(str(image_path.resolve()))
    handler = vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})
    request = vision.VNGenerateImageFeaturePrintRequest.alloc().init()

    try:
        handler.performRequests_error_([request], None)
    except Exception as exc:
        raise RuntimeError(
            f"VNGenerateImageFeaturePrintRequest failed for {image_path}: {exc}"
        ) from exc

    results = request.results()
    if not results:
        raise RuntimeError(f"No FeaturePrint results returned for {image_path}")

    observation = results[0]
    element_count: int = int(observation.elementCount())
    element_type: int = int(observation.elementType())

    # Validate and map VNElementType to numpy dtype
    if element_type not in (_VN_ELEMENT_TYPE_FLOAT, _VN_ELEMENT_TYPE_DOUBLE):
        raise RuntimeError(
            f"Unsupported VNElementType {element_type} for {image_path}. "
            f"Expected {_VN_ELEMENT_TYPE_FLOAT} (float32) or "
            f"{_VN_ELEMENT_TYPE_DOUBLE} (float64)."
        )
    dtype: type[np.float32] | type[np.float64] = (
        np.float32 if element_type == _VN_ELEMENT_TYPE_FLOAT else np.float64
    )

    raw_bytes: bytes = bytes(observation.data())
    vec: NDArray[Any] = np.frombuffer(raw_bytes, dtype=dtype)

    if len(vec) != element_count:
        raise RuntimeError(
            f"FeaturePrint data size mismatch: got {len(vec)} elements, "
            f"expected {element_count} for {image_path}"
        )

    return vec.astype(np.float32).copy()


def embed_images(
    image_paths: list[Path],
    *,
    show_progress: bool = False,
) -> NDArray[np.float32]:
    """Embed a list of images with Apple FeaturePrint and return L2-normalised vectors.

    Calls ``extract_feature_vector`` for each image, stacks the results into
    an ``(N, D)`` float32 matrix, and L2-normalises each row so that cosine
    similarity equals a dot product — matching the convention used by A1/A2/A4.

    Args:
        image_paths: Ordered list of image paths to embed.
        show_progress: If ``True``, log progress every 100 images.

    Returns:
        Float32 array of shape ``(N, D)`` with L2-normalised rows, where
        ``N = len(image_paths)`` and ``D`` is the FeaturePrint dimension.

    Raises:
        ImportError: If ``pyobjc-framework-Vision`` is not installed.
        ValueError: If ``image_paths`` is empty.
        RuntimeError: If any Vision request fails.
    """
    if not image_paths:
        raise ValueError("image_paths must be non-empty")

    n = len(image_paths)
    vecs: list[NDArray[np.float32]] = []

    logger.info("featureprint_embed_start", num_images=n)
    start = time.monotonic()

    for i, path in enumerate(image_paths):
        vec = extract_feature_vector(path)
        vecs.append(vec)
        if show_progress and (i + 1) % 100 == 0:
            elapsed = time.monotonic() - start
            throughput = (i + 1) / elapsed if elapsed > 0 else 0.0
            logger.info(
                "featureprint_embed_progress",
                done=f"{i + 1}/{n}",
                throughput_ips=round(throughput, 1),
            )

    matrix: NDArray[np.float32] = np.stack(vecs, axis=0)  # (N, D)

    # L2-normalise each row; guard against zero-norm vectors
    norms: NDArray[np.float32] = np.linalg.norm(matrix, axis=1, keepdims=True).astype(np.float32)
    norms[norms == 0.0] = 1.0
    normalised: NDArray[np.float32] = (matrix / norms).astype(np.float32)

    elapsed = time.monotonic() - start
    logger.info(
        "featureprint_embed_complete",
        num_images=n,
        embedding_dim=matrix.shape[1],
        total_time_s=round(elapsed, 2),
        throughput_ips=round(n / elapsed if elapsed > 0 else 0.0, 1),
    )
    return normalised


def measure_featureprint_latency(
    image_path: Path,
    *,
    n_warmup: int = 5,
    n_timed: int = 50,
) -> dict[str, float]:
    """Measure single-image FeaturePrint extraction latency.

    Runs ``n_warmup`` forward passes (discarded), then ``n_timed`` timed
    passes.  Reports p50, p95, p99 in milliseconds.

    Args:
        image_path: Path to a sample image used for all timing runs.
        n_warmup: Warmup iterations (not timed). Default 5.
        n_timed: Timed iterations. Default 50.

    Returns:
        Dict with keys ``"p50_ms"``, ``"p95_ms"``, ``"p99_ms"`` (float ms).

    Raises:
        ValueError: If ``n_timed`` < 1 or ``n_warmup`` < 0.
    """
    if n_timed < 1:
        raise ValueError(f"n_timed must be >= 1, got {n_timed}")
    if n_warmup < 0:
        raise ValueError(f"n_warmup must be >= 0, got {n_warmup}")
    for _ in range(n_warmup):
        extract_feature_vector(image_path)

    times_ms: list[float] = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        extract_feature_vector(image_path)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    n = len(times_ms)
    return {
        "p50_ms": times_ms[n // 2],
        "p95_ms": times_ms[max(0, int(n * 0.95) - 1)],
        "p99_ms": times_ms[max(0, int(n * 0.99) - 1)],
    }
