#!/usr/bin/env python3
"""Evaluate global embedding models against real-world phone photos (stories #15, #53).

Two evaluation sets are supported via ``--eval-set``:

  ``--eval-set complete``  (default):
    Loads the 203 labeled iPhone captures from ``test_complete/``, strips all
    EXIF metadata in-memory, embeds each photo and computes retrieval metrics
    against the test-split gallery embeddings.  Results appended to
    ``results/phone_eval_summary.csv``.

  ``--eval-set sample``  (story #53):
    Runs against the same 55-photo / 50-matched-query sample used by C2
    sample-mode evaluation (``test_sample_matched.csv``, ``test_sample`` dir),
    using ``match_score_min=60`` and the corresponding sample-mode gallery
    selection: the canonical test-split gallery plus any additional albums
    referenced by the matched CSV. Enables direct apples-to-apples comparison
    with C2 LightGlue results. Results appended to
    ``results/phone_sample_eval_summary.csv``.

Privacy guardrail: EXIF is stripped in-memory on load. No photo paths are
written to any output file. Only aggregate metrics are reported.

Usage::

    python scripts/evaluate_phone_photos.py \\
        --config configs/dataset-remote.yaml \\
        --model A4-sscd

    # Story #53: A-series vs C2 on the same 55-photo sample
    python scripts/evaluate_phone_photos.py \\
        --config configs/dataset-remote.yaml \\
        --eval-set sample \\
        --models A1-dinov2-cls,A1-dinov2-gem,A2-openclip,A4-sscd
"""

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import torch
import yaml
from numpy.typing import NDArray
from PIL import Image as PILImage
from PIL import ImageOps
from torchvision import transforms
from torchvision.transforms import functional as TF  # noqa: N812

from vinylid_ml.apple_featureprint import FEATUREPRINT_MODEL_ID, extract_feature_vector
from vinylid_ml.dataset import load_manifest, load_splits
from vinylid_ml.eval_metrics import RetrievalMetrics, compute_retrieval_metrics
from vinylid_ml.gallery import load_embeddings
from vinylid_ml.models import ALL_MODEL_IDS, EmbeddingModel
from vinylid_ml.models import create_model as _create_model

logger = structlog.get_logger()

# Minimum fuzzy-match score for a phone photo to be included in evaluation.
_DEFAULT_MATCH_SCORE_MIN: int = 60

# Evaluation set choices.
_EVAL_SET_COMPLETE: str = "complete"
_EVAL_SET_SAMPLE: str = "sample"

# All model IDs accepted by evaluate_phone_photos.py.
# A3-featureprint (macOS-only, Apple Vision) is listed here but not in ALL_MODEL_IDS.
_PHONE_EVAL_MODEL_IDS: tuple[str, ...] = (*ALL_MODEL_IDS, FEATUREPRINT_MODEL_ID)


def _get_git_sha() -> str | None:
    """Return current git commit SHA, or None if unavailable.

    Returns:
        Hex SHA string or ``None``.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _load_phone_photo_stripped(photo_path: Path) -> PILImage.Image:
    """Load a phone photo and strip all EXIF metadata in-memory.

    Creates a clean pixel-only copy without touching the original file.
    This must be called for every real-world test photo to comply with
    the project privacy policy.

    Args:
        photo_path: Path to the original phone photo (JPEG/HEIC/etc).

    Returns:
        A fresh PIL Image in RGB mode with no metadata.

    Raises:
        OSError: If the file cannot be opened.
    """
    with PILImage.open(photo_path) as img:
        # Apply EXIF orientation before stripping so iPhone photos are correctly
        # oriented (EXIF tag 274 rotates/flips many phone captures).
        img = ImageOps.exif_transpose(img) or img
        img.load()
        clean = PILImage.new(img.mode, img.size)
        clean.paste(img)

    if clean.mode != "RGB":
        clean = clean.convert("RGB")
    return clean


def _select_gallery_for_phone_eval(
    manifest: pd.DataFrame,
    splits: dict[str, str],
    extra_album_ids: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Select the canonical gallery image per test-split album, plus any extras.

    Mirrors the logic in ``evaluate_local_features._select_gallery_images``:
    picks the highest-resolution image per album.  Expands the gallery with
    ``extra_album_ids`` not already in the test split (e.g. phone photos whose
    album is in train/val).  This mirrors ``_mode_sample`` gallery expansion
    so that all matched phone photos have a corresponding gallery entry.

    Args:
        manifest: Full manifest DataFrame.
        splits: Album ID -> split name mapping.
        extra_album_ids: Optional additional album IDs to include in gallery
            (e.g. albums referenced by phone photos outside the test split).

    Returns:
        ``(gallery_paths_str, gallery_album_ids)`` — path strings and album IDs
        for the canonical image per selected album.
    """
    test_album_ids = {aid for aid, s in splits.items() if s == "test"}
    all_album_ids = set(test_album_ids)
    if extra_album_ids:
        all_album_ids |= extra_album_ids

    selected_df = manifest[manifest["album_id"].isin(all_album_ids)].copy()

    gallery_paths: list[str] = []
    gallery_album_ids: list[str] = []

    for album_id, group in selected_df.groupby("album_id"):
        if len(group) == 1:
            row = group.iloc[0]
        else:
            res = group[["width", "height"]].max(axis=1)
            best_pos = int(res.argmax())
            row = group.iloc[best_pos]
        gallery_paths.append(str(row["image_path"]))
        gallery_album_ids.append(str(album_id))

    return gallery_paths, gallery_album_ids


def _append_summary_csv(
    results_dir: Path,
    model_id: str,
    timestamp: str,
    metrics: dict[str, object],
    num_gallery: int,
    num_queries: int,
    summary_csv_name: str = "phone_eval_summary.csv",
) -> None:
    """Append a row to a phone evaluation summary CSV.

    Args:
        results_dir: Top-level results directory.
        model_id: Model identifier.
        timestamp: Run timestamp string.
        metrics: Metrics dict (retrieval sub-dict).
        num_gallery: Number of gallery items.
        num_queries: Number of query items.
        summary_csv_name: Filename within ``results_dir``.  Defaults to
            ``"phone_eval_summary.csv"``; use
            ``"phone_sample_eval_summary.csv"`` for sample-mode runs.
    """
    summary_path = results_dir / summary_csv_name
    write_header = not summary_path.exists()

    retrieval = metrics.get("retrieval", metrics)
    if not isinstance(retrieval, dict):
        retrieval = {}

    row: dict[str, object] = {
        "model_id": model_id,
        "timestamp": timestamp,
        "recall_at_1": retrieval.get("recall_at_1", ""),
        "recall_at_5": retrieval.get("recall_at_5", ""),
        "map_at_5": retrieval.get("map_at_5", ""),
        "mrr": retrieval.get("mrr", ""),
        "num_gallery": num_gallery,
        "num_queries": num_queries,
    }

    fieldnames = [
        "model_id",
        "timestamp",
        "recall_at_1",
        "recall_at_5",
        "map_at_5",
        "mrr",
        "num_gallery",
        "num_queries",
    ]
    with summary_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _tta_aug(img: PILImage.Image, aug_idx: int) -> PILImage.Image:
    """Apply a fixed, deterministic augmentation by index for TTA.

    Index 0 is always the identity (original image). Indices 1-4 apply mild
    augmentations targeting the glare, blur, and exposure variance seen in
    real phone photos of physical album covers.

    Args:
        img: PIL Image to augment.
        aug_idx: Augmentation index (0 = identity, 1-4 = mild augmentations).

    Returns:
        Augmented PIL Image (same object for index 0; new object otherwise).
    """
    if aug_idx == 0:
        return img
    if aug_idx == 1:
        return TF.hflip(img)  # type: ignore[return-value, arg-type]
    if aug_idx == 2:
        return TF.rotate(img, angle=5, interpolation=transforms.InterpolationMode.BILINEAR)  # type: ignore[return-value, arg-type]
    if aug_idx == 3:
        return TF.rotate(img, angle=-5, interpolation=transforms.InterpolationMode.BILINEAR)  # type: ignore[return-value, arg-type]
    if aug_idx == 4:
        return TF.adjust_brightness(img, brightness_factor=1.15)  # type: ignore[return-value, arg-type]
    return img


def _embed_single_image(
    img: PILImage.Image,
    model: EmbeddingModel,
    tfm: transforms.Compose,
) -> NDArray[np.float32]:
    """Embed one PIL image with a tensor-based EmbeddingModel.

    Args:
        img: PIL Image (RGB, EXIF-stripped).
        model: EmbeddingModel to use.
        tfm: Model-specific preprocessing transform.

    Returns:
        L2-normalized float32 embedding of shape ``(D,)``.
    """
    tensor: torch.Tensor = tfm(img)  # type: ignore[assignment]
    return model.embed(tensor.unsqueeze(0)).numpy()[0].astype(np.float32)


def _embed_single_image_a3(img: PILImage.Image) -> NDArray[np.float32]:
    """Embed one PIL image using Apple FeaturePrint (A3) via a temp file.

    Writes the EXIF-stripped image to a temporary JPEG, runs
    ``VNGenerateImageFeaturePrintRequest``, then deletes the file.
    The returned vector is L2-normalized.

    Args:
        img: EXIF-stripped PIL Image to embed.

    Returns:
        L2-normalized float32 FeaturePrint embedding of shape ``(D,)``.

    Raises:
        ImportError: If ``pyobjc-framework-Vision`` is not installed.
        RuntimeError: If the Vision request fails.
    """
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        img.save(tmp_path, format="JPEG", quality=95)
        vec = extract_feature_vector(tmp_path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
    norm = float(np.linalg.norm(vec))
    return (vec / norm if norm > 0 else vec).astype(np.float32)


def _embed_with_tta(
    img: PILImage.Image,
    model_id: str,
    model: EmbeddingModel | None,
    tfm: transforms.Compose | None,
    n_augs: int,
) -> NDArray[np.float32]:
    """Embed a query image with test-time augmentation (TTA).

    Applies ``n_augs`` deterministic augmentations (including the original at
    index 0), embeds each view, and returns the L2-normalised mean embedding.
    Supports both tensor-based models and A3-featureprint (macOS-only).

    Args:
        img: EXIF-stripped PIL Image to embed.
        model_id: Model identifier (e.g. ``"A4-sscd"`` or ``"A3-featureprint"``).
        model: EmbeddingModel instance; ``None`` for A3-featureprint.
        tfm: Model preprocessing transform; ``None`` for A3-featureprint.
        n_augs: Number of augmented views including the original (1 = no TTA).

    Returns:
        L2-normalized float32 mean embedding of shape ``(D,)``.
    """
    is_a3 = model_id == FEATUREPRINT_MODEL_ID
    embeddings: list[NDArray[np.float32]] = []

    for aug_idx in range(n_augs):
        aug_img = _tta_aug(img, aug_idx)
        if is_a3:
            emb = _embed_single_image_a3(aug_img)
        else:
            if model is None or tfm is None:
                raise ValueError("model and tfm are required for non-A3 models")
            emb = _embed_single_image(aug_img, model, tfm)
        embeddings.append(emb)

    mean_emb: NDArray[np.float32] = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
    norm = float(np.linalg.norm(mean_emb))
    return mean_emb / norm if norm > 0 else mean_emb


def _apply_alpha_qe(
    query_matrix: NDArray[np.float32],
    gallery_matrix: NDArray[np.float32],
    k: int,
    alpha: float,
) -> NDArray[np.float32]:
    """Apply alpha query expansion (alpha-QE) to the query embedding matrix.

    For each query, blends it with the L2-normalised mean of its top-k gallery
    neighbours, then re-normalises. This shifts each query towards the dense
    cluster of its nearest neighbours, reducing the domain gap between phone
    photos and catalogue images.

    Reference: Chum et al., "Total Recall II: Query Expansion Revisited",
    CVPR 2011.

    Args:
        query_matrix: L2-normalised query embeddings, shape ``(Q, D)``.
        gallery_matrix: L2-normalised gallery embeddings, shape ``(G, D)``.
        k: Number of top gallery neighbours to blend per query.
        alpha: Blend weight for the gallery mean (0 = no change, 1 = full).

    Returns:
        Updated query matrix of shape ``(Q, D)`` with L2-normalised rows.
    """
    # Compute initial similarities for QE retrieval
    scores: NDArray[np.float32] = query_matrix @ gallery_matrix.T
    expanded = np.empty_like(query_matrix)

    for qi in range(len(query_matrix)):
        top_k_idx = np.argsort(-scores[qi])[:k]
        gallery_mean = gallery_matrix[top_k_idx].mean(axis=0)
        blended = query_matrix[qi] + alpha * gallery_mean
        norm = float(np.linalg.norm(blended))
        expanded[qi] = (blended / norm if norm > 0 else query_matrix[qi]).astype(np.float32)

    return expanded


def evaluate_model(
    model_id: str,
    test_complete_matched_csv: Path,
    test_complete_dir: Path,
    data_dir: Path,
    gallery_paths_str: list[str],
    gallery_album_ids: list[str],
    gallery_root: Path,
    results_dir: Path,
    timestamp: str,
    match_score_min: int,
    eval_set: str = _EVAL_SET_COMPLETE,
    model: object = None,  # pre-loaded EmbeddingModel; loaded lazily if None
    use_tta: bool = False,
    n_tta_augs: int = 5,
    use_alpha_qe: bool = False,
    alpha_qe_k: int = 5,
    alpha_qe_alpha: float = 0.5,
) -> dict[str, object]:
    """Evaluate a single embedding model against real-world phone photos.

    Supports all models in ``_PHONE_EVAL_MODEL_IDS`` including A3-featureprint
    (macOS-only, Apple Vision framework) and the SSCD disc_blur variant (A5).
    Optional inference-time enhancements:

    - **TTA** (``use_tta``): embeds ``n_tta_augs`` deterministic augmentations
      of each query image and returns the L2-normalised mean embedding.
    - **Alpha-QE** (``use_alpha_qe``): after initial query embedding, blends
      each query with the mean of its top-k gallery neighbours before re-ranking.

    Args:
        model_id: Model identifier (e.g. ``"A4-sscd"``, ``"A3-featureprint"``).
        test_complete_matched_csv: Path to the matched photos CSV.
        test_complete_dir: Directory containing the phone photo files.
        data_dir: Data directory with pre-computed gallery embeddings.
        gallery_paths_str: Ordered list of gallery image path strings.
        gallery_album_ids: Album IDs aligned with gallery_paths_str.
        gallery_root: Absolute gallery root path used to resolve relative
            ``image_path`` strings for on-the-fly gallery embedding.
        results_dir: Top-level results directory.
        timestamp: ISO-style timestamp string.
        match_score_min: Minimum fuzzy-match score to include a photo.
        eval_set: ``"complete"`` (default) or ``"sample"``.
        model: Pre-loaded EmbeddingModel; loaded lazily if ``None``.
            Ignored for A3-featureprint.
        use_tta: If ``True``, average embeddings from ``n_tta_augs`` views.
        n_tta_augs: Number of TTA views including original. Default 5.
        use_alpha_qe: If ``True``, apply alpha query expansion before ranking.
        alpha_qe_k: Number of top gallery neighbours to blend. Default 5.
        alpha_qe_alpha: Blend weight for the gallery mean. Default 0.5.

    Returns:
        Metrics dict with ``"retrieval"`` sub-dict.
    """
    logger.info(
        "evaluate_model_start",
        model_id=model_id,
        use_tta=use_tta,
        n_tta_augs=n_tta_augs if use_tta else None,
        use_alpha_qe=use_alpha_qe,
        alpha_qe_k=alpha_qe_k if use_alpha_qe else None,
        alpha_qe_alpha=alpha_qe_alpha if use_alpha_qe else None,
    )

    is_a3 = model_id == FEATUREPRINT_MODEL_ID

    # Derive a descriptive run label encoding model + inference settings
    run_label = model_id
    if use_tta:
        run_label += f"-tta{n_tta_augs}"
    if use_alpha_qe:
        run_label += f"-aqe{alpha_qe_k}a{alpha_qe_alpha:.1f}"

    # ── Load pre-computed gallery embeddings ──────────────────────────────
    hint = (
        "Run embed_featureprint.py --config ... first"
        if is_a3
        else f"Run embed_gallery.py --model {model_id} --split test first"
    )
    try:
        gallery_result = load_embeddings(data_dir, model_id)
    except FileNotFoundError:
        logger.error(
            "gallery_embeddings_not_found",
            model_id=model_id,
            data_dir=str(data_dir),
            hint=hint,
        )
        raise

    # Build path → embedding lookup
    path_to_emb: dict[str, NDArray[np.float32]] = {
        p: gallery_result.embeddings[i].astype(np.float32)
        for i, p in enumerate(gallery_result.image_paths)
    }

    # ── Load model (tensor-based models only; A3 uses Vision framework) ───
    loaded_model: EmbeddingModel | None = None
    tfm: transforms.Compose | None = None
    if not is_a3:
        loaded_model = (
            model  # type: ignore[assignment]
            if isinstance(model, EmbeddingModel)
            else _create_model(model_id)
        )
        tfm = loaded_model.get_transforms()

    # ── Align gallery embeddings with canonical gallery order ─────────────
    # Extra albums (not in pre-computed embeddings) are embedded on-the-fly.
    gallery_embeddings: list[NDArray[np.float32]] = []
    valid_gallery_paths: list[str] = []
    valid_gallery_album_ids: list[str] = []
    n_cached = 0
    n_live = 0

    for gpath, gaid in zip(gallery_paths_str, gallery_album_ids, strict=True):
        emb = path_to_emb.get(gpath)
        if emb is not None:
            n_cached += 1
        else:
            # Extra album not in pre-computed embeddings — embed on-the-fly.
            gpath_resolved = Path(gpath) if Path(gpath).is_absolute() else gallery_root / gpath
            try:
                if is_a3:
                    vec = extract_feature_vector(gpath_resolved)
                    norm = float(np.linalg.norm(vec))
                    emb = (vec / norm if norm > 0 else vec).astype(np.float32)
                else:
                    if loaded_model is None or tfm is None:
                        raise ValueError("model and tfm must be set for non-A3 gallery embedding")
                    with PILImage.open(gpath_resolved) as img:
                        img.load()
                        clean = PILImage.new(img.mode, img.size)
                        clean.paste(img)
                    clean = clean.convert("RGB")
                    t: torch.Tensor = tfm(clean)  # type: ignore[assignment]
                    emb = loaded_model.embed(t.unsqueeze(0)).numpy()[0].astype(np.float32)
                n_live += 1
            except (OSError, ValueError, RuntimeError, ImportError) as exc:
                logger.debug("extra_gallery_embed_error", path=str(gpath_resolved), error=str(exc))
                continue
        gallery_embeddings.append(emb)
        valid_gallery_paths.append(gpath)
        valid_gallery_album_ids.append(gaid)

    logger.info(
        "gallery_embeddings_ready",
        model_id=model_id,
        n_cached=n_cached,
        n_live=n_live,
        total=len(gallery_embeddings),
    )

    if not gallery_embeddings:
        raise ValueError(f"No gallery embeddings matched for model {model_id}")

    gallery_matrix = np.array(gallery_embeddings, dtype=np.float32)
    num_gallery = len(valid_gallery_album_ids)
    unique_albums_gallery = sorted(set(valid_gallery_album_ids))
    album_to_label = {aid: i for i, aid in enumerate(unique_albums_gallery)}
    gallery_labels = np.array(
        [album_to_label[aid] for aid in valid_gallery_album_ids], dtype=np.intp
    )

    logger.info("gallery_loaded", model_id=model_id, num_gallery=num_gallery)

    # ── Load phone photo matched CSV ──────────────────────────────────────
    sample_df = pd.read_csv(test_complete_matched_csv)
    matched = sample_df[
        (sample_df["file_exists"] == True)  # noqa: E712
        & (sample_df["matched_album_id"].notna())
        & (sample_df["matched_album_id"].astype(str).str.len() > 0)
        & (sample_df["match_score"] >= match_score_min)
    ].copy()

    if len(matched) == 0:
        logger.error("no_matched_phone_photos", min_score=match_score_min)
        raise ValueError("No matched phone photos found")

    logger.info(
        "phone_photos_loaded",
        model_id=model_id,
        num_photos=len(matched),
        min_score=match_score_min,
    )

    # ── Embed phone photos (with in-memory EXIF stripping) ────────────────
    query_embeddings: list[NDArray[np.float32]] = []
    query_labels_list: list[int] = []
    t0 = time.monotonic()

    for qi, (_, row) in enumerate(matched.iterrows()):
        album_id = str(row["matched_album_id"])
        filename = str(row["filename"])
        photo_path = test_complete_dir / filename

        if album_id not in album_to_label:
            logger.debug("phone_album_not_in_gallery", album_id=album_id)
            continue

        try:
            img = _load_phone_photo_stripped(photo_path)
        except (OSError, ValueError) as exc:
            logger.warning("photo_load_error", filename=filename, error=str(exc))
            continue

        if use_tta:
            emb = _embed_with_tta(img, model_id, loaded_model, tfm, n_tta_augs)
        elif is_a3:
            emb = _embed_single_image_a3(img)
        else:
            if loaded_model is None or tfm is None:
                raise ValueError("model and tfm must be set for non-A3 query embedding")
            emb = _embed_single_image(img, loaded_model, tfm)

        query_embeddings.append(emb)
        query_labels_list.append(album_to_label[album_id])

        if (qi + 1) % 50 == 0 or (qi + 1) == len(matched):
            elapsed = time.monotonic() - t0
            logger.info(
                "phone_embed_progress",
                model_id=model_id,
                done=f"{qi + 1}/{len(matched)}",
                elapsed_s=round(elapsed, 1),
            )

    if not query_embeddings:
        raise ValueError(f"No valid phone photo embeddings produced for model {model_id}")

    query_matrix = np.array(query_embeddings, dtype=np.float32)
    query_labels = np.array(query_labels_list, dtype=np.intp)
    num_queries = len(query_labels)
    total_elapsed = time.monotonic() - t0

    # ── Alpha-QE: shift queries towards gallery cluster before ranking ────
    if use_alpha_qe:
        logger.info(
            "alpha_qe_start",
            model_id=model_id,
            k=alpha_qe_k,
            alpha=alpha_qe_alpha,
        )
        query_matrix = _apply_alpha_qe(query_matrix, gallery_matrix, alpha_qe_k, alpha_qe_alpha)

    # ── Cosine similarity + retrieval metrics ─────────────────────────────
    score_matrix = query_matrix @ gallery_matrix.T
    ret: RetrievalMetrics = compute_retrieval_metrics(score_matrix, query_labels, gallery_labels)
    metrics: dict[str, float | int] = ret.to_dict()
    metrics["latency_per_query_s"] = round(total_elapsed / num_queries, 3)

    logger.info(
        "model_results",
        run_label=run_label,
        R_at_1=round(float(metrics["recall_at_1"]), 4),
        R_at_5=round(float(metrics["recall_at_5"]), 4),
        mAP_at_5=round(float(metrics["map_at_5"]), 4),
        MRR=round(float(metrics["mrr"]), 4),
        num_queries=num_queries,
        num_gallery=num_gallery,
    )

    # ── Save results ──────────────────────────────────────────────────────
    run_dir_suffix = "phone-sample" if eval_set == _EVAL_SET_SAMPLE else "phone"
    run_dir = results_dir / f"{run_label}-{run_dir_suffix}" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    eval_type = f"phone_photos_{eval_set}"
    out_metrics: dict[str, object] = {
        "model_id": model_id,
        "run_label": run_label,
        "eval_type": eval_type,
        "eval_set": eval_set,
        "timestamp": timestamp,
        "git_sha": _get_git_sha(),
        "num_gallery": num_gallery,
        "num_queries": num_queries,
        "retrieval": dict(metrics),
    }
    with (run_dir / "metrics.json").open("w") as f:
        json.dump(out_metrics, f, indent=2)

    out_config: dict[str, object] = {
        "model_id": model_id,
        "run_label": run_label,
        "eval_type": eval_type,
        "eval_set": eval_set,
        "timestamp": timestamp,
        "match_score_min": match_score_min,
        "num_gallery": num_gallery,
        "num_queries": num_queries,
        "git_sha": _get_git_sha(),
        "use_tta": use_tta,
        "n_tta_augs": n_tta_augs if use_tta else None,
        "use_alpha_qe": use_alpha_qe,
        "alpha_qe_k": alpha_qe_k if use_alpha_qe else None,
        "alpha_qe_alpha": alpha_qe_alpha if use_alpha_qe else None,
    }
    with (run_dir / "config.json").open("w") as f:
        json.dump(out_config, f, indent=2)

    summary_csv_name = (
        "phone_sample_eval_summary.csv"
        if eval_set == _EVAL_SET_SAMPLE
        else "phone_eval_summary.csv"
    )
    _append_summary_csv(
        results_dir,
        run_label,
        timestamp,
        {"retrieval": dict(metrics)},
        num_gallery,
        num_queries,
        summary_csv_name=summary_csv_name,
    )

    set_label = f"phone-{eval_set}"
    print(
        f"\n{run_label} ({set_label}):\n"
        f"  R@1={float(metrics['recall_at_1']):.3f}  "
        f"R@5={float(metrics['recall_at_5']):.3f}  "
        f"mAP@5={float(metrics['map_at_5']):.3f}  "
        f"MRR={float(metrics['mrr']):.3f}\n"
        f"  queries={num_queries}  gallery={num_gallery}  "
        f"lat={float(metrics['latency_per_query_s']):.2f}s/query\n"
        f"  Results: {run_dir}\n"
    )

    return out_metrics


def main(argv: list[str] | None = None) -> None:
    """Entry point for phone photo evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate global embedding models against labeled phone photos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="Path to dataset.yaml config file (default: configs/dataset.yaml).",
    )
    parser.add_argument(
        "--eval-set",
        type=str,
        default=_EVAL_SET_COMPLETE,
        choices=[_EVAL_SET_COMPLETE, _EVAL_SET_SAMPLE],
        help=(
            "Evaluation set: 'complete' (default, 203-photo real-world set, "
            "appends to phone_eval_summary.csv) or 'sample' (55-photo sample "
            "matching C2 sample-mode dataset, story #53, appends to "
            "phone_sample_eval_summary.csv)."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=("Single model ID to evaluate (e.g. A4-sscd). Mutually exclusive with --models."),
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Comma-separated list of model IDs to evaluate in sequence. "
            "Mutually exclusive with --model."
        ),
    )
    parser.add_argument(
        "--match-score-min",
        type=int,
        default=_DEFAULT_MATCH_SCORE_MIN,
        help=f"Minimum fuzzy-match score for photo inclusion (default: {_DEFAULT_MATCH_SCORE_MIN}).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Top-level results directory (default: results/).",
    )
    # ── TTA flags ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--tta",
        action="store_true",
        default=False,
        help=(
            "Enable test-time augmentation (TTA): average embeddings from "
            "--tta-n-augs deterministic views of each query image. Directly "
            "targets glare, blur and exposure variance in phone captures."
        ),
    )
    parser.add_argument(
        "--tta-n-augs",
        type=int,
        default=5,
        help=(
            "Number of TTA views including the original (default: 5). "
            "Views are: original, hflip, +5° rotation, -5° rotation, "
            "brightness +15%%. Only used when --tta is set."
        ),
    )
    # ── Alpha-QE flags ───────────────────────────────────────────────────────
    parser.add_argument(
        "--alpha-qe",
        action="store_true",
        default=False,
        help=(
            "Enable alpha query expansion: blend each query embedding with the "
            "mean of its top-k gallery neighbours before re-ranking. "
            "See Chum et al., CVPR 2011."
        ),
    )
    parser.add_argument(
        "--alpha-qe-k",
        type=int,
        default=5,
        help="Top-k gallery neighbours to blend per query for alpha-QE (default: 5).",
    )
    parser.add_argument(
        "--alpha-qe-alpha",
        type=float,
        default=0.5,
        help="Blend weight for the gallery mean in alpha-QE (default: 0.5).",
    )
    args = parser.parse_args(argv)
    eval_set: str = args.eval_set

    # ── Validate TTA / alpha-QE args ─────────────────────────────────────────
    if args.tta and args.tta_n_augs < 1:
        parser.error("--tta-n-augs must be >= 1")
    if args.alpha_qe and args.alpha_qe_k < 1:
        parser.error("--alpha-qe-k must be >= 1")
    if args.alpha_qe and not (0.0 < args.alpha_qe_alpha <= 2.0):
        parser.error("--alpha-qe-alpha must be in (0, 2]")

    # ── Resolve model list ──────────────────────────────────────────────────
    if args.model is not None and args.models is not None:
        parser.error("--model and --models are mutually exclusive")

    if args.model is not None:
        model_ids = [args.model]
    elif args.models is not None:
        model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        # Default: evaluate all tensor-based models (A3 excluded — macOS only, opt-in)
        model_ids = list(ALL_MODEL_IDS)

    for mid in model_ids:
        if mid not in _PHONE_EVAL_MODEL_IDS:
            parser.error(f"Unknown model ID '{mid}'. Valid IDs: {', '.join(_PHONE_EVAL_MODEL_IDS)}")

    # ── Load config ───────────────────────────────────────────────────
    config_path: Path = args.config.resolve()
    if not config_path.exists():
        logger.error("config_not_found", path=str(config_path))
        sys.exit(1)

    with config_path.open() as f:
        config = yaml.safe_load(f)

    config_dir = config_path.parent
    gallery_root = Path(str(config["paths"]["gallery_root"]))
    if not gallery_root.is_absolute():
        gallery_root = (config_dir / gallery_root).resolve()

    data_dir = (config_dir / config["paths"]["output_dir"]).resolve()
    manifest_path = data_dir / "manifest.parquet"
    splits_path = data_dir / "splits.json"

    # ── Resolve eval-set-specific paths ─────────────────────────────────
    if eval_set == _EVAL_SET_SAMPLE:
        # Sample mode: same dataset as C2 sample-mode (#53)
        photo_dir_key = "test_sample"
        matched_csv_name = "test_sample_matched.csv"
    else:
        # Complete mode: 203-photo real-world set (default)
        photo_dir_key = "test_complete"
        matched_csv_name = "test_complete_matched.csv"

    raw_photo_dir = config["paths"].get(photo_dir_key)
    if raw_photo_dir is None:
        logger.error(
            "config_key_missing",
            key=f"paths.{photo_dir_key}",
            config=str(config_path),
        )
        sys.exit(1)
    photo_dir = Path(str(raw_photo_dir))
    if not photo_dir.is_absolute():
        photo_dir = (config_dir / photo_dir).resolve()

    matched_csv = data_dir / matched_csv_name

    results_dir: Path = args.results_dir.resolve() if args.results_dir else Path.cwd() / "results"
    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())

    # ── Validate inputs ──────────────────────────────────────────────────
    for path, label in [
        (manifest_path, "manifest.parquet"),
        (splits_path, "splits.json"),
        (matched_csv, matched_csv_name),
        (photo_dir, photo_dir_key + " directory"),
    ]:
        if not path.exists():
            logger.error("required_path_missing", label=label, path=str(path))
            sys.exit(1)

    # ── Build gallery path/album_id list ────────────────────────────────────
    manifest = load_manifest(manifest_path)
    splits = load_splits(splits_path)

    # Pre-load matched CSV to find albums outside the test split that need to be
    # added to the gallery.  This mirrors _mode_sample gallery expansion so
    # that ALL matched phone photos have a retrievable gallery entry.
    pre_matched = pd.read_csv(matched_csv)
    pre_matched_filtered = pre_matched[
        (pre_matched["file_exists"] == True)  # noqa: E712
        & pre_matched["matched_album_id"].notna()
        & (pre_matched["matched_album_id"].astype(str).str.len() > 0)
        & (pre_matched["match_score"] >= args.match_score_min)
    ]
    test_album_ids_in_splits = {aid for aid, s in splits.items() if s == "test"}
    extra_album_ids: set[str] = {
        str(aid)
        for aid in pre_matched_filtered["matched_album_id"].unique()
        if str(aid) not in test_album_ids_in_splits
    }
    if extra_album_ids:
        logger.info(
            "extra_albums_for_gallery",
            n=len(extra_album_ids),
            hint="phone photos whose albums are outside the test split",
        )

    gallery_paths_str, gallery_album_ids = _select_gallery_for_phone_eval(
        manifest, splits, extra_album_ids=extra_album_ids
    )

    logger.info(
        "gallery_built",
        eval_set=eval_set,
        num_gallery=len(gallery_paths_str),
        num_test_albums=len(test_album_ids_in_splits),
        num_extra_albums=len(extra_album_ids),
    )

    # ── Evaluate each model ─────────────────────────────────────────────
    all_results: list[dict[str, object]] = []
    for model_id in model_ids:
        try:
            result = evaluate_model(
                model_id=model_id,
                test_complete_matched_csv=matched_csv,
                test_complete_dir=photo_dir,
                data_dir=data_dir,
                gallery_paths_str=gallery_paths_str,
                gallery_album_ids=gallery_album_ids,
                gallery_root=gallery_root,
                results_dir=results_dir,
                timestamp=timestamp,
                match_score_min=args.match_score_min,
                eval_set=eval_set,
                use_tta=args.tta,
                n_tta_augs=args.tta_n_augs,
                use_alpha_qe=args.alpha_qe,
                alpha_qe_k=args.alpha_qe_k,
                alpha_qe_alpha=args.alpha_qe_alpha,
            )
            all_results.append(result)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("model_eval_failed", model_id=model_id, error=str(exc))
            print(f"\n[SKIP] {model_id}: {exc}\n", file=sys.stderr)

    # ── Final summary ──────────────────────────────────────────────────
    if len(all_results) > 1:
        set_label = f"phone-{eval_set}"
        print("\n" + "=" * 60)
        print(f"PHONE PHOTO EVAL SUMMARY ({set_label})")
        print("=" * 60)
        for r in all_results:
            rlabel = r.get("run_label", r["model_id"])
            ret = r.get("retrieval", {})
            if isinstance(ret, dict):
                print(
                    f"  {rlabel!s:<35}  "
                    f"R@1={ret.get('recall_at_1', 0):.3f}  "
                    f"R@5={ret.get('recall_at_5', 0):.3f}  "
                    f"mAP@5={ret.get('map_at_5', 0):.3f}"
                )
        print("=" * 60 + "\n")
        summary_csv_name = (
            "phone_sample_eval_summary.csv"
            if eval_set == _EVAL_SET_SAMPLE
            else "phone_eval_summary.csv"
        )
        print(f"  Summary CSV: {results_dir / summary_csv_name}\n")


if __name__ == "__main__":
    main()
