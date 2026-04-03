#!/usr/bin/env python3
"""Evaluate global embedding models against real-world phone photos (story #15).

Loads the 243 labeled iPhone captures from ``test_complete/``, strips all EXIF
metadata in-memory, embeds each photo with the given model, and computes
retrieval metrics against the test-split gallery embeddings.

Privacy guardrail: EXIF is stripped in-memory on load. No photo paths are
written to any output file. Only aggregate metrics are reported.

Usage::

    python scripts/evaluate_phone_photos.py \\
        --config configs/dataset-remote.yaml \\
        --model A4-sscd

    python scripts/evaluate_phone_photos.py \\
        --config configs/dataset.yaml \\
        --models A1-dinov2-cls,A1-dinov2-gem,A2-openclip,A4-sscd \\
        --batch-size 32
"""

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
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

from vinylid_ml.dataset import load_manifest, load_splits
from vinylid_ml.eval_metrics import RetrievalMetrics, compute_retrieval_metrics
from vinylid_ml.gallery import load_embeddings
from vinylid_ml.models import ALL_MODEL_IDS, EmbeddingModel
from vinylid_ml.models import create_model as _create_model

logger = structlog.get_logger()

# Minimum fuzzy-match score for a phone photo to be included in evaluation.
_DEFAULT_MATCH_SCORE_MIN: int = 60


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
) -> None:
    """Append a row to ``results/phone_eval_summary.csv``.

    Args:
        results_dir: Top-level results directory.
        model_id: Model identifier.
        timestamp: Run timestamp string.
        metrics: Metrics dict (retrieval sub-dict).
        num_gallery: Number of gallery items.
        num_queries: Number of query items.
    """
    summary_path = results_dir / "phone_eval_summary.csv"
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


def evaluate_model(
    model_id: str,
    test_complete_matched_csv: Path,
    test_complete_dir: Path,
    data_dir: Path,
    gallery_paths_str: list[str],
    gallery_album_ids: list[str],
    results_dir: Path,
    timestamp: str,
    match_score_min: int,
    model: object = None,  # pre-loaded model; loaded lazily if None
) -> dict[str, object]:
    """Evaluate a single global embedding model against phone photos.

    Args:
        model_id: Model identifier (e.g. ``"A4-sscd"``).
        test_complete_matched_csv: Path to ``test_complete_matched.csv``.
        test_complete_dir: Directory containing the phone photo files.
        data_dir: Data directory with pre-computed gallery embeddings.
        gallery_paths_str: Ordered list of gallery image path strings.
        gallery_album_ids: Album IDs aligned with gallery_paths_str.
        results_dir: Top-level results directory.
        timestamp: ISO-style timestamp string.
        match_score_min: Minimum fuzzy-match score to include a photo.

    Returns:
        Metrics dict with ``"retrieval"`` sub-dict.
    """
    logger.info("evaluate_model_start", model_id=model_id)

    # ── Load pre-computed gallery embeddings ──────────────────────────────
    try:
        gallery_result = load_embeddings(data_dir, model_id)
    except FileNotFoundError:
        logger.error(
            "gallery_embeddings_not_found",
            model_id=model_id,
            data_dir=str(data_dir),
            hint=f"Run embed_gallery.py --model {model_id} --split test first",
        )
        raise

    # Build path → embedding lookup
    path_to_emb: dict[str, NDArray[np.float32]] = {
        p: gallery_result.embeddings[i].astype(np.float32)
        for i, p in enumerate(gallery_result.image_paths)
    }

    # ── Load model (needed for on-the-fly gallery embedding of extra albums) ─
    loaded_model: EmbeddingModel = (
        model  # type: ignore[assignment]
        if isinstance(model, EmbeddingModel)
        else _create_model(model_id)
    )
    tfm = loaded_model.get_transforms()

    # Align gallery embeddings with our canonical gallery order.
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
            try:
                with PILImage.open(gpath) as img:
                    img.load()
                    clean = PILImage.new(img.mode, img.size)
                    clean.paste(img)
                clean = clean.convert("RGB")
                t: torch.Tensor = tfm(clean)  # type: ignore[assignment]
                emb = loaded_model.embed(t.unsqueeze(0)).numpy()[0].astype(np.float32)
                n_live += 1
            except (OSError, ValueError, RuntimeError) as exc:
                logger.debug("extra_gallery_embed_error", path=gpath, error=str(exc))
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
            # Phone photo's album not in test-split gallery — skip
            logger.debug("phone_album_not_in_gallery", album_id=album_id)
            continue

        try:
            # Strip EXIF in-memory — never modify original
            img = _load_phone_photo_stripped(photo_path)
        except (OSError, ValueError) as exc:
            logger.warning("photo_load_error", filename=filename, error=str(exc))
            continue

        tensor: torch.Tensor = tfm(img)  # type: ignore[assignment]
        batch = tensor.unsqueeze(0)
        emb = loaded_model.embed(batch).numpy()[0].astype(np.float32)

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

    # ── Cosine similarity + metrics ──────────────────────────────────────────────
    score_matrix = query_matrix @ gallery_matrix.T
    ret: RetrievalMetrics = compute_retrieval_metrics(score_matrix, query_labels, gallery_labels)
    metrics: dict[str, float | int] = ret.to_dict()
    metrics["latency_per_query_s"] = round(total_elapsed / num_queries, 3)

    logger.info(
        "model_results",
        model_id=model_id,
        R_at_1=round(float(metrics["recall_at_1"]), 4),
        R_at_5=round(float(metrics["recall_at_5"]), 4),
        mAP_at_5=round(float(metrics["map_at_5"]), 4),
        MRR=round(float(metrics["mrr"]), 4),
        num_queries=num_queries,
        num_gallery=num_gallery,
    )

    # ── Save results ──────────────────────────────────────────────────────
    run_dir = results_dir / f"{model_id}-phone" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    out_metrics: dict[str, object] = {
        "model_id": model_id,
        "eval_type": "phone_photos",
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
        "eval_type": "phone_photos",
        "timestamp": timestamp,
        "match_score_min": match_score_min,
        "num_gallery": num_gallery,
        "num_queries": num_queries,
        "git_sha": _get_git_sha(),
    }
    with (run_dir / "config.json").open("w") as f:
        json.dump(out_config, f, indent=2)

    _append_summary_csv(
        results_dir,
        model_id,
        timestamp,
        {"retrieval": dict(metrics)},
        num_gallery,
        num_queries,
    )

    print(
        f"\n{model_id} (phone photos):\n"
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
        "--model",
        type=str,
        default=None,
        help=(
            "Single model ID to evaluate (e.g. A4-sscd). "
            "Mutually exclusive with --models."
        ),
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
    args = parser.parse_args(argv)

    # ── Resolve model list ────────────────────────────────────────────────
    if args.model is not None and args.models is not None:
        parser.error("--model and --models are mutually exclusive")

    if args.model is not None:
        model_ids = [args.model]
    elif args.models is not None:
        model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        # Default: evaluate all global embedding models
        model_ids = list(ALL_MODEL_IDS)

    for mid in model_ids:
        if mid not in ALL_MODEL_IDS:
            parser.error(
                f"Unknown model ID '{mid}'. Valid IDs: {', '.join(ALL_MODEL_IDS)}"
            )

    # ── Load config ───────────────────────────────────────────────────────
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

    test_complete_dir = Path(str(config["paths"]["test_complete"]))
    if not test_complete_dir.is_absolute():
        test_complete_dir = (config_dir / test_complete_dir).resolve()

    data_dir = (config_dir / config["paths"]["output_dir"]).resolve()
    manifest_path = data_dir / "manifest.parquet"
    splits_path = data_dir / "splits.json"
    test_complete_matched_csv = data_dir / "test_complete_matched.csv"

    results_dir: Path = args.results_dir.resolve() if args.results_dir else Path.cwd() / "results"
    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())

    # ── Validate inputs ───────────────────────────────────────────────────
    for path, label in [
        (manifest_path, "manifest.parquet"),
        (splits_path, "splits.json"),
        (test_complete_matched_csv, "test_complete_matched.csv"),
        (test_complete_dir, "test_complete directory"),
    ]:
        if not path.exists():
            logger.error("required_path_missing", label=label, path=str(path))
            sys.exit(1)

    # ── Build gallery path/album_id list ──────────────────────────────────
    manifest = load_manifest(manifest_path)
    splits = load_splits(splits_path)

    # Pre-load matched CSV to find albums outside the test split that need to be
    # added to the gallery.  This mirrors _mode_sample gallery expansion so
    # that ALL matched phone photos have a retrievable gallery entry.
    pre_matched = pd.read_csv(test_complete_matched_csv)
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
        num_gallery=len(gallery_paths_str),
        num_test_albums=len(test_album_ids_in_splits),
        num_extra_albums=len(extra_album_ids),
    )

    # ── Evaluate each model ──────────────────────────────────────────────
    all_results: list[dict[str, object]] = []
    for model_id in model_ids:
        try:
            result = evaluate_model(
                model_id=model_id,
                test_complete_matched_csv=test_complete_matched_csv,
                test_complete_dir=test_complete_dir,
                data_dir=data_dir,
                gallery_paths_str=gallery_paths_str,
                gallery_album_ids=gallery_album_ids,
                results_dir=results_dir,
                timestamp=timestamp,
                match_score_min=args.match_score_min,
            )
            all_results.append(result)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("model_eval_failed", model_id=model_id, error=str(exc))
            print(f"\n[SKIP] {model_id}: {exc}\n", file=sys.stderr)

    # ── Final summary ─────────────────────────────────────────────────────
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("PHONE PHOTO EVAL SUMMARY")
        print("=" * 60)
        for r in all_results:
            mid = r["model_id"]
            ret = r.get("retrieval", {})
            if isinstance(ret, dict):
                print(
                    f"  {mid:<25}  "
                    f"R@1={ret.get('recall_at_1', 0):.3f}  "
                    f"R@5={ret.get('recall_at_5', 0):.3f}  "
                    f"mAP@5={ret.get('map_at_5', 0):.3f}"
                )
        print("=" * 60 + "\n")
        print(f"  Summary CSV: {results_dir / 'phone_eval_summary.csv'}\n")


if __name__ == "__main__":
    main()
