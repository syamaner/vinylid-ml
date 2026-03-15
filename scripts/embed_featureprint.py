#!/usr/bin/env python3
"""Embed gallery images with Apple FeaturePrint (A3) and save results to disk.

A3-specific analogue of ``embed_gallery.py``.  Because
``VNGenerateImageFeaturePrintRequest`` works on image files rather than
pre-processed tensors, A3 cannot use the standard ``EmbeddingModel`` pipeline.
This script produces output in the same ``embeddings.npy`` / ``metadata.json``
format so ``evaluate.py`` can be run unchanged.

After this script completes, run:

    python scripts/evaluate.py --config configs/dataset.yaml \\
        --model A3-featureprint --split test

Usage:
    python scripts/embed_featureprint.py --config configs/dataset.yaml
    python scripts/embed_featureprint.py --config configs/dataset.yaml --split val
    python scripts/embed_featureprint.py --config configs/dataset.yaml --no-latency
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd
import structlog
import yaml

from vinylid_ml.apple_featureprint import (
    FEATUREPRINT_MODEL_ID,
    embed_images,
    measure_featureprint_latency,
)
from vinylid_ml.dataset import load_manifest, load_splits
from vinylid_ml.gallery import EmbeddingResult, save_embeddings

logger = structlog.get_logger()


def _filter_manifest_by_split(
    manifest: pd.DataFrame,
    splits: dict[str, str],
    split_name: str,
) -> pd.DataFrame:
    """Filter manifest to images in the specified split.

    Args:
        manifest: Full manifest DataFrame.
        splits: Album ID → split name mapping.
        split_name: ``"train"``, ``"val"``, ``"test"``, or ``"all"``.

    Returns:
        Filtered DataFrame.
    """
    if split_name == "all":
        return manifest
    album_ids_in_split = {album_id for album_id, s in splits.items() if s == split_name}
    mask = manifest["album_id"].astype(str).isin(album_ids_in_split)
    return manifest[mask].reset_index(drop=True)


def _append_latency_csv(
    results_dir: Path,
    model_id: str,
    latency: dict[str, float],
) -> Path:
    """Append a latency row to ``results/latency_summary.csv``.

    Creates the file with a header if it does not exist; otherwise appends.

    Args:
        results_dir: Top-level results directory.
        model_id: Model identifier string.
        latency: Dict with keys ``"p50_ms"``, ``"p95_ms"``, ``"p99_ms"``.

    Returns:
        Path to the CSV file.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "latency_summary.csv"
    fieldnames = ["model_id", "p50_ms", "p95_ms", "p99_ms"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({"model_id": model_id, **latency})
    return csv_path


def main(argv: list[str] | None = None) -> None:
    """Entry point for FeaturePrint embedding script."""
    parser = argparse.ArgumentParser(
        description="Embed gallery images with Apple FeaturePrint (A3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to dataset.yaml config file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Which split to embed (default: test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for embeddings (default: from config).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Results directory for latency CSV (default: results/).",
    )
    parser.add_argument(
        "--no-latency",
        action="store_true",
        help="Skip latency benchmarking.",
    )
    parser.add_argument(
        "--latency-runs",
        type=int,
        default=50,
        help="Number of timed latency runs (default: 50).",
    )
    args = parser.parse_args(argv)

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

    if args.output_dir is not None:
        output_dir: Path = args.output_dir.resolve()
    else:
        output_dir = (config_dir / config["paths"]["output_dir"]).resolve()

    results_dir: Path = (
        args.results_dir.resolve() if args.results_dir else Path.cwd() / "results"
    )

    manifest_path = output_dir / "manifest.parquet"
    splits_path = output_dir / "splits.json"

    logger.info(
        "embed_featureprint_start",
        split=args.split,
        output_dir=str(output_dir),
    )

    manifest = load_manifest(manifest_path)
    splits = load_splits(splits_path)
    filtered = _filter_manifest_by_split(manifest, splits, args.split)

    logger.info("manifest_filtered", split=args.split, num_images=len(filtered))

    if len(filtered) == 0:
        logger.warning("no_images_in_split", split=args.split)
        sys.exit(0)

    # Build image path list and aligned metadata
    raw_paths = filtered["image_path"].tolist()
    album_ids: list[str] = [str(x) for x in filtered["album_id"].tolist()]
    rel_paths: list[str] = [str(p) for p in raw_paths]  # type: ignore[reportUnknownArgumentType]

    image_paths: list[Path] = []
    for rel in rel_paths:
        p = Path(rel)
        if not p.is_absolute():
            p = gallery_root / p
        image_paths.append(p)

    # ── Embed ──────────────────────────────────────────────────────────────────
    t_start = time.monotonic()
    embeddings = embed_images(image_paths, show_progress=True)
    elapsed = time.monotonic() - t_start

    embedding_dim: int = embeddings.shape[1]
    result = EmbeddingResult(
        embeddings=embeddings,
        image_paths=rel_paths,
        album_ids=album_ids,
        model_id=FEATUREPRINT_MODEL_ID,
        embedding_dim=embedding_dim,
    )
    save_embeddings(result, output_dir)

    logger.info(
        "embed_complete",
        model_id=FEATUREPRINT_MODEL_ID,
        num_images=len(image_paths),
        embedding_dim=embedding_dim,
        total_time_s=round(elapsed, 2),
        throughput_ips=round(len(image_paths) / elapsed if elapsed > 0 else 0.0, 1),
    )

    # ── Latency benchmark ──────────────────────────────────────────────────────
    if not args.no_latency:
        sample_image = image_paths[0]
        logger.info(
            "latency_start",
            sample_image=str(sample_image),
            n_timed=args.latency_runs,
        )
        latency = measure_featureprint_latency(
            sample_image,
            n_warmup=5,
            n_timed=args.latency_runs,
        )
        latency_csv = _append_latency_csv(results_dir, FEATUREPRINT_MODEL_ID, latency)
        logger.info(
            "latency_complete",
            p50_ms=round(latency["p50_ms"], 2),
            p95_ms=round(latency["p95_ms"], 2),
            p99_ms=round(latency["p99_ms"], 2),
            csv_path=str(latency_csv),
        )
        print(
            f"\nA3 FeaturePrint latency:  "
            f"p50={latency['p50_ms']:.1f}ms  "
            f"p95={latency['p95_ms']:.1f}ms  "
            f"p99={latency['p99_ms']:.1f}ms"
        )

    print(
        f"\nEmbedding complete: {len(image_paths)} images, "
        f"dim={embedding_dim}, "
        f"total={elapsed:.1f}s "
        f"({len(image_paths) / elapsed:.1f} img/s)\n"
    )


if __name__ == "__main__":
    main()
