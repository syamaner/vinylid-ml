#!/usr/bin/env python3
"""Orchestrate zero-shot evaluation for all global embedding models.

Runs embed_gallery → single-image latency benchmark → evaluate for each model
in sequence, then prints a summary table and saves latency results.

Usage:
    python scripts/run_zero_shot_eval.py --config configs/dataset.yaml
    python scripts/run_zero_shot_eval.py --config configs/dataset.yaml --skip-embed
    python scripts/run_zero_shot_eval.py --config configs/dataset.yaml \\
        --models A1-dinov2-cls,A4-sscd
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import structlog
import torch
import yaml
from PIL import Image

from vinylid_ml.dataset import load_manifest, load_splits
from vinylid_ml.models import ALL_MODEL_IDS, EmbeddingModel, create_model

logger = structlog.get_logger()

_SCRIPTS_DIR: Path = Path(__file__).resolve().parent

__all__ = ["measure_single_image_latency"]


def _sync_device() -> None:
    """Synchronize the active accelerator to ensure async ops have completed.

    No-op on CPU. Required for accurate wall-clock latency measurement on
    devices with asynchronous execution (MPS, CUDA).
    """
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_single_image_latency(
    model: EmbeddingModel,
    image_path: Path,
    *,
    n_warmup: int = 10,
    n_timed: int = 100,
) -> dict[str, float]:
    """Measure single-image inference latency for a model.

    Runs ``n_warmup`` forward passes (discarded for warmup), then ``n_timed``
    timed passes. Reports p50, p95, p99 in milliseconds.

    Device synchronization (MPS/CUDA) is performed before and after each
    forward pass so that the timer captures real wall-clock latency rather
    than just the time to enqueue async operations.

    Args:
        model: Loaded EmbeddingModel instance (any pooling/resolution variant).
        image_path: Path to a sample image used for all timing runs.
        n_warmup: Number of warmup iterations (not timed). Default 10.
        n_timed: Number of timed iterations. Default 100.

    Returns:
        Dict with keys ``"p50_ms"``, ``"p95_ms"``, ``"p99_ms"`` (float ms).
    """
    transform = model.get_transforms()
    with Image.open(image_path) as img:
        tensor: torch.Tensor = transform(img.convert("RGB"))  # type: ignore[assignment]
    batch = tensor.unsqueeze(0)  # (1, C, H, W)

    for _ in range(n_warmup):
        model.embed(batch)
        _sync_device()

    times_ms: list[float] = []
    for _ in range(n_timed):
        _sync_device()
        t0 = time.perf_counter()
        model.embed(batch)
        _sync_device()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    n = len(times_ms)
    return {
        "p50_ms": times_ms[n // 2],
        "p95_ms": times_ms[max(0, int(n * 0.95) - 1)],
        "p99_ms": times_ms[max(0, int(n * 0.99) - 1)],
    }


def _pick_sample_image(
    manifest_path: Path,
    splits_path: Path,
    split: str,
    gallery_root: Path,
) -> Path:
    """Pick a single sample image from the split for latency benchmarking.

    Args:
        manifest_path: Path to manifest.parquet.
        splits_path: Path to splits.json.
        split: Split name (``"test"`` or ``"val"``).
        gallery_root: Root directory for resolving relative image paths.

    Returns:
        Absolute path to the first image in the split.

    Raises:
        ValueError: If no images are found in the split.
    """
    manifest = load_manifest(manifest_path)
    splits = load_splits(splits_path)
    album_ids_in_split = {aid for aid, s in splits.items() if s == split}
    split_manifest = manifest[manifest["album_id"].astype(str).isin(album_ids_in_split)]
    if len(split_manifest) == 0:
        raise ValueError(f"No images found in split '{split}'")
    row = split_manifest.iloc[0]
    image_path = Path(str(row["image_path"]))
    if not image_path.is_absolute():
        image_path = gallery_root / image_path
    return image_path


def _run_embed_gallery(
    config: Path,
    model_id: str,
    split: str,
    batch_size: int,
    output_dir: Path,
) -> None:
    """Run embed_gallery.py as a subprocess.

    Args:
        config: Path to dataset.yaml.
        model_id: Model identifier string.
        split: Split to embed (``"test"`` or ``"val"``).
        batch_size: Batch size for the embedding run.
        output_dir: Directory to write embeddings into.

    Raises:
        subprocess.CalledProcessError: If embed_gallery exits with non-zero code.
    """
    cmd = [
        sys.executable,
        str(_SCRIPTS_DIR / "embed_gallery.py"),
        "--config",
        str(config),
        "--model",
        model_id,
        "--split",
        split,
        "--batch-size",
        str(batch_size),
        "--output-dir",
        str(output_dir),
    ]
    logger.info("running_embed_gallery", model_id=model_id, split=split)
    subprocess.run(cmd, check=True)


def _run_evaluate(
    config: Path,
    model_id: str,
    split: str,
    embeddings_dir: Path,
    results_dir: Path,
) -> None:
    """Run evaluate.py as a subprocess.

    Args:
        config: Path to dataset.yaml.
        model_id: Model identifier string.
        split: Split to evaluate (``"test"`` or ``"val"``).
        embeddings_dir: Directory containing pre-computed embeddings.
        results_dir: Top-level results directory.

    Raises:
        subprocess.CalledProcessError: If evaluate exits with non-zero code.
    """
    cmd = [
        sys.executable,
        str(_SCRIPTS_DIR / "evaluate.py"),
        "--config",
        str(config),
        "--model",
        model_id,
        "--split",
        split,
        "--embeddings-dir",
        str(embeddings_dir),
        "--output-dir",
        str(results_dir),
    ]
    logger.info("running_evaluate", model_id=model_id, split=split)
    subprocess.run(cmd, check=True)


def _find_latest_metrics(results_dir: Path, model_id: str) -> dict[str, Any] | None:
    """Load the latest metrics.json for a model run.

    Args:
        results_dir: Top-level results directory (contains ``{model_id}/`` subdirs).
        model_id: Model identifier string.

    Returns:
        Parsed metrics dict, or ``None`` if no run directory exists.
    """
    model_results_dir = results_dir / model_id
    if not model_results_dir.exists():
        return None
    run_dirs = sorted(
        [d for d in model_results_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            with metrics_path.open() as f:
                result: dict[str, Any] = json.load(f)
            return result
    return None


def _save_latency_csv(results_dir: Path, latency_rows: list[dict[str, Any]]) -> Path:
    """Save per-model latency results to results/latency_summary.csv.

    Args:
        results_dir: Top-level results directory.
        latency_rows: List of dicts with keys ``model_id``, ``p50_ms``,
            ``p95_ms``, ``p99_ms``.

    Returns:
        Path to the written CSV file.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "latency_summary.csv"
    fieldnames = ["model_id", "p50_ms", "p95_ms", "p99_ms"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(latency_rows)
    return out_path


def _decision_verdict(recall_at_1: float) -> str:
    """Map Recall@1 to a production decision verdict (plan §6 thresholds).

    Args:
        recall_at_1: Recall@1 score in [0, 1].

    Returns:
        One of ``"zero-shot sufficient"``, ``"fine-tuning likely needed"``,
        or ``"fine-tuning required"``.
    """
    if recall_at_1 >= 0.90:
        return "zero-shot sufficient"
    if recall_at_1 >= 0.80:
        return "fine-tuning likely needed"
    return "fine-tuning required"


def _print_summary_table(
    model_ids: list[str],
    metrics_by_model: dict[str, dict[str, Any]],
    latency_by_model: dict[str, dict[str, float]],
) -> None:
    """Print a formatted summary table to stdout.

    Args:
        model_ids: Ordered list of model IDs to print.
        metrics_by_model: Mapping of model_id to metrics dict.
        latency_by_model: Mapping of model_id to latency dict.
    """
    header = f"{'Model':<25} {'R@1':>7} {'R@5':>7} {'mAP@5':>7} {'MRR':>7} {'p50ms':>7}  Verdict"
    print("\n" + "=" * 80)
    print("Zero-Shot Evaluation Summary")
    print("=" * 80)
    print(header)
    print("-" * 80)
    for model_id in model_ids:
        m = metrics_by_model.get(model_id)
        lat = latency_by_model.get(model_id, {})
        if m is None:
            print(f"{model_id:<25} {'N/A':>7}")
            continue
        retrieval: dict[str, Any] = {}
        raw = m.get("retrieval")
        if isinstance(raw, dict):
            retrieval = {str(k): v for k, v in raw.items()}  # type: ignore[reportUnknownVariableType]
        r1 = float(retrieval.get("recall_at_1") or 0)
        r5 = float(retrieval.get("recall_at_5") or 0)
        map5 = float(retrieval.get("map_at_5") or 0)
        mrr = float(retrieval.get("mrr") or 0)
        p50_str = f"{lat['p50_ms']:.1f}" if lat else "N/A"
        verdict = _decision_verdict(r1)
        print(
            f"{model_id:<25} {r1:>7.4f} {r5:>7.4f} {map5:>7.4f} {mrr:>7.4f} {p50_str:>7}  {verdict}"
        )
    print("=" * 80 + "\n")


def main(argv: list[str] | None = None) -> None:
    """Entry point for zero-shot evaluation orchestration."""
    parser = argparse.ArgumentParser(
        description="Orchestrate zero-shot evaluation for all global embedding models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to dataset.yaml.")
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(ALL_MODEL_IDS),
        help=(f"Comma-separated model IDs (default: all). Available: {', '.join(ALL_MODEL_IDS)}"),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Split to embed + evaluate (default: test).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for embedding (default: 64)."
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip embedding if embeddings.npy already exists for a model.",
    )
    parser.add_argument(
        "--latency-runs",
        type=int,
        default=100,
        help="Number of timed inference runs for latency benchmark (default: 100).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write embeddings (default: from config).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Results directory (default: results/).",
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

    results_dir: Path = args.results_dir.resolve() if args.results_dir else Path.cwd() / "results"
    # Derive manifest/splits from the same output_dir used by embed_gallery.py,
    # so --output-dir is honoured consistently across the pipeline.
    manifest_path = output_dir / "manifest.parquet"
    splits_path = output_dir / "splits.json"

    model_ids: list[str] = [m.strip() for m in args.models.split(",") if m.strip()]
    logger.info(
        "zero_shot_eval_start",
        models=model_ids,
        split=args.split,
        skip_embed=args.skip_embed,
        results_dir=str(results_dir),
    )

    sample_image = _pick_sample_image(manifest_path, splits_path, args.split, gallery_root)
    logger.info("latency_sample_image", path=str(sample_image))

    latency_rows: list[dict[str, Any]] = []
    metrics_by_model: dict[str, dict[str, Any]] = {}
    latency_by_model: dict[str, dict[str, float]] = {}

    for model_id in model_ids:
        logger.info("model_start", model_id=model_id)
        t_start = time.monotonic()

        # ── Embed gallery ──────────────────────────────────────────────
        embeddings_file = output_dir / model_id / "embeddings.npy"
        if args.skip_embed and embeddings_file.exists():
            logger.info("embed_skipped", model_id=model_id, reason="--skip-embed")
        else:
            _run_embed_gallery(config_path, model_id, args.split, args.batch_size, output_dir)

        # ── Latency benchmark ──────────────────────────────────────────
        logger.info("latency_start", model_id=model_id, n_timed=args.latency_runs)
        try:
            latency_model = create_model(model_id)
            latency = measure_single_image_latency(
                latency_model, sample_image, n_warmup=10, n_timed=args.latency_runs
            )
            del latency_model  # release memory before next model
            latency_by_model[model_id] = latency
            latency_rows.append({"model_id": model_id, **latency})
            logger.info(
                "latency_complete",
                model_id=model_id,
                p50_ms=round(latency["p50_ms"], 2),
                p95_ms=round(latency["p95_ms"], 2),
                p99_ms=round(latency["p99_ms"], 2),
            )
        except Exception as exc:
            logger.warning("latency_failed", model_id=model_id, error=str(exc))

        # ── Evaluate ───────────────────────────────────────────────────
        _run_evaluate(config_path, model_id, args.split, output_dir, results_dir)

        # ── Collect metrics ────────────────────────────────────────────
        m = _find_latest_metrics(results_dir, model_id)
        if m is not None:
            metrics_by_model[model_id] = m

        elapsed = time.monotonic() - t_start
        logger.info("model_complete", model_id=model_id, elapsed_s=round(elapsed, 1))

    # ── Save latency CSV ───────────────────────────────────────────────
    if latency_rows:
        latency_csv_path = _save_latency_csv(results_dir, latency_rows)
        logger.info("latency_csv_saved", path=str(latency_csv_path))

    # ── Print summary ──────────────────────────────────────────────────
    _print_summary_table(model_ids, metrics_by_model, latency_by_model)


if __name__ == "__main__":
    main()
