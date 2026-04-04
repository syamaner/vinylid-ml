#!/usr/bin/env python3
"""Push Sprint 3 evaluation artifacts to the Hugging Face Hub (story #23).

Publishes safe-to-share artifacts to ``syamaner/vinylid-eval`` (model repo).

What is uploaded
----------------
- Comparison reports: ``comparison.csv``, ``comparison.html``,
  ``multi_context_comparison.csv``, ``multi_context_comparison.html``
- Per-run metrics: ``results/{model_id}/{timestamp}/metrics.json``
- Per-run configs: ``results/{model_id}/{timestamp}/config.json``

What is NOT uploaded (privacy & safety guardrails)
---------------------------------------------------
- ``per_query.csv`` files — may contain phone photo filenames
- Directories matching ``*-phone*/`` or ``*-phone-sample*/`` — real-world photo
  evaluation outputs; only aggregate summaries (metrics.json, config.json) from
  these dirs ARE uploaded since they contain no per-image paths
- Model checkpoints (large files belong on HF as model weights, not eval repos)
- Any file matching ``*.pt``, ``*.pth``, ``*.bin``, ``*.onnx``, ``*.mlpackage``

Usage::

    python scripts/push_to_hub.py
    python scripts/push_to_hub.py --results-dir results/ --dry-run
    python scripts/push_to_hub.py --repo syamaner/vinylid-eval --commit-message "Sprint 3 results"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog

logger = structlog.get_logger()

# ── Constants ────────────────────────────────────────────────────────────────

_DEFAULT_REPO_ID: str = "syamaner/vinylid-eval"

# Comparison report filenames (from compare_models.py)
_COMPARISON_FILES: list[str] = [
    "comparison.csv",
    "comparison.html",
    "multi_context_comparison.csv",
    "multi_context_comparison.html",
    "phone_eval_summary.csv",
    "phone_sample_eval_summary.csv",
    "summary.csv",
]

# Per-run files that are safe to upload (aggregate metrics only, no photo paths)
_SAFE_RUN_FILES: list[str] = ["metrics.json", "config.json"]

# File extensions that must never be uploaded (model weights / blobs)
_BLOCKED_EXTENSIONS: frozenset[str] = frozenset(
    {".pt", ".pth", ".bin", ".onnx", ".safetensors", ".npy", ".npz"}
)

# per_query.csv is blocked regardless of directory (may contain photo filenames)
_BLOCKED_FILENAMES: frozenset[str] = frozenset({"per_query.csv"})


def _collect_upload_pairs(
    results_dir: Path,
    repo_prefix: str = "results",
) -> list[tuple[Path, str]]:
    """Collect (local_path, hub_path) pairs for safe-to-upload files.

    Args:
        results_dir: Local top-level results directory.
        repo_prefix: Path prefix within the HF repo (default: ``"results"``).

    Returns:
        List of ``(local_path, hub_path)`` tuples to upload.
    """
    pairs: list[tuple[Path, str]] = []

    # ── Comparison report files ───────────────────────────────────────────
    for fname in _COMPARISON_FILES:
        local = results_dir / fname
        if local.exists():
            pairs.append((local, f"{repo_prefix}/{fname}"))
        else:
            logger.debug("comparison_file_not_found", file=fname)

    # ── Per-run metrics and configs ───────────────────────────────────────
    # Walk results/{model_id}/{timestamp}/{metrics.json,config.json}
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        # Skip the run directories themselves if they contain blocked content,
        # but still allow uploading metrics.json / config.json inside them.
        for ts_dir in sorted(model_dir.iterdir()):
            if not ts_dir.is_dir():
                continue
            for fname in _SAFE_RUN_FILES:
                local = ts_dir / fname
                if not local.exists():
                    continue
                # Paranoia: double-check extension is allowed
                if local.suffix.lower() in _BLOCKED_EXTENSIONS:
                    logger.warning("blocked_extension_skipped", path=str(local))
                    continue
                if local.name in _BLOCKED_FILENAMES:
                    logger.warning("blocked_filename_skipped", path=str(local))
                    continue
                rel = local.relative_to(results_dir)
                pairs.append((local, f"{repo_prefix}/{rel}"))

    return pairs


def push_to_hub(
    results_dir: Path,
    repo_id: str,
    commit_message: str,
    dry_run: bool = False,
) -> None:
    """Upload Sprint 3 evaluation artifacts to the Hugging Face Hub.

    Args:
        results_dir: Local top-level results directory.
        repo_id: HF Hub repository ID (e.g. ``"syamaner/vinylid-eval"``).
        commit_message: Commit message for the upload.
        dry_run: If True, print what would be uploaded without actually uploading.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
        SystemExit: If results directory does not exist.
    """
    if not results_dir.exists():
        logger.error("results_dir_not_found", path=str(results_dir))
        sys.exit(1)

    pairs = _collect_upload_pairs(results_dir)
    if not pairs:
        logger.warning("no_artifacts_found", results_dir=str(results_dir))
        print("No artifacts found to upload.")
        return

    print(f"\nPreparing to upload {len(pairs)} file(s) to {repo_id}:")
    for local, hub_path in pairs:
        size_kb = local.stat().st_size / 1024
        print(f"  {hub_path}  ({size_kb:.1f} KB)")

    if dry_run:
        print("\n[DRY RUN] No files uploaded.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error(
            "huggingface_hub_not_installed",
            hint="pip install huggingface-hub",
        )
        sys.exit(1)

    api = HfApi()

    # Verify the repo exists and the user has write access before uploading.
    try:
        repo_info = api.repo_info(repo_id=repo_id, repo_type="model")
        logger.info("repo_found", repo_id=repo_id, sha=repo_info.sha)
    except Exception as exc:
        logger.error("repo_not_accessible", repo_id=repo_id, error=str(exc))
        print(
            f"\nError: Cannot access repo '{repo_id}'. "
            "Ensure you are logged in (`huggingface-cli login`) and the repo exists."
        )
        sys.exit(1)

    # Upload files one by one so progress is visible and partial uploads succeed.
    n_uploaded = 0
    n_failed = 0
    for local, hub_path in pairs:
        try:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=hub_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
            )
            logger.info("uploaded", local=str(local), hub_path=hub_path)
            n_uploaded += 1
        except Exception as exc:
            logger.error("upload_failed", local=str(local), error=str(exc))
            n_failed += 1

    print(
        f"\nUpload complete: {n_uploaded} uploaded, {n_failed} failed.\n"
        f"View at: https://huggingface.co/{repo_id}/tree/main\n"
    )
    if n_failed > 0:
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    """Entry point for HF Hub push script (story #23)."""
    parser = argparse.ArgumentParser(
        description=(
            "Push Sprint 3 evaluation artifacts to the Hugging Face Hub "
            f"(default repo: {_DEFAULT_REPO_ID})."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Top-level results directory (default: results/).",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=_DEFAULT_REPO_ID,
        help=f"Hugging Face Hub repository ID (default: {_DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Sprint 3: add evaluation artifacts",
        help="Commit message for the upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually uploading.",
    )
    args = parser.parse_args(argv)

    push_to_hub(
        results_dir=args.results_dir.resolve(),
        repo_id=args.repo,
        commit_message=args.commit_message,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
