#!/usr/bin/env python3
"""Analyse A4-sscd failure cases on the 203-photo phone evaluation set.

Classifies each failure (rank-1 wrong) as either:
  - Degradation-recoverable: phone photo has high blur or glare, but the
    matched gallery cover has sufficient visual complexity. Gallery expansion
    with synthetic degradations should help.
  - Ambiguity-limited: phone photo is reasonably clear, but the matched
    gallery cover is minimalist / visually simple. No augmentation helps.

Privacy: EXIF is stripped in-memory. No photo paths or filenames are written
to any output. Only aggregate statistics are reported.

Usage::

    python scripts/analyse_failures.py --config configs/dataset.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image as PILImage
from PIL import ImageOps

# ---------------------------------------------------------------------------
# Inline metric helpers (no heavy deps beyond cv2 + numpy)
# ---------------------------------------------------------------------------


def _laplacian_var(img: PILImage.Image) -> float:
    """Blur proxy: Laplacian variance of the grayscale image.

    High = sharp detail.  Low = blurry.  Threshold ~200 distinguishes
    clearly sharp from noticeably blurry at typical phone capture sizes.
    """
    gray = np.array(img.convert("L"))  # uint8
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _glare_fraction(img: PILImage.Image, threshold: int = 230) -> float:
    """Glare proxy: fraction of pixels with luma above threshold.

    A glossy vinyl sleeve under room light or phone flash typically
    saturates >5% of pixels.
    """
    gray = np.array(img.convert("L"))
    return float((gray >= threshold).mean())


def _edge_density(img: PILImage.Image) -> float:
    """Cover complexity proxy: mean Sobel gradient magnitude.

    Low = minimalist (flat colour, text-only).
    High = textured / detailed artwork.
    Threshold ~15 separates simple from complex covers in practice.
    """
    gray = np.array(img.convert("L"))  # uint8
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.sqrt(sx**2 + sy**2).mean())


def _color_entropy(img: PILImage.Image) -> float:
    """Cover colour diversity proxy: mean Shannon entropy across RGB channels.

    Low entropy = few colours (minimalist). High = colourful / complex.
    """
    arr = np.array(img.convert("RGB"))
    entropy = 0.0
    for c in range(3):
        hist, _ = np.histogram(arr[:, :, c].ravel(), bins=64, range=(0, 256))
        hist = hist.astype(float)
        hist = hist[hist > 0] / hist.sum()
        entropy += -float((hist * np.log2(hist)).sum())
    return entropy / 3.0


def _load_stripped(path: Path) -> PILImage.Image:
    with PILImage.open(path) as img:
        img = ImageOps.exif_transpose(img) or img
        img.load()
        clean = PILImage.new(img.mode, img.size)
        clean.paste(img)
    if clean.mode != "RGB":
        clean = clean.convert("RGB")
    return clean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset.yaml"),
    )
    parser.add_argument(
        "--match-score-min",
        type=int,
        default=60,
    )
    # Thresholds for classification
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=200.0,
        help="Laplacian var below this = blurry (default 200)",
    )
    parser.add_argument(
        "--glare-threshold",
        type=float,
        default=0.05,
        help="Glare fraction above this = glarey (default 0.05)",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=15.0,
        help="Sobel mean above this = complex cover (default 15)",
    )
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    with config_path.open() as f:
        config = yaml.safe_load(f)

    config_dir = config_path.parent
    data_dir = (config_dir / config["paths"]["output_dir"]).resolve()
    photo_dir = Path(str(config["paths"]["test_complete"]))
    if not photo_dir.is_absolute():
        photo_dir = (config_dir / photo_dir).resolve()

    matched_csv = data_dir / "test_complete_matched.csv"
    manifest_path = data_dir / "manifest.parquet"
    splits_path = data_dir / "splits.json"

    for p in [matched_csv, manifest_path, splits_path, photo_dir]:
        if not p.exists():
            print(f"Missing: {p}", file=sys.stderr)
            sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────────────
    from vinylid_ml.dataset import load_manifest, load_splits
    from vinylid_ml.eval_metrics import compute_retrieval_metrics
    from vinylid_ml.models import SSCDEmbedder

    manifest = load_manifest(manifest_path)
    splits = load_splits(splits_path)

    raw = pd.read_csv(matched_csv)
    matched = raw[
        (raw["file_exists"] == True)  # noqa: E712
        & raw["matched_album_id"].notna()
        & (raw["matched_album_id"].astype(str).str.len() > 0)
        & (raw["match_score"] >= args.match_score_min)
    ].copy()
    matched["matched_album_id"] = matched["matched_album_id"].astype(str)

    print(f"\n{'=' * 60}")
    print(f"A4-sscd Failure Analysis — {len(matched)} phone photos")
    print(f"{'=' * 60}")

    # ── Build gallery ─────────────────────────────────────────────────────────
    test_aids = {aid for aid, s in splits.items() if s == "test"}
    extra_aids = {
        str(aid) for aid in matched["matched_album_id"].unique() if str(aid) not in test_aids
    }
    gallery_aids = test_aids | extra_aids

    gallery_manifest = manifest[manifest["album_id"].isin(gallery_aids)].copy()
    album_to_row: dict[str, pd.Series] = {}  # type: ignore[type-arg]
    for aid, group in gallery_manifest.groupby("album_id"):
        if len(group) == 1:
            album_to_row[str(aid)] = group.iloc[0]
        else:
            res = group[["width", "height"]].max(axis=1)
            album_to_row[str(aid)] = group.iloc[int(res.argmax())]

    gallery_album_ids = sorted(album_to_row.keys())
    gallery_paths = [Path(str(album_to_row[a]["image_path"])) for a in gallery_album_ids]
    unique_aids = sorted(set(gallery_album_ids))
    album_to_label = {a: i for i, a in enumerate(unique_aids)}
    gallery_labels = np.array([album_to_label[a] for a in gallery_album_ids], dtype=np.intp)

    # ── Load model + embed gallery ────────────────────────────────────────────
    print("\nLoading A4-sscd and embedding gallery (979 images)…")
    model = SSCDEmbedder()
    tfm = model.get_transforms()

    gallery_feats: list[np.ndarray] = []
    for i, gpath in enumerate(gallery_paths):
        img = _load_stripped(gpath)
        t = tfm(img)  # type: ignore[assignment]
        emb = model.embed(t.unsqueeze(0)).numpy()[0]  # type: ignore[arg-type]
        gallery_feats.append(emb.astype(np.float32))
        if (i + 1) % 100 == 0:
            print(f"  gallery {i + 1}/{len(gallery_paths)}")

    gallery_matrix = np.stack(gallery_feats)
    print(f"  gallery embedded: {gallery_matrix.shape}")

    # ── Embed phone photos + collect metrics ──────────────────────────────────
    print("\nEmbedding phone photos and computing metrics…")

    query_feats: list[np.ndarray] = []
    query_labels_list: list[int] = []
    # Per-query metric storage
    phone_blur: list[float] = []
    phone_glare: list[float] = []
    gallery_edge: list[float] = []  # matched gallery album's edge density
    gallery_entropy: list[float] = []

    for _, row in matched.iterrows():
        aid = str(row["matched_album_id"])
        if aid not in album_to_label:
            continue
        photo_path = photo_dir / str(row["filename"])
        try:
            img = _load_stripped(photo_path)
        except OSError:
            continue

        # Phone photo degradation metrics (on full-res before downscaling)
        phone_blur.append(_laplacian_var(img))
        phone_glare.append(_glare_fraction(img))

        # Gallery cover complexity metrics
        gallery_row = album_to_row[aid]
        gallery_img_path = Path(str(gallery_row["image_path"]))
        try:
            g_img = _load_stripped(gallery_img_path)
            gallery_edge.append(_edge_density(g_img))
            gallery_entropy.append(_color_entropy(g_img))
        except OSError:
            gallery_edge.append(float("nan"))
            gallery_entropy.append(float("nan"))

        # SSCD embedding for retrieval
        t = tfm(img)  # type: ignore[assignment]
        emb = model.embed(t.unsqueeze(0)).numpy()[0]  # type: ignore[arg-type]
        query_feats.append(emb.astype(np.float32))
        query_labels_list.append(album_to_label[aid])

    query_matrix = np.stack(query_feats)
    query_labels = np.array(query_labels_list, dtype=np.intp)

    # ── Retrieval + failure classification ───────────────────────────────────
    scores = query_matrix @ gallery_matrix.T
    top1_idx = np.argmax(scores, axis=1)
    top1_labels = gallery_labels[top1_idx]
    correct = top1_labels == query_labels

    n_total = len(query_labels)
    n_correct = int(correct.sum())
    n_fail = n_total - n_correct

    phone_blur_arr = np.array(phone_blur)
    phone_glare_arr = np.array(phone_glare)
    gallery_edge_arr = np.array(gallery_edge)
    gallery_entropy_arr = np.array(gallery_entropy)

    fail_mask = ~correct
    # Classify failures
    is_blurry = phone_blur_arr < args.blur_threshold
    is_glarey = phone_glare_arr > args.glare_threshold
    is_degraded = is_blurry | is_glarey  # phone photo problem
    is_complex_cover = gallery_edge_arr > args.edge_threshold  # cover has features

    fail_degraded_complex = fail_mask & is_degraded & is_complex_cover
    fail_degraded_simple = fail_mask & is_degraded & ~is_complex_cover
    fail_clear_complex = fail_mask & ~is_degraded & is_complex_cover
    fail_clear_simple = fail_mask & ~is_degraded & ~is_complex_cover

    # ── Print report ──────────────────────────────────────────────────────────
    ret = compute_retrieval_metrics(scores, query_labels, gallery_labels)
    metrics = ret.to_dict()

    print(f"\n{'─' * 60}")
    print(
        f"Overall: R@1={metrics['recall_at_1']:.4f}  R@5={metrics['recall_at_5']:.4f}  "
        f"MRR={metrics['mrr']:.4f}"
    )
    print(f"Correct: {n_correct}/{n_total}  |  Failures: {n_fail}/{n_total}")

    print(f"\n{'─' * 60}")
    print(f"FAILURE BREAKDOWN (n={n_fail})")
    print(f"{'─' * 60}")

    cats = [
        ("Degraded phone + complex cover → GALLERY EXPANSION CAN HELP", fail_degraded_complex),
        ("Degraded phone + simple cover  → augment + ambiguity compound", fail_degraded_simple),
        (
            "Clear phone  + complex cover   → retrieval failure / look-alike album",
            fail_clear_complex,
        ),
        ("Clear phone  + simple cover    → AMBIGUITY-LIMITED (hard ceiling)", fail_clear_simple),
    ]

    recoverable = int(fail_degraded_complex.sum())
    for label, mask in cats:
        n = int(mask.sum())
        pct = 100 * n / n_fail if n_fail > 0 else 0
        print(f"  {n:>3} ({pct:4.1f}%)  {label}")

    pct_recoverable = 100 * recoverable / n_fail if n_fail > 0 else 0
    print(
        f"\n  → {recoverable}/{n_fail} failures ({pct_recoverable:.0f}%) are "
        f"'degradation-recoverable' — gallery expansion target population"
    )

    # Phone photo metric distributions
    print(f"\n{'─' * 60}")
    print("PHONE PHOTO METRICS (failures vs. successes)")
    print(f"{'─' * 60}")
    for name, arr in [
        ("Blur (Laplacian var, higher=sharper)", phone_blur_arr),
        ("Glare fraction (%, higher=more glare)", phone_glare_arr * 100),
    ]:
        f_vals = arr[fail_mask]
        s_vals = arr[~fail_mask]
        print(f"\n  {name}")
        print(
            f"    Failures  — median={np.nanmedian(f_vals):.1f}  "
            f"p25={np.nanpercentile(f_vals, 25):.1f}  "
            f"p75={np.nanpercentile(f_vals, 75):.1f}"
        )
        print(
            f"    Successes — median={np.nanmedian(s_vals):.1f}  "
            f"p25={np.nanpercentile(s_vals, 25):.1f}  "
            f"p75={np.nanpercentile(s_vals, 75):.1f}"
        )

    print(f"\n{'─' * 60}")
    print("GALLERY COVER COMPLEXITY (matched albums of failures vs. successes)")
    print(f"{'─' * 60}")
    for name, arr in [
        ("Edge density (Sobel mean, higher=more detailed)", gallery_edge_arr),
        ("Colour entropy (bits, higher=more colourful)", gallery_entropy_arr),
    ]:
        f_vals = arr[fail_mask]
        s_vals = arr[~fail_mask]
        print(f"\n  {name}")
        print(
            f"    Failures  — median={np.nanmedian(f_vals):.2f}  "
            f"p25={np.nanpercentile(f_vals, 25):.2f}  "
            f"p75={np.nanpercentile(f_vals, 75):.2f}"
        )
        print(
            f"    Successes — median={np.nanmedian(s_vals):.2f}  "
            f"p25={np.nanpercentile(s_vals, 25):.2f}  "
            f"p75={np.nanpercentile(s_vals, 75):.2f}"
        )

    # Blur distribution across failure types
    print(f"\n{'─' * 60}")
    print("DEGRADATION PREVALENCE")
    print(f"{'─' * 60}")
    print(
        f"  Blurry photos  (Laplacian var < {args.blur_threshold:.0f}):  "
        f"{is_blurry[fail_mask].sum()} of {n_fail} failures  "
        f"({100 * is_blurry[fail_mask].mean():.0f}%)"
    )
    print(
        f"  Glarey photos  (glare > {args.glare_threshold * 100:.0f}% pixels):  "
        f"{is_glarey[fail_mask].sum()} of {n_fail} failures  "
        f"({100 * is_glarey[fail_mask].mean():.0f}%)"
    )
    print(
        f"  Any degradation:  "
        f"{is_degraded[fail_mask].sum()} of {n_fail} failures  "
        f"({100 * is_degraded[fail_mask].mean():.0f}%)"
    )
    print(
        f"  Simple covers  (edge density < {args.edge_threshold:.0f}):  "
        f"{(~is_complex_cover)[fail_mask].sum()} of {n_fail} failures  "
        f"({100 * (~is_complex_cover)[fail_mask].mean():.0f}%)"
    )

    print(f"\n{'=' * 60}")
    print("VERDICT")
    print(f"{'=' * 60}")
    if recoverable >= n_fail * 0.5:
        print("  ✅ Majority of failures are degradation-recoverable.")
        print("  → Gallery expansion with calibrated 2D augmentation is JUSTIFIED.")
        print(
            f"  → Expected addressable headroom: ~{recoverable} queries "
            f"= +{recoverable / n_total * 100:.1f}pp R@1 ceiling"
        )
    elif recoverable >= n_fail * 0.3:
        print("  ⚠️  Significant minority of failures are degradation-recoverable.")
        print("  → Gallery expansion may help but won't close the full gap.")
        print(
            f"  → Expected partial gain: ~{recoverable} queries "
            f"= +{recoverable / n_total * 100:.1f}pp R@1 ceiling (if all recovered)"
        )
    else:
        print("  ❌ Most failures are ambiguity-limited.")
        print("  → Gallery expansion is unlikely to help significantly.")
        print("  → The bottleneck is cover visual ambiguity, not photo degradation.")
    print()


if __name__ == "__main__":
    main()
