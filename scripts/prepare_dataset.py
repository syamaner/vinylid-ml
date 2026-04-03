"""Build dataset manifest, album-level splits, and integrate real-world test sets.

Usage:
    python scripts/prepare_dataset.py [--config configs/dataset.yaml]

Steps:
    4a. Walk album-export/, extract EXIF → manifest.parquet
    4b. Album-level train/val/test split
    4c. Parse real-world test CSVs, fuzzy-match to gallery albums
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import time
from collections import defaultdict
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import structlog
import yaml
from thefuzz import fuzz

from vinylid_ml.exif import ExifExtractionError, extract_metadata

logger = structlog.get_logger()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}


def build_manifest(
    gallery_root: Path,
    db_path: Path | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Walk gallery directory, extract EXIF from every image, build manifest.

    Args:
        gallery_root: Root directory containing artist/album/image structure.
        db_path: Optional path to SQLite DB for CoverType cross-reference.

    Returns:
        DataFrame with one row per image.
    """
    logger.info("building_manifest", gallery_root=str(gallery_root))

    # Load CoverType lookup from SQLite if available
    cover_types: dict[str, str] = {}
    if db_path and db_path.exists():
        cover_types = _load_cover_types(db_path)
        logger.info("cover_types_loaded", num_entries=len(cover_types))

    records: list[dict[str, str | int]] = []
    errors: list[dict[str, str]] = []
    total_files = 0

    for artist_dir in sorted(gallery_root.iterdir()):
        if not artist_dir.is_dir():
            continue
        for album_dir in sorted(artist_dir.iterdir()):
            if not album_dir.is_dir():
                continue
            for image_file in sorted(album_dir.iterdir()):
                if image_file.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                total_files += 1

                try:
                    meta = extract_metadata(image_file)
                except (ExifExtractionError, FileNotFoundError, OSError, ValueError) as e:
                    errors.append(
                        {
                            "image_path": str(image_file),
                            "error": str(e),
                        }
                    )
                    continue

                # Deterministic album_id from directory structure
                album_id = _hash_album_id(artist_dir.name, album_dir.name)

                # Look up CoverType from DB using filename pattern
                # Filename: {release_uuid}-{caa_image_id}.{ext}
                cover_type = _match_cover_type(meta.release_id, image_file.name, cover_types)

                records.append(
                    {
                        "image_path": str(image_file),
                        "release_id": meta.release_id,
                        "artist": meta.artist,
                        "album": meta.album,
                        "artist_dir": artist_dir.name,
                        "album_dir": album_dir.name,
                        "album_id": album_id,
                        "width": meta.width,
                        "height": meta.height,
                        "format": meta.image_format,
                        "cover_type": cover_type,
                    }
                )

                if total_files % 5000 == 0:
                    logger.info("progress", processed=total_files, records=len(records))

    df = pd.DataFrame(records)

    logger.info(
        "manifest_built",
        total_files=total_files,
        valid_records=len(records),
        errors=len(errors),
        num_albums=df["album_id"].nunique() if len(df) > 0 else 0,
        num_artists=df["artist_dir"].nunique() if len(df) > 0 else 0,
    )

    if errors:
        error_df = pd.DataFrame(errors)
        logger.warning("extraction_errors", count=len(errors), sample=errors[:3])
        # Save error log for debugging
        if output_dir is not None:
            error_path = output_dir / "manifest_errors.csv"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_df.to_csv(error_path, index=False)

    return df


def build_splits(
    manifest: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
    prefer_single_to_train: bool = True,
) -> dict[str, str]:
    """Split albums into train/val/test sets.

    Split at album level to prevent data leakage. Single-image albums
    preferentially go to train since val/test need gallery+query pairs.

    Args:
        manifest: DataFrame from build_manifest().
        train_ratio: Fraction of albums for training.
        val_ratio: Fraction of albums for validation.
        seed: Random seed for reproducibility.
        prefer_single_to_train: Whether to prioritize single-image albums for train.

    Returns:
        Mapping of album_id -> split name.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Count images per album
    album_counts: dict[str, int] = manifest.groupby("album_id").size().to_dict()  # type: ignore[assignment]
    all_album_ids: list[str] = sorted(album_counts.keys())

    if prefer_single_to_train:
        single_albums: list[str] = [aid for aid in all_album_ids if album_counts[aid] == 1]
        multi_albums: list[str] = [aid for aid in all_album_ids if album_counts[aid] > 1]
    else:
        single_albums = []
        multi_albums = list(all_album_ids)

    # Stratify multi-image albums by artist for balanced splits
    artist_albums: dict[str, list[str]] = defaultdict(list)
    album_artist = manifest.drop_duplicates("album_id").set_index("album_id")["artist_dir"]
    for aid in multi_albums:
        artist = str(album_artist.get(aid, "unknown"))
        artist_albums[artist].append(aid)

    # Shuffle within each artist group
    for albums in artist_albums.values():
        random.shuffle(albums)

    # Flatten with round-robin across artists for stratification
    artists_sorted = sorted(artist_albums.keys())
    multi_shuffled: list[str] = []
    max_len = max((len(v) for v in artist_albums.values()), default=0)
    for i in range(max_len):
        for artist in artists_sorted:
            albums = artist_albums[artist]
            if i < len(albums):
                multi_shuffled.append(albums[i])

    # Determine split sizes
    total = len(all_album_ids)
    n_val = max(1, int(total * val_ratio))
    n_test = max(1, int(total * (1.0 - train_ratio - val_ratio)))
    n_val_test = n_val + n_test

    # Assign multi-image albums to val/test first (they have gallery+query pairs)
    splits: dict[str, str] = {}

    val_test_pool = multi_shuffled[:n_val_test]
    val_albums = set(val_test_pool[:n_val])
    test_albums = set(val_test_pool[n_val:n_val_test])
    train_multi = multi_shuffled[n_val_test:]

    for aid in val_albums:
        splits[aid] = "val"
    for aid in test_albums:
        splits[aid] = "test"
    for aid in train_multi:
        splits[aid] = "train"
    for aid in single_albums:
        splits[aid] = "train"

    # Verify
    counts: dict[str, int] = defaultdict(int)
    for s in splits.values():
        counts[s] += 1

    logger.info(
        "splits_created",
        total_albums=len(splits),
        train=counts["train"],
        val=counts["val"],
        test=counts["test"],
        single_image_in_train=len(single_albums),
    )

    return splits


def integrate_test_set(
    csv_path: Path,
    test_dir: Path,
    manifest: pd.DataFrame,
    name: str = "complete",
) -> pd.DataFrame:
    """Parse real-world test CSV and fuzzy-match to gallery albums.

    Args:
        csv_path: Path to test_data.csv.
        test_dir: Directory containing the actual test photo files.
        manifest: Gallery manifest for fuzzy matching.
        name: Label for this test set (e.g., "complete", "sample").

    Returns:
        DataFrame with columns: filename, artist, album_name, release_url,
        matched_album_id, match_score, file_exists.
    """
    logger.info("integrating_test_set", csv_path=str(csv_path), name=name)

    df = pd.read_csv(csv_path)

    # Extract just the filename from Windows paths
    df["filename"] = df["FilePath"].apply(
        lambda p: PureWindowsPath(str(p)).name  # type: ignore[reportUnknownLambdaType]
    )

    # Check which files actually exist in the test directory
    existing_files = {f.name for f in test_dir.iterdir() if f.is_file()}
    df["file_exists"] = df["filename"].isin(existing_files)

    # Build gallery lookup: (artist_lower, album_lower) -> album_id
    gallery_lookup: dict[tuple[str, str], str] = {}
    album_info = manifest.drop_duplicates("album_id")[["artist", "album", "album_id"]]
    for _, row in album_info.iterrows():
        key = (str(row["artist"]).lower().strip(), str(row["album"]).lower().strip())
        gallery_lookup[key] = row["album_id"]

    # Fuzzy match each test entry to gallery
    matched_ids: list[str | None] = []
    match_scores: list[int] = []

    for _, row in df.iterrows():
        artist = str(row.get("Artist", "")).strip()
        album = str(row.get("AlbumName", "")).strip()

        if not album:
            matched_ids.append(None)
            match_scores.append(0)
            continue

        best_id, best_score = _fuzzy_match_album(artist, album, gallery_lookup)
        matched_ids.append(best_id)
        match_scores.append(best_score)

    df["matched_album_id"] = matched_ids
    df["match_score"] = match_scores

    n_matched = sum(1 for m in matched_ids if m is not None)
    n_empty_album = sum(1 for a in df["AlbumName"] if not str(a).strip())
    n_missing_files = sum(1 for e in df["file_exists"] if not e)

    logger.info(
        "test_set_integrated",
        name=name,
        total_entries=len(df),
        matched=n_matched,
        unmatched=len(df) - n_matched,
        empty_album_name=n_empty_album,
        missing_files=n_missing_files,
        high_confidence=sum(1 for s in match_scores if s >= 90),
    )

    return df


def print_eda_stats(manifest: pd.DataFrame, splits: dict[str, str]) -> None:
    """Print exploratory data analysis statistics."""
    print("\n" + "=" * 60)
    print("DATASET EDA SUMMARY")
    print("=" * 60)

    print(f"\nTotal images: {len(manifest):,}")
    print(f"Total albums: {manifest['album_id'].nunique():,}")
    print(f"Total artists: {manifest['artist_dir'].nunique():,}")

    # Image count distribution
    album_counts = manifest.groupby("album_id").size()
    print("\nImages per album:")
    print(f"  Mean: {album_counts.mean():.1f}")
    print(f"  Median: {album_counts.median():.0f}")
    print(f"  Max: {album_counts.max()}")
    print(f"  1 image: {(album_counts == 1).sum():,} ({(album_counts == 1).mean():.1%})")
    print(f"  2-5 images: {((album_counts >= 2) & (album_counts <= 5)).sum():,}")
    print(f"  6+ images: {(album_counts >= 6).sum():,}")

    # Resolution distribution
    width_arr = np.asarray(manifest["width"].values, dtype=np.int64)
    height_arr = np.asarray(manifest["height"].values, dtype=np.int64)
    max_dim = np.maximum(width_arr, height_arr)
    print("\nImage resolution (max dimension):")
    print(f"  Min: {max_dim.min()}px")
    print(f"  Median: {int(np.median(max_dim))}px")
    print(f"  Max: {max_dim.max()}px")
    print(f"  <400px: {(max_dim < 400).sum():,}")
    print(f"  400-800px: {((max_dim >= 400) & (max_dim <= 800)).sum():,}")
    print(f"  800+px: {(max_dim > 800).sum():,}")

    # Format distribution
    fmt_counts = manifest["format"].value_counts()
    print("\nImage formats:")
    for fmt, count in fmt_counts.items():
        print(f"  {fmt}: {count:,} ({count / len(manifest):.1%})")

    # Cover type distribution
    if "cover_type" in manifest.columns:
        ct_counts = manifest["cover_type"].value_counts()
        print("\nCover types:")
        for ct, count in ct_counts.items():
            print(f"  {ct}: {count:,}")

    # Split sizes
    print("\nSplit distribution:")
    for split_name in ("train", "val", "test"):
        album_ids = {aid for aid, s in splits.items() if s == split_name}
        n_images = manifest[manifest["album_id"].isin(album_ids)].shape[0]
        print(f"  {split_name}: {len(album_ids):,} albums, {n_images:,} images")

    print("=" * 60 + "\n")


# --- Private helpers ---


def _hash_album_id(artist_dir: str, album_dir: str) -> str:
    """Create a deterministic album ID from directory names."""
    key = f"{artist_dir}/{album_dir}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _load_cover_types(db_path: Path) -> dict[str, str]:
    """Load CoverType from SQLite Covers table.

    Returns mapping of (release_id_lower, image_filename_stem) -> CoverType.
    We key on release_id since that's what we extract from EXIF.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT ReleaseId, CoverType, Path FROM Covers")
        rows = cursor.fetchall()
    except sqlite3.OperationalError as exc:
        logger.warning(
            "covers_table_unavailable",
            db_path=str(db_path),
            error=str(exc),
            hint="CoverType metadata will be set to 'Unknown' for all images",
        )
        rows = []
    finally:
        conn.close()

    result: dict[str, str] = {}
    for release_id, cover_type, path_str in rows:
        if release_id and cover_type:
            # Build lookup key from the filename in the DB path
            if path_str:
                filename = PureWindowsPath(path_str).stem.lower()
                result[filename] = cover_type
            # Also store by release_id for simpler matching
            result[release_id.lower()] = cover_type

    return result


def _match_cover_type(release_id: str, filename: str, cover_types: dict[str, str]) -> str:
    """Try to match an image to a CoverType from the DB.

    Args:
        release_id: MusicBrainz release UUID from EXIF.
        filename: Image filename (e.g., "uuid-imageid.jpg").
        cover_types: Lookup dict from _load_cover_types().

    Returns:
        CoverType string ("Front", "Back", etc.) or "Unknown".
    """
    if not cover_types:
        return "Unknown"

    # Try filename stem match first (most specific)
    stem = Path(filename).stem.lower()
    if stem in cover_types:
        return cover_types[stem]

    # Try release_id match (less specific — may have multiple covers)
    if release_id.lower() in cover_types:
        return cover_types[release_id.lower()]

    return "Unknown"


def _fuzzy_match_album(
    artist: str,
    album: str,
    gallery_lookup: dict[tuple[str, str], str],
    threshold: int = 70,
) -> tuple[str | None, int]:
    """Fuzzy match (artist, album) to gallery albums.

    Args:
        artist: Artist name from test CSV.
        album: Album name from test CSV.
        gallery_lookup: Mapping of (artist_lower, album_lower) -> album_id.
        threshold: Minimum combined fuzzy score to accept a match.

    Returns:
        Tuple of (matched_album_id or None, match_score).
    """
    artist_lower = artist.lower().strip()
    album_lower = album.lower().strip()

    best_score = 0
    best_id: str | None = None

    for (g_artist, g_album), album_id in gallery_lookup.items():
        # Combined score: weighted average of artist + album similarity
        artist_score: int = fuzz.ratio(artist_lower, g_artist)  # type: ignore[reportUnknownMemberType]
        album_score: int = fuzz.ratio(album_lower, g_album)  # type: ignore[reportUnknownMemberType]
        # Album match is more important — weight it higher
        combined = int(artist_score * 0.3 + album_score * 0.7)

        if combined > best_score:
            best_score = combined
            best_id = album_id

    if best_score >= threshold:
        return best_id, best_score
    return None, best_score


def main() -> None:
    """Main entry point for dataset preparation."""
    parser = argparse.ArgumentParser(description="Build dataset manifest and splits")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="Path to dataset configuration YAML",
    )
    args = parser.parse_args()

    # Load config
    with args.config.open() as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    split_config = config["splits"]

    # Resolve relative paths against the config file's directory
    config_dir = args.config.resolve().parent

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (config_dir / path).resolve()

    gallery_root = _resolve(paths["gallery_root"])
    db_path = _resolve(paths["sqlite_db"])
    output_dir = _resolve(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    # Step 4a: Build manifest
    manifest = build_manifest(gallery_root, db_path, output_dir=output_dir)
    manifest_path = output_dir / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    logger.info("manifest_saved", path=str(manifest_path))

    # Step 4b: Build splits
    splits = build_splits(
        manifest,
        train_ratio=split_config["train_ratio"],
        val_ratio=split_config["val_ratio"],
        seed=split_config["seed"],
        prefer_single_to_train=split_config.get("prefer_single_image_to_train", True),
    )
    splits_path = output_dir / "splits.json"
    with splits_path.open("w") as f:
        json.dump(splits, f, indent=2)
    logger.info("splits_saved", path=str(splits_path))

    # Step 4c: Integrate real-world test sets
    test_complete_dir = _resolve(paths["test_complete"])
    test_complete_csv = test_complete_dir / "test_data.csv"
    if test_complete_csv.exists():
        test_complete = integrate_test_set(
            test_complete_csv, test_complete_dir, manifest, name="complete"
        )
        test_complete.to_csv(output_dir / "test_complete_matched.csv", index=False)

    test_sample_dir = _resolve(paths["test_sample"])
    test_sample_csv = test_sample_dir / "test_data.csv"
    if test_sample_csv.exists():
        test_sample = integrate_test_set(test_sample_csv, test_sample_dir, manifest, name="sample")
        test_sample.to_csv(output_dir / "test_sample_matched.csv", index=False)

    # Print EDA
    print_eda_stats(manifest, splits)

    elapsed = time.time() - start
    logger.info("dataset_preparation_complete", elapsed_seconds=f"{elapsed:.1f}")


if __name__ == "__main__":
    main()
