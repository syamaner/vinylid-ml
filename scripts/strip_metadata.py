"""Strip all metadata (EXIF, XMP, IPTC) from images for approved sharing.

Usage:
    python scripts/strip_metadata.py <image_path> [--output-dir test_photos_approved]
    python scripts/strip_metadata.py <image_path> <image_path2> ...

Creates clean copies with no metadata in the output directory.
Originals are never modified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


def strip_metadata(image_path: Path, output_path: Path) -> None:
    """Strip all metadata from an image and save a clean copy.

    Removes EXIF, XMP, IPTC, and any other metadata. The output
    contains only pixel data. Original file is never modified.

    Args:
        image_path: Source image to strip.
        output_path: Where to save the clean copy.

    Raises:
        FileNotFoundError: If source image doesn't exist.
        ValueError: If the image cannot be opened.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path)
    img.load()

    # Create a fresh image with no metadata by pasting pixels into a blank canvas
    clean = Image.new(img.mode, img.size)
    clean.paste(img)

    # Save as JPEG (no metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if clean.mode in ("RGBA", "P", "LA"):
        clean = clean.convert("RGB")
    clean.save(output_path, format="JPEG", quality=95)

    img.close()
    clean.close()

    # Verify no metadata in output
    check = Image.open(output_path)
    exif = check.getexif()
    if exif:
        print(f"WARNING: Output still has {len(exif)} EXIF tags — investigate!", file=sys.stderr)
    else:
        print(f"✓ Clean copy saved: {output_path} (no metadata)")
    check.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strip all metadata from images for approved sharing"
    )
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="Image file(s) to strip",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_photos_approved"),
        help="Output directory for clean copies",
    )
    args = parser.parse_args()

    for image_path in args.images:
        output_path = args.output_dir / image_path.name
        try:
            strip_metadata(image_path, output_path)
        except (FileNotFoundError, ValueError, OSError) as e:
            print(f"ERROR: {image_path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
