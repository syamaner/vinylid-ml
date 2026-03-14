"""EXIF metadata extraction and privacy scrubbing utilities.

Gallery images have metadata embedded in EXIF tags:
- Tag 33432 (Copyright): JSON with ReleaseId, Artist, Album
- Tag 315 (Artist): "Artist - Album" string
- Tag 42112 (XPComment): MusicBrainz API URL
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from PIL import Image as _PILImage

from vinylid_ml import VinylIDError

# Gallery contains legitimate large album art scans (up to ~10K px).
# Raise Pillow's decompression bomb threshold so getexif() doesn't choke.
_PILImage.MAX_IMAGE_PIXELS = 200_000_000

# Some PNGs embed large text chunks (e.g. XML metadata). Raise the limit.
from PIL import PngImagePlugin as _PngPlugin  # noqa: E402

_PngPlugin.MAX_TEXT_CHUNK = 10 * 1024 * 1024  # 10 MB

if TYPE_CHECKING:
    from PIL.Image import Image

__all__ = [
    "AlbumMetadata",
    "ExifExtractionError",
    "extract_metadata",
    "load_image_stripped",
]

logger = structlog.get_logger()

# EXIF tag IDs used in our gallery images
_TAG_COPYRIGHT = 33432
_TAG_ARTIST = 315
_TAG_XP_COMMENT = 42112


class ExifExtractionError(VinylIDError):
    """Raised when EXIF metadata cannot be extracted from an image."""


@dataclass(frozen=True)
class AlbumMetadata:
    """Metadata extracted from a gallery image's EXIF tags.

    Attributes:
        release_id: MusicBrainz release UUID.
        artist: Artist name from the Copyright JSON tag.
        album: Album title from the Copyright JSON tag.
        artist_tag: Combined "Artist - Album" string from tag 315.
        musicbrainz_url: MusicBrainz API URL from tag 42112, if present.
        image_path: Absolute path to the source image.
        width: Image width in pixels.
        height: Image height in pixels.
        image_format: Image format (JPEG, PNG, etc.).
    """

    release_id: str
    artist: str
    album: str
    artist_tag: str
    musicbrainz_url: str
    image_path: Path
    width: int
    height: int
    image_format: str


def extract_metadata(image_path: Path) -> AlbumMetadata:
    """Extract album metadata from a gallery image's EXIF tags.

    Args:
        image_path: Path to a gallery image file.

    Returns:
        AlbumMetadata with all fields populated from EXIF.

    Raises:
        ExifExtractionError: If the Copyright JSON tag is missing or malformed.
        FileNotFoundError: If the image file does not exist.
    """
    from PIL import Image

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path)
    width, height = img.size
    image_format = img.format or "UNKNOWN"

    exif = img.getexif()
    if not exif:
        raise ExifExtractionError(f"No EXIF data found in {image_path}")

    # Extract Copyright JSON tag (primary metadata source)
    copyright_raw = exif.get(_TAG_COPYRIGHT)
    if copyright_raw is None:
        raise ExifExtractionError(
            f"Missing Copyright tag ({_TAG_COPYRIGHT}) in {image_path}"
        )

    try:
        copyright_json = json.loads(copyright_raw)
    except (json.JSONDecodeError, TypeError) as e:
        raise ExifExtractionError(
            f"Malformed Copyright JSON in {image_path}: {e}"
        ) from e

    release_id = copyright_json.get("ReleaseId", "")
    artist = copyright_json.get("Artist", "")
    album = copyright_json.get("Album", "")

    if not release_id:
        raise ExifExtractionError(
            f"Empty ReleaseId in Copyright JSON for {image_path}"
        )

    # Extract supplementary tags
    artist_tag = exif.get(_TAG_ARTIST, "")
    musicbrainz_url = exif.get(_TAG_XP_COMMENT, "")

    # Close to free file handle
    img.close()

    return AlbumMetadata(
        release_id=release_id,
        artist=artist,
        album=album,
        artist_tag=artist_tag,
        musicbrainz_url=musicbrainz_url,
        image_path=image_path.resolve(),
        width=width,
        height=height,
        image_format=image_format,
    )


def load_image_stripped(image_path: Path) -> Image:
    """Load an image with all EXIF/metadata stripped in-memory.

    Returns a clean PIL Image with pixel data only — no EXIF, XMP, or IPTC.
    The original file on disk is never modified.

    Args:
        image_path: Path to the image file.

    Returns:
        A new PIL Image containing only pixel data (no metadata).

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    from PIL import Image

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path)
    # Force load pixel data before stripping
    img.load()

    # Create a clean copy with no metadata by re-encoding through a buffer
    buf = io.BytesIO()
    # Convert to RGB if necessary (some PNGs are RGBA/P mode)
    clean = img.convert("RGB") if img.mode not in ("RGB", "L") else img.copy()
    clean.save(buf, format="PNG")
    buf.seek(0)
    stripped = Image.open(buf)
    stripped.load()

    img.close()
    return stripped
