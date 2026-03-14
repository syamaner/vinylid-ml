"""Shared test fixtures for VinylID ML tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def tmp_image(tmp_path: Path) -> Path:
    """Create a minimal test JPEG image with EXIF metadata."""

    img = Image.new("RGB", (100, 100), color=(255, 0, 0))

    # Build EXIF data matching our gallery format
    exif = img.getexif()
    copyright_json = json.dumps({
        "ReleaseId": "abc12345-def6-7890-abcd-ef1234567890",
        "Artist": "Test Artist",
        "Album": "Test Album",
    })
    exif[33432] = copyright_json  # Copyright tag
    exif[315] = "Test Artist - Test Album"  # Artist tag
    exif[42112] = "http://musicbrainz.org/ws/2/release/abc12345"  # XPComment

    image_path = tmp_path / "test_image.jpg"
    img.save(image_path, exif=exif.tobytes())
    img.close()
    return image_path


@pytest.fixture
def tmp_image_no_exif(tmp_path: Path) -> Path:
    """Create a test image with no EXIF data."""
    img = Image.new("RGB", (50, 50), color=(0, 255, 0))
    image_path = tmp_path / "no_exif.png"
    img.save(image_path)
    img.close()
    return image_path


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "gallery"


@pytest.fixture
def real_gallery_images() -> list[Path]:
    """Return paths to real album cover images from the fixtures directory.

    These are scraped MusicBrainz covers with safe EXIF metadata (no GPS/faces).
    Available images span different formats (JPEG, PNG) and sizes (300px-4000px).
    """
    assert FIXTURES_DIR.exists(), f"Fixtures dir missing: {FIXTURES_DIR}"
    images = sorted(FIXTURES_DIR.glob("*"))
    assert len(images) >= 5, f"Expected ≥5 fixture images, found {len(images)}"
    return images


@pytest.fixture
def tmp_gallery(tmp_path: Path) -> Path:
    """Create a minimal gallery directory structure with EXIF-tagged images."""
    gallery_root = tmp_path / "album-export"

    albums = [
        ("ArtistA", "Album1", [
            ("release1-img1.jpg", "release-uuid-1", "ArtistA", "Album1"),
            ("release1-img2.jpg", "release-uuid-1", "ArtistA", "Album1"),
        ]),
        ("ArtistA", "Album2", [
            ("release2-img1.jpg", "release-uuid-2", "ArtistA", "Album2"),
        ]),
        ("ArtistB", "Album3", [
            ("release3-img1.jpg", "release-uuid-3", "ArtistB", "Album3"),
            ("release3-img2.jpg", "release-uuid-3", "ArtistB", "Album3"),
            ("release3-img3.jpg", "release-uuid-3", "ArtistB", "Album3"),
        ]),
    ]

    for artist, album, images in albums:
        album_dir = gallery_root / artist / album
        album_dir.mkdir(parents=True)

        for filename, release_id, artist_name, album_name in images:
            img = Image.new("RGB", (200, 200), color=(128, 128, 128))
            exif = img.getexif()
            exif[33432] = json.dumps({
                "ReleaseId": release_id,
                "Artist": artist_name,
                "Album": album_name,
            })
            exif[315] = f"{artist_name} - {album_name}"

            img.save(album_dir / filename, exif=exif.tobytes())
            img.close()

    return gallery_root
