"""Tests for vinylid_ml.exif."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from vinylid_ml.exif import (
    ExifExtractionError,
    extract_metadata,
    load_image_stripped,
)


class TestExtractMetadata:
    """Tests for extract_metadata()."""

    def test_extracts_all_fields(self, tmp_image: Path) -> None:
        """Happy path: all EXIF tags present and well-formed."""
        meta = extract_metadata(tmp_image)

        assert meta.release_id == "abc12345-def6-7890-abcd-ef1234567890"
        assert meta.artist == "Test Artist"
        assert meta.album == "Test Album"
        assert meta.artist_tag == "Test Artist - Test Album"
        assert "musicbrainz" in meta.musicbrainz_url
        assert meta.image_path == tmp_image.resolve()
        assert meta.width == 100
        assert meta.height == 100
        assert meta.image_format == "JPEG"

    def test_frozen_dataclass(self, tmp_image: Path) -> None:
        """AlbumMetadata is immutable."""
        meta = extract_metadata(tmp_image)
        with pytest.raises(AttributeError):
            meta.artist = "Modified"  # type: ignore[misc]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Image not found"):
            extract_metadata(tmp_path / "nonexistent.jpg")

    def test_no_exif_raises(self, tmp_image_no_exif: Path) -> None:
        """Image with no EXIF data raises ExifExtractionError."""
        with pytest.raises(ExifExtractionError, match="No EXIF data"):
            extract_metadata(tmp_image_no_exif)

    def test_malformed_copyright_json_raises(self, tmp_path: Path) -> None:
        """Non-JSON Copyright tag raises ExifExtractionError."""
        img = Image.new("RGB", (50, 50))
        exif = img.getexif()
        exif[33432] = "not-valid-json"
        path = tmp_path / "bad_json.jpg"
        img.save(path, exif=exif.tobytes())
        img.close()

        with pytest.raises(ExifExtractionError, match="Malformed Copyright JSON"):
            extract_metadata(path)

    def test_missing_copyright_tag_raises(self, tmp_path: Path) -> None:
        """Image with EXIF but no Copyright tag raises ExifExtractionError."""
        img = Image.new("RGB", (50, 50))
        exif = img.getexif()
        exif[315] = "Some Artist - Some Album"  # Artist tag only
        path = tmp_path / "no_copyright.jpg"
        img.save(path, exif=exif.tobytes())
        img.close()

        with pytest.raises(ExifExtractionError, match="Missing Copyright tag"):
            extract_metadata(path)

    def test_empty_release_id_raises(self, tmp_path: Path) -> None:
        """Copyright JSON with empty ReleaseId raises ExifExtractionError."""
        img = Image.new("RGB", (50, 50))
        exif = img.getexif()
        exif[33432] = json.dumps({"ReleaseId": "", "Artist": "A", "Album": "B"})
        path = tmp_path / "empty_release.jpg"
        img.save(path, exif=exif.tobytes())
        img.close()

        with pytest.raises(ExifExtractionError, match="Empty ReleaseId"):
            extract_metadata(path)

    def test_missing_optional_tags(self, tmp_path: Path) -> None:
        """Missing Artist (315) and XPComment (42112) don't raise — they default to empty."""
        img = Image.new("RGB", (80, 60))
        exif = img.getexif()
        exif[33432] = json.dumps(
            {
                "ReleaseId": "uuid-123",
                "Artist": "Foo",
                "Album": "Bar",
            }
        )
        # No tag 315 or 42112
        path = tmp_path / "minimal.jpg"
        img.save(path, exif=exif.tobytes())
        img.close()

        meta = extract_metadata(path)
        assert meta.release_id == "uuid-123"
        assert meta.artist_tag == ""
        assert meta.musicbrainz_url == ""
        assert meta.width == 80
        assert meta.height == 60


class TestExtractMetadataRealImages:
    """Integration tests using real album cover images from fixtures."""

    def test_all_fixtures_have_valid_metadata(self, real_gallery_images: list[Path]) -> None:
        """Every fixture image should produce valid AlbumMetadata."""
        for img_path in real_gallery_images:
            meta = extract_metadata(img_path)
            assert meta.release_id, f"Empty release_id for {img_path.name}"
            assert meta.artist, f"Empty artist for {img_path.name}"
            assert meta.album, f"Empty album for {img_path.name}"
            assert meta.width > 0
            assert meta.height > 0

    def test_jpeg_and_png_formats(self, real_gallery_images: list[Path]) -> None:
        """Fixtures should contain both JPEG and PNG images."""
        formats = {extract_metadata(p).image_format for p in real_gallery_images}
        assert "JPEG" in formats
        assert "PNG" in formats

    def test_resolution_range(self, real_gallery_images: list[Path]) -> None:
        """Fixtures should span a range of resolutions."""
        sizes = [
            max(extract_metadata(p).width, extract_metadata(p).height) for p in real_gallery_images
        ]
        assert min(sizes) < 400, "Expected at least one small image (<400px)"
        assert max(sizes) >= 2000, "Expected at least one large image (>=2000px)"


class TestLoadImageStripped:
    """Tests for load_image_stripped()."""

    def test_returns_image_with_no_exif(self, tmp_image: Path) -> None:
        """Stripped image should have zero EXIF tags."""
        stripped = load_image_stripped(tmp_image)

        assert stripped.getexif() is not None
        # Pillow getexif() returns an IFD dict — empty means no tags
        assert len(stripped.getexif()) == 0

    def test_preserves_pixel_data(self, tmp_image: Path) -> None:
        """Pixel dimensions should match the original."""
        original = Image.open(tmp_image)
        stripped = load_image_stripped(tmp_image)

        assert stripped.size == original.size
        original.close()

    def test_original_untouched(self, tmp_image: Path) -> None:
        """Original file on disk should still have EXIF after stripping."""
        load_image_stripped(tmp_image)

        # Re-read original
        check = Image.open(tmp_image)
        exif = check.getexif()
        assert 33432 in exif  # Copyright tag still present
        check.close()

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Image not found"):
            load_image_stripped(tmp_path / "ghost.jpg")

    def test_png_rgba_converted_to_rgb(self, tmp_path: Path) -> None:
        """RGBA PNGs should be converted to RGB during strip."""
        img = Image.new("RGBA", (32, 32), color=(255, 0, 0, 128))
        path = tmp_path / "rgba.png"
        img.save(path)
        img.close()

        stripped = load_image_stripped(path)
        assert stripped.mode == "RGB"
