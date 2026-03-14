"""Tests for scripts/strip_metadata.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from scripts.strip_metadata import strip_metadata


class TestStripMetadata:
    """Tests for strip_metadata()."""

    def test_output_has_no_exif(self, tmp_image: Path, tmp_path: Path) -> None:
        """Stripped copy should contain zero EXIF tags."""
        output = tmp_path / "clean" / "output.jpg"
        strip_metadata(tmp_image, output)

        check = Image.open(output)
        assert len(check.getexif()) == 0
        check.close()

    def test_output_is_jpeg(self, tmp_image: Path, tmp_path: Path) -> None:
        """Output should always be a JPEG file."""
        output = tmp_path / "out.jpg"
        strip_metadata(tmp_image, output)

        check = Image.open(output)
        assert check.format == "JPEG"
        check.close()

    def test_creates_parent_directories(self, tmp_image: Path, tmp_path: Path) -> None:
        """Output directories are created automatically."""
        output = tmp_path / "a" / "b" / "c" / "out.jpg"
        strip_metadata(tmp_image, output)
        assert output.exists()

    def test_preserves_pixel_dimensions(self, tmp_image: Path, tmp_path: Path) -> None:
        """Output dimensions should match input."""
        output = tmp_path / "out.jpg"
        strip_metadata(tmp_image, output)

        original = Image.open(tmp_image)
        cleaned = Image.open(output)
        assert cleaned.size == original.size
        original.close()
        cleaned.close()

    def test_original_untouched(self, tmp_image: Path, tmp_path: Path) -> None:
        """Source file should still have its EXIF after stripping."""
        output = tmp_path / "out.jpg"
        strip_metadata(tmp_image, output)

        check = Image.open(tmp_image)
        exif = check.getexif()
        assert 33432 in exif  # Copyright tag still in original
        check.close()

    def test_missing_source_raises(self, tmp_path: Path) -> None:
        """Non-existent source raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Image not found"):
            strip_metadata(tmp_path / "nope.jpg", tmp_path / "out.jpg")


class TestStripMetadataRealImages:
    """Integration tests using real album cover images from fixtures."""

    def test_all_fixtures_strip_cleanly(self, real_gallery_images: list[Path], tmp_path: Path) -> None:
        """Every fixture image should strip to a clean JPEG with no EXIF."""
        for img_path in real_gallery_images:
            output = tmp_path / f"stripped_{img_path.stem}.jpg"
            strip_metadata(img_path, output)

            check = Image.open(output)
            assert len(check.getexif()) == 0, f"EXIF leaked in {img_path.name}"
            assert check.format == "JPEG"
            check.close()

    def test_pixel_dimensions_preserved(self, real_gallery_images: list[Path], tmp_path: Path) -> None:
        """Stripped output should match original dimensions."""
        for img_path in real_gallery_images:
            original = Image.open(img_path)
            orig_size = original.size
            original.close()

            output = tmp_path / f"dim_{img_path.stem}.jpg"
            strip_metadata(img_path, output)

            cleaned = Image.open(output)
            assert cleaned.size == orig_size, f"Size mismatch for {img_path.name}"
            cleaned.close()


class TestStripMetadataSyntheticEdgeCases:
    """Edge case tests using synthetic images."""

    def test_rgba_input_converted(self, tmp_path: Path) -> None:
        """RGBA images should be converted to RGB in JPEG output."""
        img = Image.new("RGBA", (40, 40), color=(0, 255, 0, 128))
        src = tmp_path / "rgba_src.png"
        img.save(src)
        img.close()

        output = tmp_path / "rgba_out.jpg"
        strip_metadata(src, output)

        check = Image.open(output)
        assert check.mode == "RGB"
        check.close()
