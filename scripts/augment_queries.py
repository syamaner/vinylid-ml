"""Generate augmented "phone capture" query variants from val/test gallery images.

Usage:
    python scripts/augment_queries.py [--config configs/dataset.yaml] [--split val]

Produces N augmented variants per image simulating real-world capture conditions:
perspective warp, brightness/contrast, blur, crop, noise, glare, downscale.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import yaml
from PIL import Image, ImageDraw, ImageFilter

logger = structlog.get_logger()

# Augmentation type labels for stratified evaluation
AUG_TYPES = [
    "perspective",
    "brightness_contrast",
    "blur",
    "crop",
    "noise",
    "glare",
    "downscale",
]


def augment_perspective(img: Image.Image, max_angle: float = 30.0) -> Image.Image:
    """Apply random perspective warp simulating viewing angles.

    Args:
        img: Input PIL image.
        max_angle: Maximum viewing angle in degrees.

    Returns:
        Perspective-warped image.
    """
    w, h = img.size
    # Random corner offsets proportional to angle
    strength = max_angle / 90.0
    max_offset = int(min(w, h) * 0.15 * strength)

    coeffs = _find_perspective_coeffs(
        w,
        h,
        [random.randint(-max_offset, max_offset) for _ in range(8)],
    )
    return img.transform(  # type: ignore[reportUnknownMemberType]
        (w, h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC
    )


def augment_brightness_contrast(
    img: Image.Image,
    brightness_range: float = 0.30,
    contrast_range: float = 0.20,
) -> Image.Image:
    """Apply random brightness and contrast jitter.

    Args:
        img: Input PIL image.
        brightness_range: Maximum brightness shift (±).
        contrast_range: Maximum contrast shift (±).

    Returns:
        Adjusted image.
    """
    from PIL import ImageEnhance

    b_factor = 1.0 + random.uniform(-brightness_range, brightness_range)
    c_factor = 1.0 + random.uniform(-contrast_range, contrast_range)

    img = ImageEnhance.Brightness(img).enhance(b_factor)
    img = ImageEnhance.Contrast(img).enhance(c_factor)
    return img


def augment_blur(img: Image.Image, sigma_range: tuple[float, float] = (0.5, 2.0)) -> Image.Image:
    """Apply Gaussian blur simulating camera defocus.

    Args:
        img: Input PIL image.
        sigma_range: Range for blur sigma.

    Returns:
        Blurred image.
    """
    sigma = random.uniform(*sigma_range)
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def augment_crop(img: Image.Image, scale_range: tuple[float, float] = (0.70, 0.90)) -> Image.Image:
    """Apply random crop simulating partial cover visibility.

    Args:
        img: Input PIL image.
        scale_range: Fraction of image to keep.

    Returns:
        Cropped and resized image (back to original size).
    """
    w, h = img.size
    scale = random.uniform(*scale_range)
    crop_w, crop_h = int(w * scale), int(h * scale)

    left = random.randint(0, w - crop_w)
    top = random.randint(0, h - crop_h)

    cropped = img.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((w, h), Image.Resampling.BICUBIC)


def augment_noise(img: Image.Image, sigma_range: tuple[int, int] = (5, 25)) -> Image.Image:
    """Add Gaussian noise simulating phone camera sensor noise.

    Args:
        img: Input PIL image.
        sigma_range: Noise standard deviation range (8-bit scale).

    Returns:
        Noisy image.
    """
    arr = np.array(img, dtype=np.float32)
    sigma = random.uniform(*[float(s) for s in sigma_range])
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def augment_glare(img: Image.Image) -> Image.Image:
    """Add specular highlight simulating shrinkwrap glare.

    Draws a random bright elliptical spot overlay.

    Args:
        img: Input PIL image.

    Returns:
        Image with glare spot.
    """
    result = img.copy()
    w, h = result.size

    # Random ellipse position and size
    cx = random.randint(w // 4, 3 * w // 4)
    cy = random.randint(h // 4, 3 * h // 4)
    rx = random.randint(w // 10, w // 4)
    ry = random.randint(h // 10, h // 4)

    # Create semi-transparent white ellipse
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    alpha = random.randint(80, 160)
    draw.ellipse(
        [cx - rx, cy - ry, cx + rx, cy + ry],
        fill=(255, 255, 255, alpha),
    )
    # Blur the ellipse for a soft glow
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max(rx, ry) // 3))

    # Composite
    result = result.convert("RGBA")
    result = Image.alpha_composite(result, overlay)
    return result.convert("RGB")


def augment_downscale(img: Image.Image, target_range: tuple[int, int] = (400, 600)) -> Image.Image:
    """Downscale then upscale simulating lower-quality captures.

    Args:
        img: Input PIL image.
        target_range: Range of intermediate downscale sizes (px).

    Returns:
        Downscaled-then-upscaled image at original size.
    """
    w, h = img.size
    target = random.randint(*target_range)

    # Downscale
    scale = target / max(w, h)
    small_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    small = img.resize(small_size, Image.Resampling.BILINEAR)

    # Upscale back
    return small.resize((w, h), Image.Resampling.BICUBIC)


def generate_augmented_queries(
    image_path: Path,
    output_dir: Path,
    album_id: str,
    config: dict[str, Any],
    seed: int = 42,
) -> list[dict[str, str | int]]:
    """Generate augmented variants for a single image.

    Args:
        image_path: Source gallery image.
        output_dir: Directory to save augmented images.
        album_id: Album ID for the source image.
        config: Augmentation config from dataset.yaml.
        seed: Random seed for this image.

    Returns:
        List of dicts with augmented image metadata.
    """
    random.seed(seed)
    np.random.seed(seed)

    img = Image.open(image_path).convert("RGB")
    stem = image_path.stem
    records: list[dict[str, str | int]] = []

    augmenters: dict[str, Callable[[Image.Image], Image.Image]] = {
        "perspective": lambda im: augment_perspective(
            im, float(config.get("perspective_max_angle", 30))
        ),
        "brightness_contrast": lambda im: augment_brightness_contrast(
            im, float(config.get("brightness_range", 0.30)),
            float(config.get("contrast_range", 0.20)),
        ),
        "blur": lambda im: augment_blur(
            im, tuple(config.get("blur_sigma_range", [0.5, 2.0]))  # type: ignore[arg-type]
        ),
        "crop": lambda im: augment_crop(
            im, tuple(config.get("crop_scale_range", [0.70, 0.90]))  # type: ignore[arg-type]
        ),
        "noise": lambda im: augment_noise(
            im, tuple(config.get("noise_sigma_range", [5, 25]))  # type: ignore[arg-type]
        ),
        "downscale": lambda im: augment_downscale(
            im, tuple(config.get("downscale_range", [400, 600]))  # type: ignore[arg-type]
        ),
    }

    if config.get("glare_enabled", True):
        augmenters["glare"] = augment_glare

    n_variants: int = int(config.get("variants_per_image", 5))

    # Cycle through augmentation types
    aug_names = list(augmenters.keys())
    for i in range(n_variants):
        aug_name = aug_names[i % len(aug_names)]
        augmenter = augmenters[aug_name]

        augmented = augmenter(img)
        out_name = f"{stem}_aug{i}_{aug_name}.jpg"
        out_path = output_dir / out_name
        augmented.save(out_path, quality=90)

        records.append(
            {
                "image_path": str(out_path),
                "source_image": str(image_path),
                "album_id": album_id,
                "augmentation_type": aug_name,
                "variant_index": i,
            }
        )

    img.close()
    return records


# --- Private helpers ---


def _find_perspective_coeffs(w: int, h: int, offsets: list[int]) -> tuple[float, ...]:
    """Compute perspective transform coefficients from corner offsets."""
    # Source corners
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)

    # Destination corners with random offsets
    dst = np.array(
        [
            [offsets[0], offsets[1]],
            [w + offsets[2], offsets[3]],
            [w + offsets[4], h + offsets[5]],
            [offsets[6], h + offsets[7]],
        ],
        dtype=np.float64,
    )

    # Solve for perspective coefficients
    matrix: list[list[float]] = []
    for s, d in zip(src, dst, strict=True):
        matrix.append([d[0], d[1], 1, 0, 0, 0, -s[0] * d[0], -s[0] * d[1]])
        matrix.append([0, 0, 0, d[0], d[1], 1, -s[1] * d[0], -s[1] * d[1]])

    a = np.array(matrix, dtype=np.float64)
    b = np.array(src.flatten(), dtype=np.float64)

    try:
        res = np.linalg.lstsq(a, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

    return tuple(res.tolist())


def main() -> None:
    """Generate augmented queries for val and/or test splits."""
    parser = argparse.ArgumentParser(description="Generate augmented query images")
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "both"])
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.parquet"))
    parser.add_argument("--splits-file", type=Path, default=Path("data/splits.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/augmented"))
    args = parser.parse_args()

    import pandas as pd

    with args.config.open() as f:
        config = yaml.safe_load(f)

    aug_config = config["augmentation"]

    manifest = pd.read_parquet(args.manifest)
    with args.splits_file.open() as f:
        splits = json.load(f)

    splits_to_process = ["val", "test"] if args.split == "both" else [args.split]

    for split_name in splits_to_process:
        album_ids = {aid for aid, s in splits.items() if s == split_name}
        split_df = manifest[manifest["album_id"].isin(album_ids)]

        split_output = args.output_dir / split_name
        split_output.mkdir(parents=True, exist_ok=True)

        all_records: list[dict[str, str | int]] = []
        for idx, (_, row) in enumerate(split_df.iterrows()):
            image_path = Path(row["image_path"])
            per_image_seed = int(aug_config.get("seed", 42)) + idx

            records = generate_augmented_queries(
                image_path, split_output, str(row["album_id"]), aug_config,
                seed=per_image_seed,
            )
            all_records.extend(records)

            if (idx + 1) % 500 == 0:
                logger.info(
                    "augmentation_progress",
                    split=split_name,
                    processed=idx + 1,
                    total=len(split_df),
                )

        # Save manifest of augmented images
        aug_manifest = pd.DataFrame(all_records)
        aug_manifest.to_parquet(split_output / "augmented_manifest.parquet", index=False)

        logger.info(
            "augmentation_complete",
            split=split_name,
            source_images=len(split_df),
            augmented_images=len(all_records),
        )


if __name__ == "__main__":
    main()
