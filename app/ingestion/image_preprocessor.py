"""Image preprocessing for OCR.

The preprocessor improves OCR consistency for scanned pages and photographs.

Pseudo-code:

    open image
    correct EXIF orientation
    convert to grayscale
    optionally upscale small images
    optionally apply autocontrast
    optionally sharpen
    optionally binarize using threshold
    return processed image and diagnostics
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass(frozen=True)
class ImagePreprocessingConfig:
    """Configuration for OCR-oriented image cleanup."""

    grayscale: bool = True
    autocontrast: bool = True
    sharpen: bool = True
    threshold: int | None = None
    minimum_width: int = 1600
    upscale_small_images: bool = True

    def __post_init__(self) -> None:
        if self.threshold is not None and not 0 <= self.threshold <= 255:
            raise ValueError("threshold must be between 0 and 255")
        if self.minimum_width <= 0:
            raise ValueError("minimum_width must be greater than zero")


@dataclass(frozen=True)
class ImagePreprocessingResult:
    image: Image.Image
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    operations: tuple[str, ...]


def load_image(path: str | Path) -> Image.Image:
    """Load a local JPEG/JPG/PNG file.

    Pseudo-code:
    1. Resolve path.
    2. Reject missing files.
    3. Open with Pillow.
    4. Load pixels before closing the file handle.
    5. Return an independent image object.
    """
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"image does not exist: {source}")
    if source.suffix.lower() not in {".jpeg", ".jpg", ".png"}:
        raise ValueError("supported image formats are .jpeg, .jpg, and .png")

    with Image.open(source) as image:
        image.load()
        return image.copy()


def preprocess_image(
    image: Image.Image,
    *,
    config: ImagePreprocessingConfig | None = None,
) -> ImagePreprocessingResult:
    """Prepare an image for OCR without modifying the caller's object.

    Pseudo-code:
    1. Copy image.
    2. Correct orientation using EXIF metadata.
    3. Convert to grayscale.
    4. Upscale when width is below the configured minimum.
    5. Increase contrast.
    6. Sharpen character edges.
    7. Apply binary threshold when configured.
    8. Return image plus the operations performed.
    """
    cfg = config or ImagePreprocessingConfig()
    original_size = image.size
    processed = ImageOps.exif_transpose(image.copy())
    operations: list[str] = ["exif_transpose"]

    if cfg.grayscale and processed.mode != "L":
        processed = processed.convert("L")
        operations.append("grayscale")

    if cfg.upscale_small_images and processed.width < cfg.minimum_width:
        scale = cfg.minimum_width / processed.width
        new_size = (
            int(round(processed.width * scale)),
            int(round(processed.height * scale)),
        )
        processed = processed.resize(new_size, Image.Resampling.LANCZOS)
        operations.append(f"upscale:{new_size[0]}x{new_size[1]}")

    if cfg.autocontrast:
        processed = ImageOps.autocontrast(processed)
        operations.append("autocontrast")

    if cfg.sharpen:
        processed = processed.filter(ImageFilter.SHARPEN)
        operations.append("sharpen")

    if cfg.threshold is not None:
        if processed.mode != "L":
            processed = processed.convert("L")
        threshold = cfg.threshold
        processed = processed.point(
            lambda pixel: 255 if pixel >= threshold else 0,
            mode="1",
        ).convert("L")
        operations.append(f"threshold:{threshold}")

    return ImagePreprocessingResult(
        image=processed,
        original_size=original_size,
        processed_size=processed.size,
        operations=tuple(operations),
    )


def preprocess_image_file(
    path: str | Path,
    *,
    config: ImagePreprocessingConfig | None = None,
) -> ImagePreprocessingResult:
    """Load and preprocess one local image file."""
    return preprocess_image(load_image(path), config=config)