from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from heritage_restore.io import ImageValidationError, load_image_bytes


def png_bytes(width: int = 96, height: int = 64) -> bytes:
    image = Image.fromarray(np.full((height, width, 3), (120, 150, 175), dtype=np.uint8), "RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_load_valid_image_and_metadata() -> None:
    image, metadata = load_image_bytes(png_bytes(), "archive.png")
    assert image.shape == (64, 96, 3)
    assert metadata.width == 96
    assert metadata.height == 64
    assert metadata.channels == 3
    assert metadata.bit_depth == 8
    assert metadata.format == "PNG"
    assert metadata.file_size_bytes > 0


def test_rejects_unsupported_extension() -> None:
    with pytest.raises(ImageValidationError, match="Unsupported extension"):
        load_image_bytes(png_bytes(), "archive.gif")


def test_rejects_corrupted_file() -> None:
    with pytest.raises(ImageValidationError):
        load_image_bytes(b"not an image", "broken.png")


def test_rejects_tiny_image() -> None:
    with pytest.raises(ImageValidationError, match="at least 32"):
        load_image_bytes(png_bytes(16, 16), "tiny.png")
