from __future__ import annotations

import io
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_FILE_SIZE_BYTES = 40 * 1024 * 1024


class ImageValidationError(ValueError):
    """Raised when an input is unsupported, corrupted, or unsafe to process."""


@dataclass(frozen=True)
class ImageMetadata:
    filename: str
    format: str
    width: int
    height: int
    channels: int
    bit_depth: int
    file_size_bytes: int
    selected_region: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _decode_image(data: bytes) -> np.ndarray:
    if not data:
        raise ImageValidationError("The selected file is empty.")
    if len(data) > MAX_FILE_SIZE_BYTES:
        raise ImageValidationError("The file exceeds the 40 MB safety limit.")
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise ImageValidationError("OpenCV could not decode the image; it may be corrupted.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ImageValidationError("Only grayscale, RGB, and RGBA images are supported.")
    if image.shape[0] < 32 or image.shape[1] < 32:
        raise ImageValidationError("Images must be at least 32 x 32 pixels.")
    if image.dtype not in (np.uint8, np.uint16):
        raise ImageValidationError("Only 8-bit and 16-bit integer images are supported.")
    if image.dtype == np.uint16:
        image = cv2.convertScaleAbs(image, alpha=255.0 / 65535.0)
    return image


def load_image_bytes(data: bytes, filename: str = "uploaded_image.png") -> tuple[np.ndarray, ImageMetadata]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported extension '{suffix or '[none]'}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )
    try:
        with Image.open(io.BytesIO(data)) as pil_image:
            detected_format = pil_image.format or suffix.lstrip(".").upper()
            original_bit_depth = 16 if "16" in pil_image.mode else 8
    except Exception as exc:
        raise ImageValidationError(f"The image header is invalid: {exc}") from exc
    image = _decode_image(data)
    height, width = image.shape[:2]
    metadata = ImageMetadata(
        filename=Path(filename).name,
        format=detected_format,
        width=width,
        height=height,
        channels=3,
        bit_depth=original_bit_depth,
        file_size_bytes=len(data),
        selected_region=f"full image: x=0, y=0, width={width}, height={height}",
    )
    return image, metadata


def load_image(path: str | os.PathLike[str]) -> tuple[np.ndarray, ImageMetadata]:
    image_path = Path(path)
    if not image_path.is_file():
        raise ImageValidationError(f"Image file does not exist: {image_path}")
    try:
        data = image_path.read_bytes()
    except OSError as exc:
        raise ImageValidationError(f"Could not read image: {exc}") from exc
    return load_image_bytes(data, image_path.name)


def image_to_png_bytes(image_bgr: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image_bgr)
    if not success:
        raise RuntimeError("OpenCV could not encode the output image.")
    return encoded.tobytes()


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", mask)
    if not success:
        raise RuntimeError("OpenCV could not encode the output mask.")
    return encoded.tobytes()


def update_region_metadata(metadata: ImageMetadata, region: tuple[int, int, int, int]) -> ImageMetadata:
    x, y, width, height = region
    values = metadata.as_dict()
    values["selected_region"] = f"ROI: x={x}, y={y}, width={width}, height={height}"
    return ImageMetadata(**values)
