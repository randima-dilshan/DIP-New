from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ProcessingConfig:
    """Validated controls exposed by the CLI and Streamlit interface."""

    clahe_clip_limit: float = 1.5
    clahe_tile_size: int = 8
    illumination_kernel: int = 151
    denoise_method: str = "auto"
    gaussian_kernel: int = 3
    median_kernel: int = 3
    crack_kernel: int = 17
    adaptive_block_size: int = 31
    adaptive_c: int = 7
    morphology_kernel: int = 3
    minimum_region_area: int = 18
    stain_sensitivity: float = 2.7
    inpaint_radius: int = 3
    rotation_degrees: float = 0.0
    roi_x: float = 0.0
    roi_y: float = 0.0
    roi_width: float = 1.0
    roi_height: float = 1.0

    def validate(self) -> "ProcessingConfig":
        if not 0.5 <= self.clahe_clip_limit <= 8.0:
            raise ValueError("CLAHE clip limit must be between 0.5 and 8.0.")
        if not 2 <= self.clahe_tile_size <= 32:
            raise ValueError("CLAHE tile size must be between 2 and 32.")
        for name in (
            "gaussian_kernel",
            "median_kernel",
            "crack_kernel",
            "adaptive_block_size",
            "morphology_kernel",
            "illumination_kernel",
        ):
            value = int(getattr(self, name))
            if value < 3 or value % 2 == 0:
                raise ValueError(f"{name} must be an odd integer of at least 3.")
        if self.denoise_method not in {"auto", "median", "gaussian"}:
            raise ValueError("Denoise method must be auto, median, or gaussian.")
        if not 0 <= self.minimum_region_area <= 100_000:
            raise ValueError("Minimum region area is outside the supported range.")
        if not 0.5 <= self.stain_sensitivity <= 6.0:
            raise ValueError("Stain sensitivity must be between 0.5 and 6.0.")
        if not 1 <= self.inpaint_radius <= 15:
            raise ValueError("Inpaint radius must be between 1 and 15.")
        if not -180 <= self.rotation_degrees <= 180:
            raise ValueError("Rotation must be between -180 and 180 degrees.")
        if not all(0.0 <= v <= 1.0 for v in (self.roi_x, self.roi_y, self.roi_width, self.roi_height)):
            raise ValueError("ROI values must be normalized to the range 0..1.")
        if self.roi_width <= 0 or self.roi_height <= 0:
            raise ValueError("ROI width and height must be greater than zero.")
        if self.roi_x + self.roi_width > 1.000001 or self.roi_y + self.roi_height > 1.000001:
            raise ValueError("ROI extends beyond the image boundary.")
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
