from __future__ import annotations

import cv2
import numpy as np

from heritage_restore.config import ProcessingConfig
from heritage_restore.pipeline import RestorationPipeline


def damaged_sample() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean = np.full((180, 240, 3), (185, 202, 220), dtype=np.uint8)
    cv2.rectangle(clean, (40, 40), (200, 145), (110, 130, 150), 3)
    cv2.putText(clean, "ARCHIVE", (58, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 90, 105), 2)
    damaged = clean.copy()
    mask = np.zeros(clean.shape[:2], dtype=np.uint8)
    crack = np.array([[25, 20], [70, 62], [103, 75], [145, 125], [210, 155]], np.int32)
    cv2.polylines(damaged, [crack], False, (20, 20, 20), 4)
    cv2.polylines(mask, [crack], False, 255, 8)
    cv2.circle(damaged, (185, 55), 20, (45, 70, 125), -1)
    cv2.circle(mask, (185, 55), 23, 255, -1)
    return damaged, clean, mask


def test_integrated_pipeline_outputs_and_metrics() -> None:
    damaged, clean, mask = damaged_sample()
    result = RestorationPipeline().process(damaged, clean, mask)
    assert result.restored.shape == damaged.shape
    assert result.damage_mask.shape == damaged.shape[:2]
    assert result.reference_difference_mask.shape == damaged.shape[:2]
    assert np.count_nonzero(result.reference_difference_mask) > 0
    assert result.overlay.shape == damaged.shape
    assert {"grayscale", "lab", "hsv", "binary", "illumination_corrected"}.issubset(result.color_spaces)
    assert {"gaussian", "median"} == set(result.filter_comparison)
    required_metrics = {
        "damage_area_percent",
        "detected_region_count",
        "total_region_perimeter",
        "mean_region_orientation_degrees",
        "contrast_improvement",
        "restoration_quality_score",
        "decision",
        "selected_filter",
        "analysis_mode",
        "segmentation_iou",
        "segmentation_precision",
        "segmentation_recall",
        "segmentation_f1",
        "psnr_before_db",
        "psnr_after_db",
        "ssim_before",
        "ssim_after",
        "processing_time_ms",
    }
    assert required_metrics.issubset(result.measurements)
    assert 0 <= result.measurements["segmentation_iou"] <= 1
    assert 0 <= result.measurements["restoration_quality_score"] <= 100
    assert result.measurements["processing_time_ms"] > 0


def test_roi_and_rotation_are_applied() -> None:
    damaged, _, _ = damaged_sample()
    config = ProcessingConfig(roi_x=0.1, roi_y=0.1, roi_width=0.6, roi_height=0.7, rotation_degrees=2.0)
    result = RestorationPipeline(config).process(damaged)
    assert result.roi_image.shape[:2] == (126, 144)
    assert result.measurements["roi_x"] == 24
    assert result.measurements["roi_y"] == 18


def test_filter_override_is_respected() -> None:
    damaged, _, _ = damaged_sample()
    result = RestorationPipeline(ProcessingConfig(denoise_method="gaussian")).process(damaged)
    assert result.measurements["selected_filter"] == "gaussian"
