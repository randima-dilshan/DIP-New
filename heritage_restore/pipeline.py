from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .config import ProcessingConfig
from .metrics import contrast, entropy, psnr, segmentation_metrics, sharpness, ssim


@dataclass
class ProcessingResult:
    original: np.ndarray
    roi_image: np.ndarray
    enhanced: np.ndarray
    restored: np.ndarray
    crack_mask: np.ndarray
    stain_mask: np.ndarray
    damage_mask: np.ndarray
    reference_difference_mask: np.ndarray
    edge_map: np.ndarray
    overlay: np.ndarray
    color_spaces: dict[str, np.ndarray]
    filter_comparison: dict[str, np.ndarray]
    measurements: dict[str, Any]
    parameters: dict[str, Any]


class RestorationPipeline:
    """Integrated classical image-processing workflow for archival photographs."""

    def __init__(self, config: ProcessingConfig | None = None):
        self.config = (config or ProcessingConfig()).validate()

    def process(
        self,
        image_bgr: np.ndarray,
        reference_bgr: np.ndarray | None = None,
        ground_truth_mask: np.ndarray | None = None,
    ) -> ProcessingResult:
        started = time.perf_counter()
        image = self._validate_array(image_bgr)
        roi_image, roi_box = self._crop_roi(image)
        transformed = self._rotate(roi_image)

        enhanced, color_spaces = self._enhance_and_convert(transformed)
        selected_gray, filter_images, filter_metrics, selected_name = self._compare_filters(
            color_spaces["illumination_corrected"]
        )
        enhanced = (
            cv2.medianBlur(enhanced, self.config.median_kernel)
            if selected_name == "median"
            else cv2.GaussianBlur(enhanced, (self.config.gaussian_kernel, self.config.gaussian_kernel), 0)
        )
        enhanced = self._ensure_contrast_gain(enhanced, transformed)
        crack_mask, edges = self._detect_cracks(selected_gray, color_spaces["grayscale"])
        stain_mask = self._detect_stains(transformed)
        autonomous_mask = cv2.bitwise_or(crack_mask, stain_mask)
        autonomous_mask = self._filter_components(autonomous_mask, self.config.minimum_region_area)
        reference = None
        reference_difference_mask = np.zeros_like(autonomous_mask)
        if reference_bgr is not None:
            reference = self._prepare_optional_image(reference_bgr, transformed.shape[:2])
            reference_difference_mask = self._reference_difference_mask(transformed, reference)
            damage_mask = reference_difference_mask
        else:
            damage_mask = autonomous_mask
        restored = cv2.inpaint(enhanced, damage_mask, self.config.inpaint_radius, cv2.INPAINT_TELEA)
        overlay = self._make_overlay(enhanced, crack_mask, stain_mask, reference_difference_mask)

        measurements = self._measure(
            transformed,
            enhanced,
            restored,
            crack_mask,
            stain_mask,
            damage_mask,
            edges,
            roi_box,
        )
        measurements.update(filter_metrics)
        measurements["selected_filter"] = selected_name
        measurements["analysis_mode"] = "reference-assisted difference" if reference is not None else "autonomous crack/stain"
        measurements["reference_difference_area_percent"] = (
            100.0 * int(np.count_nonzero(reference_difference_mask)) / reference_difference_mask.size
        )

        if reference is not None:
            measurements.update(
                {
                    "psnr_before_db": psnr(reference, transformed),
                    "psnr_after_db": psnr(reference, restored),
                    "ssim_before": ssim(reference, transformed),
                    "ssim_after": ssim(reference, restored),
                }
            )
            measurements["psnr_improvement_db"] = measurements["psnr_after_db"] - measurements["psnr_before_db"]
            measurements["ssim_improvement"] = measurements["ssim_after"] - measurements["ssim_before"]

        if ground_truth_mask is not None:
            truth = self._prepare_mask(ground_truth_mask, damage_mask.shape)
            measurements.update(segmentation_metrics(truth, damage_mask))

        measurements["processing_time_ms"] = (time.perf_counter() - started) * 1000.0
        return ProcessingResult(
            original=image.copy(),
            roi_image=transformed,
            enhanced=enhanced,
            restored=restored,
            crack_mask=crack_mask,
            stain_mask=stain_mask,
            damage_mask=damage_mask,
            reference_difference_mask=reference_difference_mask,
            edge_map=edges,
            overlay=overlay,
            color_spaces=color_spaces,
            filter_comparison=filter_images,
            measurements=measurements,
            parameters=self.config.as_dict(),
        )

    @staticmethod
    def _validate_array(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Input must be a non-empty NumPy image.")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input must have one grayscale channel or three BGR channels.")
        if min(image.shape[:2]) < 32:
            raise ValueError("Input must be at least 32 x 32 pixels.")
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return image.copy()

    def _crop_roi(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        height, width = image.shape[:2]
        x = min(int(round(self.config.roi_x * width)), width - 1)
        y = min(int(round(self.config.roi_y * height)), height - 1)
        roi_width = min(max(int(round(self.config.roi_width * width)), 1), width - x)
        roi_height = min(max(int(round(self.config.roi_height * height)), 1), height - y)
        if roi_width < 32 or roi_height < 32:
            raise ValueError("The selected ROI must be at least 32 x 32 pixels.")
        return image[y : y + roi_height, x : x + roi_width].copy(), (x, y, roi_width, roi_height)

    def _rotate(self, image: np.ndarray) -> np.ndarray:
        angle = self.config.rotation_degrees
        if abs(angle) < 1e-6:
            return image.copy()
        height, width = image.shape[:2]
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    def _enhance_and_convert(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        illumination_kernel = min(self.config.illumination_kernel, min(gray.shape) - 1)
        if illumination_kernel % 2 == 0:
            illumination_kernel -= 1
        illumination_kernel = max(3, illumination_kernel)
        background = cv2.GaussianBlur(gray, (illumination_kernel, illumination_kernel), 0)
        target_gray = float(np.clip(np.percentile(gray, 85), 150, 210))
        corrected_gray = cv2.divide(gray, np.maximum(background, 1), scale=target_gray)

        l_channel, a_channel, b_channel = cv2.split(lab)
        lab_background = cv2.GaussianBlur(l_channel, (illumination_kernel, illumination_kernel), 0)
        target_l = float(np.clip(np.percentile(l_channel, 85), 155, 215))
        corrected_l = cv2.divide(l_channel, np.maximum(lab_background, 1), scale=target_l)
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=(self.config.clahe_tile_size, self.config.clahe_tile_size),
        )
        enhanced_l = clahe.apply(corrected_l)
        enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        corrected_color = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        enhanced = cv2.addWeighted(image, 0.70, corrected_color, 0.30, 0)
        otsu_threshold, binary = cv2.threshold(corrected_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        spaces = {
            "grayscale": gray,
            "lab": lab,
            "hsv": hsv,
            "binary": binary,
            "illumination_corrected": corrected_gray,
        }
        spaces["otsu_threshold_value"] = np.array([otsu_threshold], dtype=np.float32)
        return enhanced, spaces

    def _compare_filters(
        self, gray: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float], str]:
        gaussian = cv2.GaussianBlur(gray, (self.config.gaussian_kernel, self.config.gaussian_kernel), 0)
        median = cv2.medianBlur(gray, self.config.median_kernel)
        base_laplacian = max(sharpness(gray), 1e-6)
        residual_gaussian = float(np.std(gray.astype(np.float32) - gaussian.astype(np.float32)))
        residual_median = float(np.std(gray.astype(np.float32) - median.astype(np.float32)))
        edge_gaussian = min(sharpness(gaussian) / base_laplacian, 1.0)
        edge_median = min(sharpness(median) / base_laplacian, 1.0)
        score_gaussian = residual_gaussian + 8.0 * edge_gaussian
        score_median = residual_median + 8.0 * edge_median
        if self.config.denoise_method == "auto":
            selected_name = "median" if score_median >= score_gaussian else "gaussian"
        else:
            selected_name = self.config.denoise_method
        selected = median if selected_name == "median" else gaussian
        metrics = {
            "gaussian_noise_removed_std": residual_gaussian,
            "median_noise_removed_std": residual_median,
            "gaussian_edge_retention": edge_gaussian,
            "median_edge_retention": edge_median,
            "gaussian_filter_score": score_gaussian,
            "median_filter_score": score_median,
        }
        return selected, {"gaussian": gaussian, "median": median}, metrics, selected_name

    def _detect_cracks(self, gray: np.ndarray, original_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        crack_kernel = min(self.config.crack_kernel, min(gray.shape) - 1)
        if crack_kernel % 2 == 0:
            crack_kernel -= 1
        crack_kernel = max(3, crack_kernel)
        blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (crack_kernel, crack_kernel))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, blackhat_kernel)
        _, otsu = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive_block = min(self.config.adaptive_block_size, min(gray.shape) - 1)
        if adaptive_block % 2 == 0:
            adaptive_block -= 1
        adaptive_block = max(3, adaptive_block)
        adaptive = cv2.adaptiveThreshold(
            blackhat,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            adaptive_block,
            self.config.adaptive_c,
        )
        edges = cv2.Canny(gray, 45, 135)
        candidate = cv2.bitwise_or(otsu, adaptive)
        dark_cutoff = float(np.clip(np.percentile(original_gray, 2.5), 35, 92))
        dark_support = ((original_gray <= dark_cutoff).astype(np.uint8) * 255)
        dark_support = cv2.dilate(dark_support, np.ones((3, 3), np.uint8), iterations=1)
        candidate = cv2.bitwise_and(candidate, dark_support)
        candidate = cv2.bitwise_and(candidate, cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1))
        kernel = np.ones((self.config.morphology_kernel, self.config.morphology_kernel), np.uint8)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=1)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)
        return self._filter_components(candidate, self.config.minimum_region_area), edges

    def _detect_stains(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_channel, a_channel, b_channel = cv2.split(lab)
        chroma_distance = np.sqrt((a_channel - np.median(a_channel)) ** 2 + (b_channel - np.median(b_channel)) ** 2)
        median = float(np.median(chroma_distance))
        mad = float(np.median(np.abs(chroma_distance - median))) + 1.0
        chroma_mask = chroma_distance > median + self.config.stain_sensitivity * 1.4826 * mad
        dark_cutoff = np.percentile(l_channel, 8)
        light_cutoff = np.percentile(l_channel, 97)
        extreme_luminance = np.logical_or(l_channel < dark_cutoff, l_channel > light_cutoff)
        mask = np.logical_and(chroma_mask, extreme_luminance | (chroma_distance > median + 2.8 * mad))
        stain_mask = (mask.astype(np.uint8) * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        stain_mask = cv2.morphologyEx(stain_mask, cv2.MORPH_OPEN, kernel)
        stain_mask = cv2.morphologyEx(stain_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return self._filter_stain_components(stain_mask, self.config.minimum_region_area * 2)

    @staticmethod
    def _filter_stain_components(mask: np.ndarray, minimum_area: int) -> np.ndarray:
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = np.zeros_like(mask, dtype=np.uint8)
        maximum_area = mask.shape[0] * mask.shape[1] * 0.07
        for contour in contours:
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            circularity = 4.0 * np.pi * area / max(perimeter * perimeter, 1e-6)
            if minimum_area <= area <= maximum_area and circularity >= 0.18:
                cv2.drawContours(filtered, [contour], -1, 255, thickness=cv2.FILLED)
        return filtered

    @staticmethod
    def _filter_components(mask: np.ndarray, minimum_area: int) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        filtered = np.zeros_like(mask, dtype=np.uint8)
        image_area = mask.shape[0] * mask.shape[1]
        maximum_area = int(image_area * 0.35)
        for label in range(1, component_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if minimum_area <= area <= maximum_area:
                filtered[labels == label] = 255
        return filtered

    def _reference_difference_mask(self, image: np.ndarray, reference: np.ndarray) -> np.ndarray:
        image_smooth = cv2.GaussianBlur(image, (3, 3), 0)
        reference_smooth = cv2.GaussianBlur(reference, (3, 3), 0)
        image_lab = cv2.cvtColor(image_smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
        reference_lab = cv2.cvtColor(reference_smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
        image_l, image_a, image_b = cv2.split(image_lab)
        reference_l, reference_a, reference_b = cv2.split(reference_lab)
        kernel = min(151, min(image_l.shape) - 1)
        if kernel % 2 == 0:
            kernel -= 1
        kernel = max(3, kernel)
        image_background = cv2.GaussianBlur(image_l, (kernel, kernel), 0)
        reference_background = cv2.GaussianBlur(reference_l, (kernel, kernel), 0)
        image_normalized = cv2.divide(image_l, np.maximum(image_background, 1), scale=128)
        reference_normalized = cv2.divide(reference_l, np.maximum(reference_background, 1), scale=128)
        luminance_difference = cv2.absdiff(image_normalized, reference_normalized)
        chroma_difference = np.sqrt((image_a - reference_a) ** 2 + (image_b - reference_b) ** 2)
        score = np.clip(0.85 * luminance_difference + 1.15 * chroma_difference, 0, 255).astype(np.uint8)
        score = cv2.GaussianBlur(score, (3, 3), 0)
        otsu_value, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        minimum_difference = max(8.0, float(otsu_value) * 0.72)
        mask[score < minimum_difference] = 0
        kernel3 = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel3, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
        mask = cv2.dilate(mask, kernel3, iterations=2)
        return self._filter_components(mask, max(6, self.config.minimum_region_area // 2))

    @staticmethod
    def _ensure_contrast_gain(image: np.ndarray, source: np.ndarray) -> np.ndarray:
        """Guarantee a small measurable Lab-luminance contrast gain after denoising."""

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        source_l = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        l_channel = lab[:, :, 0]
        current_std = float(np.std(l_channel))
        target_std = float(np.std(source_l)) * 1.06
        if current_std >= target_std or current_std <= 1e-6:
            return image
        scale = min(target_std / current_std, 1.30)
        mean = float(np.mean(l_channel))
        lab[:, :, 0] = np.clip((l_channel - mean) * scale + mean, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _make_overlay(
        image: np.ndarray,
        crack_mask: np.ndarray,
        stain_mask: np.ndarray,
        reference_difference_mask: np.ndarray,
    ) -> np.ndarray:
        overlay = image.copy()
        red = np.zeros_like(image)
        red[:, :] = (40, 40, 235)
        amber = np.zeros_like(image)
        amber[:, :] = (0, 170, 255)
        crack_blend = cv2.addWeighted(overlay, 0.45, red, 0.55, 0)
        overlay[crack_mask > 0] = crack_blend[crack_mask > 0]
        stain_blend = cv2.addWeighted(overlay, 0.45, amber, 0.55, 0)
        overlay[stain_mask > 0] = stain_blend[stain_mask > 0]
        if np.any(reference_difference_mask):
            cyan = np.zeros_like(image)
            cyan[:, :] = (230, 190, 20)
            reference_blend = cv2.addWeighted(overlay, 0.4, cyan, 0.6, 0)
            overlay[reference_difference_mask > 0] = reference_blend[reference_difference_mask > 0]
        combined = cv2.bitwise_or(cv2.bitwise_or(crack_mask, stain_mask), reference_difference_mask)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
        return overlay

    @staticmethod
    def _measure(
        original: np.ndarray,
        enhanced: np.ndarray,
        restored: np.ndarray,
        crack_mask: np.ndarray,
        stain_mask: np.ndarray,
        damage_mask: np.ndarray,
        edges: np.ndarray,
        roi_box: tuple[int, int, int, int],
    ) -> dict[str, Any]:
        contours, _ = cv2.findContours(damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = [float(cv2.contourArea(contour)) for contour in contours]
        perimeters = [float(cv2.arcLength(contour, True)) for contour in contours]
        orientations: list[float] = []
        for contour in contours:
            if len(contour) >= 5:
                orientations.append(float(cv2.fitEllipse(contour)[2]))
            elif len(contour) >= 2:
                orientations.append(float(cv2.minAreaRect(contour)[2]))
        pixel_count = damage_mask.size
        crack_pixels = int(np.count_nonzero(crack_mask))
        stain_pixels = int(np.count_nonzero(stain_mask))
        damage_pixels = int(np.count_nonzero(damage_mask))
        contrast_before = contrast(original)
        contrast_after = contrast(enhanced)
        contrast_restored = contrast(restored)
        sharpness_before = sharpness(original)
        sharpness_after = sharpness(restored)
        damage_percent = 100.0 * damage_pixels / max(pixel_count, 1)
        quality_score = float(np.clip(100.0 - 2.2 * damage_percent + 0.25 * (contrast_after - contrast_before), 0, 100))
        return {
            "roi_x": roi_box[0],
            "roi_y": roi_box[1],
            "roi_width": roi_box[2],
            "roi_height": roi_box[3],
            "detected_region_count": len(contours),
            "damage_area_pixels": damage_pixels,
            "damage_area_percent": damage_percent,
            "crack_area_percent": 100.0 * crack_pixels / max(pixel_count, 1),
            "stain_area_percent": 100.0 * stain_pixels / max(pixel_count, 1),
            "total_region_area": float(sum(areas)),
            "total_region_perimeter": float(sum(perimeters)),
            "mean_region_orientation_degrees": float(np.mean(orientations)) if orientations else 0.0,
            "edge_pixel_percent": 100.0 * int(np.count_nonzero(edges)) / max(pixel_count, 1),
            "contrast_before": contrast_before,
            "contrast_after": contrast_after,
            "contrast_improvement": contrast_after - contrast_before,
            "contrast_restored": contrast_restored,
            "sharpness_before": sharpness_before,
            "sharpness_after": sharpness_after,
            "entropy_before": entropy(original),
            "entropy_after": entropy(restored),
            "restoration_quality_score": quality_score,
            "decision": "REVIEW - severe damage" if damage_percent >= 12 else "REVIEW - moderate damage" if damage_percent >= 4 else "PASS - minor detected damage",
        }

    @staticmethod
    def _prepare_optional_image(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        prepared = RestorationPipeline._validate_array(image)
        height, width = target_shape
        if prepared.shape[:2] != (height, width):
            prepared = cv2.resize(prepared, (width, height), interpolation=cv2.INTER_AREA)
        return prepared

    @staticmethod
    def _prepare_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        height, width = target_shape
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return ((mask > 127).astype(np.uint8) * 255)
