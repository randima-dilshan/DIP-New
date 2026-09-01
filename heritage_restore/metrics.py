from __future__ import annotations

import math

import cv2
import numpy as np


def grayscale(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def contrast(image: np.ndarray) -> float:
    return float(np.std(grayscale(image)))


def sharpness(image: np.ndarray) -> float:
    return float(cv2.Laplacian(grayscale(image), cv2.CV_64F).var())


def entropy(image: np.ndarray) -> float:
    gray = grayscale(image)
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probabilities = histogram / max(float(histogram.sum()), 1.0)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = reference.astype(np.float64)
    cand = candidate.astype(np.float64)
    mse = float(np.mean((ref - cand) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Channel-averaged structural similarity with an 11x11 Gaussian window."""

    if reference.shape != candidate.shape:
        raise ValueError("SSIM inputs must have the same shape.")
    ref = reference.astype(np.float64)
    cand = candidate.astype(np.float64)
    channels = 1 if ref.ndim == 2 else ref.shape[2]
    values: list[float] = []
    for index in range(channels):
        x = ref if channels == 1 else ref[:, :, index]
        y = cand if channels == 1 else cand[:, :, index]
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        mu_x_sq, mu_y_sq, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sigma_x_sq = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x_sq
        sigma_y_sq = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y_sq
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        values.append(float(np.mean(numerator / np.maximum(denominator, 1e-12))))
    return float(np.mean(values))


def segmentation_metrics(truth_mask: np.ndarray, predicted_mask: np.ndarray) -> dict[str, float]:
    truth = grayscale(truth_mask) > 127
    predicted = grayscale(predicted_mask) > 127
    if truth.shape != predicted.shape:
        raise ValueError("Ground-truth and predicted masks must have the same dimensions.")
    tp = int(np.logical_and(truth, predicted).sum())
    fp = int(np.logical_and(~truth, predicted).sum())
    fn = int(np.logical_and(truth, ~predicted).sum())
    tn = int(np.logical_and(~truth, ~predicted).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {
        "segmentation_precision": float(precision),
        "segmentation_recall": float(recall),
        "segmentation_f1": float(f1),
        "segmentation_iou": float(iou),
        "segmentation_accuracy": float(accuracy),
    }
