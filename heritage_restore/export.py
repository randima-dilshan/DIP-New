from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .io import image_to_png_bytes, mask_to_png_bytes
from .pipeline import ProcessingResult


def json_safe_measurements(measurements: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in measurements.items():
        if isinstance(value, (np.floating, np.integer)):
            safe[key] = value.item()
        else:
            safe[key] = value
    return safe


def save_result(result: ProcessingResult, output_dir: str | Path, stem: str) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    safe_stem = "".join(character if character.isalnum() or character in "-_" else "_" for character in stem)
    paths = {
        "restored": destination / f"{safe_stem}_restored.png",
        "enhanced": destination / f"{safe_stem}_enhanced.png",
        "overlay": destination / f"{safe_stem}_damage_overlay.png",
        "damage_mask": destination / f"{safe_stem}_damage_mask.png",
        "reference_difference_mask": destination / f"{safe_stem}_reference_difference_mask.png",
        "crack_mask": destination / f"{safe_stem}_crack_mask.png",
        "stain_mask": destination / f"{safe_stem}_stain_mask.png",
        "metrics": destination / f"{safe_stem}_metrics.json",
    }
    image_outputs = {
        "restored": result.restored,
        "enhanced": result.enhanced,
        "overlay": result.overlay,
        "damage_mask": result.damage_mask,
        "reference_difference_mask": result.reference_difference_mask,
        "crack_mask": result.crack_mask,
        "stain_mask": result.stain_mask,
    }
    for name, image in image_outputs.items():
        if not cv2.imwrite(str(paths[name]), image):
            raise OSError(f"Failed to save {paths[name]}.")
    payload = {
        "measurements": json_safe_measurements(result.measurements),
        "parameters": result.parameters,
    }
    paths["metrics"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return paths


def write_summary_csv(rows: Iterable[dict[str, Any]], output_path: str | Path) -> Path:
    rows_list = [dict(row) for row in rows]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows_list:
        path.write_text("status,message\nempty,No valid images were processed\n", encoding="utf-8")
        return path
    fieldnames: list[str] = []
    for row in rows_list:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_list)
    return path


def result_zip_bytes(result: ProcessingResult, stem: str, metadata: dict[str, Any] | None = None) -> bytes:
    buffer = io.BytesIO()
    safe_stem = "".join(character if character.isalnum() or character in "-_" else "_" for character in stem)
    payload = {
        "metadata": metadata or {},
        "measurements": json_safe_measurements(result.measurements),
        "parameters": result.parameters,
    }
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{safe_stem}_restored.png", image_to_png_bytes(result.restored))
        archive.writestr(f"{safe_stem}_enhanced.png", image_to_png_bytes(result.enhanced))
        archive.writestr(f"{safe_stem}_damage_overlay.png", image_to_png_bytes(result.overlay))
        archive.writestr(f"{safe_stem}_damage_mask.png", mask_to_png_bytes(result.damage_mask))
        archive.writestr(f"{safe_stem}_reference_difference_mask.png", mask_to_png_bytes(result.reference_difference_mask))
        archive.writestr(f"{safe_stem}_edge_map.png", mask_to_png_bytes(result.edge_map))
        archive.writestr(f"{safe_stem}_result.json", json.dumps(payload, indent=2))
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=list(payload["measurements"].keys()))
        writer.writeheader()
        writer.writerow(payload["measurements"])
        archive.writestr(f"{safe_stem}_summary.csv", csv_buffer.getvalue())
    return buffer.getvalue()
