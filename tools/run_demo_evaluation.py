from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from heritage_restore.config import ProcessingConfig
from heritage_restore.export import save_result, write_summary_csv
from heritage_restore.io import load_image
from heritage_restore.pipeline import RestorationPipeline


METRICS = [
    "segmentation_iou",
    "segmentation_precision",
    "segmentation_recall",
    "segmentation_f1",
    "segmentation_accuracy",
    "psnr_before_db",
    "psnr_after_db",
    "psnr_improvement_db",
    "ssim_before",
    "ssim_after",
    "ssim_improvement",
    "processing_time_ms",
    "damage_area_percent",
    "contrast_improvement",
]


def main() -> int:
    project_root = PROJECT_ROOT
    manifest_path = project_root / "dataset" / "manifest.csv"
    output_dir = project_root / "output" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "unseen_test"]
    if len(rows) < 10:
        raise RuntimeError("The manifest must contain at least 10 unseen-test images.")
    pipeline = RestorationPipeline(ProcessingConfig())
    results: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        damaged = load_image(project_root / row["input_path"])[0]
        clean = load_image(project_root / row["clean_reference_path"])[0]
        mask = load_image(project_root / row["ground_truth_mask_path"])[0]
        result = pipeline.process(damaged, clean, mask)
        record: dict[str, object] = {
            "image_id": row["image_id"],
            "split": row["split"],
            "source_type": row["source_type"],
        }
        record.update(result.measurements)
        results.append(record)
        if index < 3:
            save_result(result, output_dir / "representative_samples", row["image_id"])
    write_summary_csv(results, output_dir / "unseen_test_results.csv")
    aggregates: dict[str, object] = {
        "dataset_label": "synthetic demonstration only",
        "sample_size": len(results),
        "split": "unseen_test",
        "parameters": pipeline.config.as_dict(),
    }
    for metric in METRICS:
        values = np.array([float(row[metric]) for row in results if metric in row], dtype=np.float64)
        aggregates[f"mean_{metric}"] = float(values.mean())
        aggregates[f"std_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    (output_dir / "aggregate_metrics.json").write_text(json.dumps(aggregates, indent=2), encoding="utf-8")
    print(json.dumps(aggregates, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
