from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .export import save_result, write_summary_csv
from .io import ImageValidationError, SUPPORTED_EXTENSIONS, load_image
from .pipeline import RestorationPipeline


@dataclass(frozen=True)
class BatchSummary:
    processed: int
    failed: int
    rows: list[dict[str, object]]
    summary_csv: Path


def process_folder(
    input_folder: str | Path,
    output_folder: str | Path,
    pipeline: RestorationPipeline,
) -> BatchSummary:
    source = Path(input_folder)
    if not source.is_dir():
        raise ValueError(f"Input folder does not exist: {source}")
    destination = Path(output_folder)
    destination.mkdir(parents=True, exist_ok=True)
    candidates = sorted(path for path in source.iterdir() if path.is_file())
    rows: list[dict[str, object]] = []
    processed = 0
    failed = 0
    for path in candidates:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            rows.append({"filename": path.name, "status": "skipped", "error": "unsupported extension"})
            failed += 1
            continue
        try:
            image, metadata = load_image(path)
            result = pipeline.process(image)
            save_result(result, destination / path.stem, path.stem)
            row: dict[str, object] = {"filename": path.name, "status": "processed"}
            row.update(metadata.as_dict())
            row.update(result.measurements)
            rows.append(row)
            processed += 1
        except (ImageValidationError, ValueError, OSError, RuntimeError) as exc:
            rows.append({"filename": path.name, "status": "failed", "error": str(exc)})
            failed += 1
    summary_path = write_summary_csv(rows, destination / "batch_summary.csv")
    return BatchSummary(processed=processed, failed=failed, rows=rows, summary_csv=summary_path)
