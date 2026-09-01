from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from heritage_restore.batch import process_folder
from heritage_restore.pipeline import RestorationPipeline


def test_batch_processes_valid_and_reports_unsupported(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    image = np.full((96, 128, 3), 180, dtype=np.uint8)
    cv2.line(image, (10, 10), (100, 80), (15, 15, 15), 3)
    assert cv2.imwrite(str(input_dir / "valid.png"), image)
    (input_dir / "notes.txt").write_text("not an image", encoding="utf-8")
    summary = process_folder(input_dir, output_dir, RestorationPipeline())
    assert summary.processed == 1
    assert summary.failed == 1
    assert summary.summary_csv.is_file()
    assert (output_dir / "valid" / "valid_restored.png").is_file()
    assert any(row["status"] == "skipped" for row in summary.rows)
