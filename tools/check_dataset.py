from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED_FIELDS = {
    "image_id",
    "split",
    "input_path",
    "source_type",
    "student_created",
    "resolution",
    "format",
    "test_purpose",
    "license_or_permission",
}


def check(manifest_path: Path, project_root: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not manifest_path.is_file():
        return False, [f"Manifest is missing: {manifest_path}"]
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not REQUIRED_FIELDS.issubset(reader.fieldnames):
            missing = sorted(REQUIRED_FIELDS - set(reader.fieldnames or []))
            return False, [f"Manifest columns missing: {', '.join(missing)}"]
        rows = list(reader)
    if len(rows) < 30:
        errors.append(f"Dataset has {len(rows)} manifest rows; at least 30 are required.")
    unseen = [row for row in rows if row["split"].strip().lower() == "unseen_test"]
    if len(unseen) < 10:
        errors.append(f"Unseen-test split has {len(unseen)} images; at least 10 are required.")
    personal = [
        row
        for row in rows
        if row["student_created"].strip().lower() == "true"
        and row["source_type"].strip().lower() in {"student_captured", "student_produced"}
    ]
    if len(personal) < 10:
        errors.append(
            f"Only {len(personal)} images are certified as personally captured/produced; at least 10 are required before submission."
        )
    ids: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        image_id = row["image_id"].strip()
        if not image_id:
            errors.append(f"Row {line_number}: image_id is empty.")
        elif image_id in ids:
            errors.append(f"Row {line_number}: duplicate image_id '{image_id}'.")
        ids.add(image_id)
        input_path = project_root / row["input_path"]
        if not input_path.is_file():
            errors.append(f"Row {line_number}: input image not found: {row['input_path']}")
        for field in ("resolution", "format", "test_purpose", "license_or_permission"):
            if not row[field].strip():
                errors.append(f"Row {line_number}: {field} is empty.")
    return not errors, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SE3071 dataset and unseen-test compliance.")
    parser.add_argument("--manifest", default=Path("dataset/manifest.csv"), type=Path)
    parser.add_argument("--project-root", default=Path("."), type=Path)
    args = parser.parse_args()
    compliant, errors = check(args.manifest, args.project_root)
    if compliant:
        print("PASS: dataset satisfies the recorded assignment checks.")
        return 0
    print("NOT SUBMISSION-READY:")
    for error in errors:
        print(f"- {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
