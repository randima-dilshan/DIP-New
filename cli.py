from __future__ import annotations

import argparse
import json
from pathlib import Path

from heritage_restore.batch import process_folder
from heritage_restore.config import ProcessingConfig
from heritage_restore.export import json_safe_measurements, save_result
from heritage_restore.io import load_image
from heritage_restore.pipeline import RestorationPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HeritageGuard old-photograph restoration and damage analysis."
    )
    parser.add_argument("--denoise", choices=["auto", "median", "gaussian"], default="auto")
    parser.add_argument("--rotation", type=float, default=0.0)
    parser.add_argument("--clahe", type=float, default=1.5, help="CLAHE clip limit")
    parser.add_argument("--stain-sensitivity", type=float, default=2.7)
    parser.add_argument("--min-area", type=int, default=18)
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", help="Process one image")
    single.add_argument("input")
    single.add_argument("--output", default="output/single_run")
    single.add_argument("--reference")
    single.add_argument("--ground-truth-mask")

    batch = subparsers.add_parser("batch", help="Process every supported image in a folder")
    batch.add_argument("input_folder")
    batch.add_argument("--output", default="output/batch_run")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    config = ProcessingConfig(
        denoise_method=arguments.denoise,
        rotation_degrees=arguments.rotation,
        clahe_clip_limit=arguments.clahe,
        stain_sensitivity=arguments.stain_sensitivity,
        minimum_region_area=arguments.min_area,
    )
    pipeline = RestorationPipeline(config)
    if arguments.command == "batch":
        summary = process_folder(arguments.input_folder, arguments.output, pipeline)
        print(
            json.dumps(
                {
                    "processed": summary.processed,
                    "failed": summary.failed,
                    "summary_csv": str(summary.summary_csv),
                },
                indent=2,
            )
        )
        return 0 if summary.processed else 2

    image, metadata = load_image(arguments.input)
    reference = load_image(arguments.reference)[0] if arguments.reference else None
    truth_mask = load_image(arguments.ground_truth_mask)[0] if arguments.ground_truth_mask else None
    result = pipeline.process(image, reference, truth_mask)
    output_paths = save_result(result, arguments.output, Path(arguments.input).stem)
    print(
        json.dumps(
            {
                "metadata": metadata.as_dict(),
                "measurements": json_safe_measurements(result.measurements),
                "outputs": {key: str(path) for key, path in output_paths.items()},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
