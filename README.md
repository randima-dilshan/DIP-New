# HeritageGuard

HeritageGuard is a practical Heritage & Culture image-analysis system for SE3071 Digital Image Processing. It solves one specific problem: **detect cracks and stains in digitized old photographs, restore the detected regions, and produce a measurable damage decision and reproducible quality summary**.

The assessed core uses Python and classical OpenCV. There is no deep-learning dependency.

## What the application does

1. Loads one image, multiple uploaded images, or every supported image in a folder.
2. Rejects missing, corrupted, oversized, tiny, or unsupported files with a clear message.
3. Records filename, format, dimensions, channels, bit depth, file size, and selected ROI.
4. Applies ROI cropping and optional rotation correction.
5. Computes Grayscale, Lab, HSV, and Otsu-binary representations.
6. Corrects uneven illumination and improves contrast with CLAHE.
7. compares Gaussian and median filtering, records edge retention/noise removal, and selects one automatically or by user choice.
8. Detects cracks using black-hat morphology, Otsu/adaptive thresholding, Canny edges, closing/opening, and connected components.
9. Detects stains using robust Lab color-distance segmentation and morphological cleanup.
10. Extracts contour area, perimeter, orientation, region count, edge density, contrast, sharpness, and entropy.
11. Restores detected regions with Telea inpainting.
12. Produces a damage percentage, quality score, highlighted overlay, and pass/review decision.
13. Optionally uses a matching clean reference for locally normalized Lab difference detection and PSNR/SSIM; a mask enables accuracy, precision, recall, F1, and IoU.
14. Exports processed images, masks, JSON, CSV, and ZIP results for one image or a batch.

## Quick start

Python 3.10 or later is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/generate_demo_dataset.py
streamlit run app.py
```

Open the local address printed by Streamlit. Upload `dataset/demo/damaged/demo_001_damaged.png`, its clean reference, and its mask to see every evaluation function.

## Command-line operation

Process one image with optional objective evaluation:

```bash
python cli.py single dataset/demo/damaged/demo_001_damaged.png \
  --reference dataset/demo/clean/demo_001_clean.png \
  --ground-truth-mask dataset/demo/masks/demo_001_mask.png \
  --output output/single_run
```

Process an input folder:

```bash
python cli.py batch dataset/demo/damaged --output output/batch_run
```

Run the fixed ten-image unseen synthetic evaluation:

```bash
python tools/run_demo_evaluation.py
```

## Dataset and academic-integrity gate

The included 30-image dataset is synthetic, openly labeled, and designed for repeatable testing. It contains 20 development and 10 unseen-test image/reference/mask sets. It is not evidence that the student personally captured or produced images.

Before submission, add at least ten genuinely personal images and complete `dataset/manifest.csv` as described in `dataset/README.md`. Then run:

```bash
python tools/check_dataset.py
```

This check intentionally remains red until the personal-image requirement is honestly fulfilled. Never replace the synthetic label or fabricate personal evidence or evaluation results.

## Verification

```bash
pytest
python tools/run_demo_evaluation.py
python tools/check_dataset.py
```

Generated demo images, evaluation results, processed outputs, course documents, reports, screenshots, and personal images are intentionally not versioned. Recreate the demo dataset and evaluation locally with the commands above.

## Supported formats and limitations

Supported files are JPG/JPEG, PNG, BMP, TIFF, and TIF up to 40 MB and at least 32 x 32 pixels. Detection is designed for thin dark cracks and chromatic stains. It can confuse legitimate dark lines, handwriting, strong borders, or sepia regions with damage. Inpainting estimates missing pixels; it cannot recover historically authentic detail that no longer exists. Results are preservation-support indicators, not proof of authenticity.

## Repository contents

- Source: `app.py`, `cli.py`, and `heritage_restore/`
- Interface: Streamlit single and batch workflows
- Dataset protocol and local-data placeholders: `dataset/`
- Reproducible data/evaluation utilities: `tools/`
- Dependencies: `requirements.txt`
- Executable instructions: this README
