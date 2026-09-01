# Dataset protocol

The assignment requires at least 30 relevant images and at least 10 images captured or produced by the student. It also requires testing on at least 10 unseen images.

## Included demonstration data

`dataset/demo/` is generated deterministically by `tools/generate_demo_dataset.py`. It contains 30 synthetic heritage-style scenes:

- 20 images in the `development` split;
- 10 images in the locked `unseen_test` split;
- a clean reference and exact synthetic damage mask for each image;
- controlled cracks, stains, dust, fading, noise, and uneven illumination.

These files make the pipeline testable and the reported demo metrics reproducible. They are **not** student-captured evidence and must never be described as such.

## Required personal evidence before submission

Add at least 10 images that you personally captured or produced to `dataset/personal/`. Suitable inputs include photographs you take of permitted old family photographs, public monuments, inscriptions, paintings, artifacts, or controlled prints that you deliberately stain/scratch for a safe experiment.

For each personal image, append a row to `dataset/manifest.csv` and set:

- `source_type` to `student_captured` or `student_produced`;
- `student_created` to `true`;
- `creator` to your name and IT number;
- `resolution`, `format`, `test_purpose`, and `license_or_permission` accurately;
- `split` before tuning: use `development` or reserve it as `unseen_test`.

Do not include identifiable people without informed permission. Do not claim ownership of public-domain or third-party images. Keep the original files unchanged and use copies for processing.

Run the submission gate:

```bash
python tools/check_dataset.py
```

The command deliberately fails until the personal-image requirement is genuinely satisfied.
