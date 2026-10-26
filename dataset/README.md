
## Included demonstration data

- 20 images in the `development` split;
- 10 images in the locked `unseen_test` split;
- a clean reference and exact synthetic damage mask for each image;
- controlled cracks, stains, dust, fading, noise, and uneven illumination.


Run the submission gate:

```bash
python tools/check_dataset.py
```

The command deliberately fails until the personal-image requirement is genuinely satisfied.
