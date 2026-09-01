from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


WIDTH, HEIGHT = 640, 480
COUNT = 30


def paper_texture(rng: np.random.Generator) -> Image.Image:
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    radial = np.sqrt(((x - WIDTH / 2) / WIDTH) ** 2 + ((y - HEIGHT / 2) / HEIGHT) ** 2)
    base = 224 - 40 * radial + rng.normal(0, 3.0, (HEIGHT, WIDTH))
    rgb = np.stack((base + 12, base + 3, base - 12), axis=2)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


def draw_archival_scene(image: Image.Image, index: int, rng: np.random.Generator) -> Image.Image:
    draw = ImageDraw.Draw(image)
    ink = (70, 58, 43)
    accent = (110, 83, 55)
    margin = 26
    draw.rectangle((margin, margin, WIDTH - margin, HEIGHT - margin), outline=accent, width=3)
    draw.rectangle((margin + 7, margin + 7, WIDTH - margin - 7, HEIGHT - margin - 7), outline=(145, 115, 78), width=1)
    draw.text((48, 44), f"CEYLON HERITAGE ARCHIVE - PLATE {index + 1:02d}", fill=ink)
    horizon = 325 + int(rng.integers(-12, 13))
    draw.line((55, horizon, WIDTH - 55, horizon), fill=ink, width=3)

    if index % 3 == 0:
        # Temple facade and columns.
        draw.polygon([(145, 190), (320, 95), (495, 190)], outline=ink, fill=(193, 170, 126))
        draw.rectangle((165, 190, 475, horizon), outline=ink, fill=(205, 186, 145), width=3)
        for column_x in range(190, 455, 52):
            draw.rectangle((column_x, 205, column_x + 22, horizon), outline=ink, fill=(224, 210, 175), width=2)
        draw.ellipse((275, 130, 365, 185), outline=ink, width=3)
    elif index % 3 == 1:
        # Historic railway station.
        draw.rectangle((120, 180, 520, horizon), outline=ink, fill=(199, 178, 134), width=3)
        draw.polygon([(95, 180), (320, 105), (545, 180)], outline=ink, fill=(174, 142, 97))
        draw.ellipse((285, 136, 355, 206), outline=ink, fill=(221, 205, 164), width=3)
        for window_x in (155, 225, 385, 455):
            draw.rectangle((window_x, 220, window_x + 42, 285), outline=ink, fill=(119, 137, 133), width=2)
        draw.line((60, 365, 580, 365), fill=ink, width=3)
        draw.line((60, 390, 580, 390), fill=ink, width=3)
    else:
        # Palm-lined colonial-era house without real people.
        draw.rectangle((185, 205, 480, horizon), outline=ink, fill=(211, 193, 153), width=3)
        draw.polygon([(160, 205), (330, 120), (505, 205)], outline=ink, fill=(161, 124, 80))
        for window_x in (220, 300, 405):
            draw.rectangle((window_x, 235, window_x + 45, 300), outline=ink, fill=(126, 145, 136), width=2)
        for palm_x in (95, 545):
            draw.line((palm_x, horizon, palm_x + 15, 160), fill=(69, 78, 49), width=7)
            for direction in (-1, 1):
                leaf_x = palm_x + 50 * direction
                draw.arc((min(palm_x, leaf_x) - 20, 120, max(palm_x, leaf_x) + 20, 220), 190, 340, fill=(69, 92, 52), width=5)

    draw.text((60, 412), "Digitized preservation study - synthetic teaching specimen", fill=accent)
    for _ in range(90):
        x = int(rng.integers(35, WIDTH - 35))
        y = int(rng.integers(35, HEIGHT - 35))
        gray = int(rng.integers(110, 190))
        draw.point((x, y), fill=(gray + 15, gray + 5, max(0, gray - 10)))
    return image.filter(ImageFilter.GaussianBlur(radius=0.25))


def add_damage(clean: Image.Image, index: int, rng: np.random.Generator) -> tuple[Image.Image, Image.Image]:
    damaged = clean.copy()
    damage_draw = ImageDraw.Draw(damaged, "RGBA")
    mask = Image.new("L", clean.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # Dark, branching cracks.
    crack_count = 4 + index % 5
    for _ in range(crack_count):
        start_x = int(rng.integers(50, WIDTH - 50))
        start_y = int(rng.integers(45, HEIGHT - 45))
        points = [(start_x, start_y)]
        for _segment in range(int(rng.integers(3, 7))):
            x = int(np.clip(points[-1][0] + rng.integers(-42, 43), 32, WIDTH - 32))
            y = int(np.clip(points[-1][1] + rng.integers(15, 55), 32, HEIGHT - 32))
            points.append((x, y))
        width = int(rng.integers(2, 5))
        damage_draw.line(points, fill=(38, 25, 18, 225), width=width)
        mask_draw.line(points, fill=255, width=width + 4)
        if rng.random() > 0.45:
            branch_at = points[len(points) // 2]
            branch_end = (int(np.clip(branch_at[0] + rng.integers(-35, 36), 20, WIDTH - 20)), branch_at[1] + 30)
            damage_draw.line((branch_at, branch_end), fill=(45, 29, 19, 220), width=max(2, width - 1))
            mask_draw.line((branch_at, branch_end), fill=255, width=width + 3)

    # Brown stains and faded circular damage.
    for _ in range(2 + index % 3):
        center_x = int(rng.integers(70, WIDTH - 70))
        center_y = int(rng.integers(70, HEIGHT - 70))
        radius_x = int(rng.integers(10, 29))
        radius_y = int(rng.integers(8, 25))
        box = (center_x - radius_x, center_y - radius_y, center_x + radius_x, center_y + radius_y)
        damage_draw.ellipse(box, fill=(135, 55, 12, int(rng.integers(150, 216))))
        mask_draw.ellipse(box, fill=255)

    # Small dust defects.
    for _ in range(12 + index % 7):
        x = int(rng.integers(35, WIDTH - 35))
        y = int(rng.integers(35, HEIGHT - 35))
        radius = int(rng.integers(1, 4))
        damage_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(30, 24, 19, 220))
        mask_draw.ellipse((x - radius - 1, y - radius - 1, x + radius + 1, y + radius + 1), fill=255)

    damaged = ImageEnhance.Contrast(damaged).enhance(0.84 + 0.03 * (index % 4))
    damaged = ImageEnhance.Brightness(damaged).enhance(0.90 + 0.025 * (index % 3))
    array = np.asarray(damaged).astype(np.float32)
    gradient = np.linspace(0.86, 1.08, WIDTH, dtype=np.float32)[None, :, None]
    if index % 2:
        gradient = gradient[:, ::-1]
    noise = rng.normal(0, 3.0 + index % 3, array.shape)
    array = np.clip(array * gradient + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(array, "RGB"), mask.filter(ImageFilter.MaxFilter(3))


def generate(destination: Path, count: int = COUNT) -> Path:
    clean_dir = destination / "demo" / "clean"
    damaged_dir = destination / "demo" / "damaged"
    masks_dir = destination / "demo" / "masks"
    for directory in (clean_dir, damaged_dir, masks_dir):
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.csv"
    fieldnames = [
        "image_id",
        "split",
        "input_path",
        "clean_reference_path",
        "ground_truth_mask_path",
        "source_type",
        "student_created",
        "creator",
        "resolution",
        "format",
        "test_purpose",
        "license_or_permission",
        "notes",
    ]
    rows: list[dict[str, str]] = []
    for index in range(count):
        rng = np.random.default_rng(20260902 + index * 97)
        clean = draw_archival_scene(paper_texture(rng), index, rng)
        damaged, mask = add_damage(clean, index, rng)
        image_id = f"demo_{index + 1:03d}"
        clean_path = clean_dir / f"{image_id}_clean.png"
        damaged_path = damaged_dir / f"{image_id}_damaged.png"
        mask_path = masks_dir / f"{image_id}_mask.png"
        clean.save(clean_path, format="PNG")
        damaged.save(damaged_path, format="PNG")
        mask.save(mask_path, format="PNG")
        split = "development" if index < max(0, count - 10) else "unseen_test"
        rows.append(
            {
                "image_id": image_id,
                "split": split,
                "input_path": damaged_path.relative_to(destination.parent).as_posix(),
                "clean_reference_path": clean_path.relative_to(destination.parent).as_posix(),
                "ground_truth_mask_path": mask_path.relative_to(destination.parent).as_posix(),
                "source_type": "synthetic_demo",
                "student_created": "false",
                "creator": "HeritageGuard deterministic demo generator",
                "resolution": f"{WIDTH}x{HEIGHT}",
                "format": "PNG",
                "test_purpose": "parameter development" if split == "development" else "unseen synthetic evaluation",
                "license_or_permission": "Project-generated; no identifiable person or third-party image",
                "notes": "Synthetic heritage-style scene with controlled crack, stain, dust, fade, noise, and illumination defects",
            }
        )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create transparent synthetic demo data for HeritageGuard.")
    parser.add_argument("--destination", default="dataset", type=Path)
    parser.add_argument("--count", default=COUNT, type=int)
    args = parser.parse_args()
    if args.count < 10:
        parser.error("At least 10 images are required so an unseen-test split can be created.")
    manifest = generate(args.destination, args.count)
    print(f"Generated {args.count} synthetic image/reference/mask triplets and {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
