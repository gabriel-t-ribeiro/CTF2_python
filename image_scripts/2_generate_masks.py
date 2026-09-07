import argparse
from pathlib import Path

import cv2
import numpy as np


def segment_bubbles(
        img_path: Path, min_area: int = 1, max_area: int = 50000
) -> np.ndarray:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"OpenCV could not load image at: {img_path}")

    # Invert so bubble edges are bright peaks
    inverted = cv2.bitwise_not(img)

    # Dynamic Thresholding
    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
    _, binary_edges = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Fill bubble interiors
    contours, _ = cv2.findContours(
        binary_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled_mask = np.zeros_like(binary_edges)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    solid_bubbles = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel)

    # Filter by size (inclusive bounds)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        solid_bubbles, connectivity=8
    )
    clean_mask = np.zeros_like(solid_bubbles)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            clean_mask[labels == i] = 255

    return clean_mask


def generate_masks(
        input_dir: Path,
        output_dir: Path,
        min_area: int = 1,
        max_area: int = 50000,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}
    image_files = (
        sorted(
            [
                p
                for p in input_dir.iterdir()
                if p.is_file() and p.suffix.lower() in valid_exts
            ]
        )
        if input_dir.exists()
        else []
    )

    print(f"Checking for images in: {input_dir}")
    print(f"Found {len(image_files)} image(s) to process.")

    if len(image_files) == 0:
        print(
            f"\n[WARNING] No images found! Place your image files inside: {input_dir}"
        )
        return

    saved_count = 0
    for img_path in image_files:
        mask = segment_bubbles(img_path, min_area=min_area, max_area=max_area)
        out_path = output_dir / f"{img_path.stem}_mask.png"
        success = cv2.imwrite(str(out_path), mask)

        if success:
            saved_count += 1
            print(f"[SUCCESS] Saved mask: {out_path.name}")
        else:
            print(f"[ERROR] Failed to save mask to {out_path}")

    print(
        f"\n[DONE] Successfully generated {saved_count}/{len(image_files)} masks in {output_dir}"
    )


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate initial binary bubble masks from images"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=base_dir / "images_1",
        help="Input folder with flow images",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=base_dir / "generated_masks",
        help="Output folder for generated masks",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=1,
        help="Minimum bubble area in pixels (default: 1)",
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=50000,
        help="Maximum bubble area in pixels (default: 50000)",
    )
    args = parser.parse_args()

    generate_masks(
        args.input, args.output, min_area=args.min_area, max_area=args.max_area
    )
