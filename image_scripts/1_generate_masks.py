from pathlib import Path

import cv2
import numpy as np

# 1. Define paths dynamically
base_dir = Path(__file__).resolve().parent
input_dir = base_dir / "images_1"
output_dir = base_dir / "generated_masks"

# Automatically create the output folder if it does not exist
output_dir.mkdir(parents=True, exist_ok=True)


def segment_bubbles(img_path, min_area=1, max_area=50000):
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

    # Filter by size
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        solid_bubbles, connectivity=8
    )
    clean_mask = np.zeros_like(solid_bubbles)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area < area < max_area:
            clean_mask[labels == i] = 255

    return clean_mask


# 2. Process all images in the folder
image_files = (
    list(input_dir.glob("*.png"))
    + list(input_dir.glob("*.jpg"))
    + list(input_dir.glob("*.jpeg"))
)

print(f"Base Directory: {base_dir}")
print(f"Checking for images in: {input_dir}")
print(f"Found {len(image_files)} image(s) to process.")

if len(image_files) == 0:
    print(f"\n[WARNING] No images found! Place your image files inside: {input_dir}")
else:
    for img_path in image_files:
        mask = segment_bubbles(img_path)
        out_path = output_dir / f"{img_path.stem}_mask.png"
        success = cv2.imwrite(str(out_path), mask)

        if success:
            print(f"[SUCCESS] Saved mask: {out_path.name}")
        else:
            print(f"[ERROR] Failed to save mask to {out_path}")
