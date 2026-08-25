from pathlib import Path

import cv2
import numpy as np

base_dir = Path(__file__).resolve().parent

image_path = (
    base_dir
    / "images_1"
    / "Test_run_21_2_51200kHz_div_lens_4_2000fps_C001H001S0001001179.png"
)

print(f"Looking for image at: {image_path}")
print(f"File exists: {image_path.exists()}")


def segment_bubbles(image_path, min_area=1, max_area=50000):
    # 1. Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 2. Invert so dark bubble borders become bright peaks
    inverted = cv2.bitwise_not(img)

    # 3. Dynamic Thresholding (Otsu + Gaussian Blur)
    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
    _, binary_edges = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 4. Fill bubble interiors using contours (replaces cv2.floodFill)
    contours, _ = cv2.findContours(
        binary_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filled_mask = np.zeros_like(binary_edges)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

    # Seal near-complete edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    solid_bubbles = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel)

    # 5. Filter out noise specks and separate bulk wave by area/shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        solid_bubbles, connectivity=8
    )

    clean_mask = np.zeros_like(solid_bubbles)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Keep only components within bubble size range
        if min_area < area < max_area:
            clean_mask[labels == i] = 255

    return img, clean_mask


# Test on your image
orig, mask = segment_bubbles(str(image_path))

cv2.imwrite("generated_bubble_mask.png", mask)
