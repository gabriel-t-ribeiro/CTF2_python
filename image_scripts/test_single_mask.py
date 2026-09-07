import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def segment_bubbles(img: np.ndarray, min_area: int = 1, max_area: int = 50000) -> np.ndarray:
    """Segment bubbles from a single grayscale image using Otsu thresholding and morphological filtering."""
    # 1. Invert so dark bubble borders become bright peaks
    inverted = cv2.bitwise_not(img)

    # 2. Dynamic thresholding (Gaussian blur + Otsu)
    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
    _, binary_edges = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Fill bubble interiors using contours
    contours, _ = cv2.findContours(binary_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(binary_edges)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

    # 4. Seal near-complete edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    solid_bubbles = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel)

    # 5. Filter components by area
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(solid_bubbles, connectivity=8)

    clean_mask = np.zeros_like(solid_bubbles)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            clean_mask[labels == i] = 255

    return clean_mask


def visualize_segmentation(
    img: np.ndarray,
    mask: np.ndarray,
    title: str = "Bubble Segmentation Test",
    save_path: Path | None = None,
    show: bool = True,
):
    """Plot original image, segmented mask, and an overlay side-by-side."""
    # Create RGB overlay
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), thickness=2)  # Red contours

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14)

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Generated Binary Mask")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (Detected Boundaries)")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[SUCCESS] Saved visualization to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Single-image test and visualization script for bubble segmentation"
    )
    parser.add_argument(
        "--image",
        "-i",
        type=Path,
        default=None,
        help="Path to the input image. If omitted, uses the first image found in images_1/",
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
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional path to save the visualization figure (e.g. test_result.png)",
    )
    parser.add_argument(
        "--save-mask",
        type=Path,
        default=None,
        help="Optional path to save the raw binary mask PNG",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable interactive display window (useful for headless / script execution)",
    )
    args = parser.parse_args()

    # Find image to test
    image_path = args.image
    if image_path is None:
        images_dir = base_dir / "images_1"
        valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}
        if images_dir.exists():
            candidates = sorted(
                [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
            )
            if candidates:
                image_path = candidates[0]

    if image_path is None or not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        print("Please provide a valid image using `--image /path/to/frame.png`")
        return

    print(f"Testing bubble segmentation on: {image_path}")
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] OpenCV could not read image at: {image_path}")
        return

    mask = segment_bubbles(img, min_area=args.min_area, max_area=args.max_area)

    num_bubbles = len(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    print(f"[DONE] Detected {num_bubbles} bubble component(s).")

    if args.save_mask:
        cv2.imwrite(str(args.save_mask), mask)
        print(f"[SUCCESS] Saved binary mask to {args.save_mask}")

    visualize_segmentation(
        img=img,
        mask=mask,
        title=f"Bubble Segmentation Test: {image_path.name} ({num_bubbles} bubbles)",
        save_path=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
