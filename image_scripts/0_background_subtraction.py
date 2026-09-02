import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def compute_temporal_median_background(
    image_paths: list[Path], sample_size: int = 50
) -> np.ndarray:
    """
    Computes a clean static background image by calculating the temporal median
    across a representative sample of video frames.

    Why Median? Moving bubbles occupy any single pixel only momentarily.
    The median intensity cleanly filters out moving bubbles without leaving ghost shadows.
    """
    if not image_paths:
        raise ValueError("Image paths list is empty.")

    # Select evenly spaced sample frames across the sequence
    if len(image_paths) <= sample_size or sample_size <= 0:
        sampled_paths = image_paths
    else:
        step = max(1, len(image_paths) // sample_size)
        sampled_paths = image_paths[::step][:sample_size]

    print(f"Estimating static background from {len(sampled_paths)} frames...")

    frames = []
    for p in sampled_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            frames.append(img)

    if not frames:
        raise FileNotFoundError("Could not read any sample images.")

    if len(frames) == 1:
        return frames[0]

    # Stack along 3rd dimension and compute median per pixel
    stack = np.stack(frames, axis=2)
    bg_median = np.median(stack, axis=2).astype(np.uint8)
    return bg_median


def load_or_compute_background(
    input_images: list[Path],
    bg_source: Optional[Path] = None,
    sample_size: int = 50,
) -> np.ndarray:
    """
    Resolves the static background reference image:
      1. If bg_source is a single image file: loads it directly.
      2. If bg_source is a directory of background-only frames: computes median across them.
      3. If bg_source is None: computes temporal median across the input video sequence.
    """
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}

    if bg_source is not None:
        if bg_source.is_file():
            print(f"Loading dedicated background reference image: {bg_source}")
            bg = cv2.imread(str(bg_source), cv2.IMREAD_GRAYSCALE)
            if bg is None:
                raise ValueError(f"Could not load background image file: {bg_source}")
            return bg
        elif bg_source.is_dir():
            bg_files = sorted(
                [p for p in bg_source.iterdir() if p.suffix.lower() in valid_exts]
            )
            if not bg_files:
                raise FileNotFoundError(
                    f"No valid background images found in directory: {bg_source}"
                )
            print(
                f"Computing background reference from dedicated background folder: {bg_source} ({len(bg_files)} images)"
            )
            return compute_temporal_median_background(bg_files, sample_size=sample_size)
        else:
            raise FileNotFoundError(
                f"Specified background path does not exist: {bg_source}"
            )

    # Fallback to temporal median across video sequence
    print(
        f"No external background provided. Computing temporal median from sequence ({len(input_images)} frames)..."
    )
    return compute_temporal_median_background(input_images, sample_size=sample_size)


def process_background_subtraction(
    raw_img: np.ndarray,
    bg_img: np.ndarray,
    diff_threshold: int = 15,
    margin_radius_px: int = 10,
    fill_background_value: int = 255,
) -> np.ndarray:
    """
    Removes the static background while completely preserving:
      1. The bubble interior / glare point (via morphological hole filling).
      2. The outer defocus blur halo (via controlled dilation margin).

    Parameters:
      raw_img: Grayscale source frame.
      bg_img: Computed/loaded static background reference.
      diff_threshold: Minimum intensity difference to detect bubble boundary edges.
      margin_radius_px: Safety padding radius (in pixels) kept around the bubble to preserve blur.
      fill_background_value: Background replacement intensity (default 255 = clean white, or 0 = black).
    """
    # Resize background if dimensions differ slightly
    if raw_img.shape != bg_img.shape:
        bg_img = cv2.resize(bg_img, (raw_img.shape[1], raw_img.shape[0]))

    # 1. Compute absolute intensity difference
    abs_diff = cv2.absdiff(raw_img, bg_img)

    # 2. Threshold differences to isolate bubble boundary rings
    _, thresh = cv2.threshold(abs_diff, diff_threshold, 255, cv2.THRESH_BINARY)

    # 3. Morphological close to bridge tiny gaps in the contour
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

    # 4. Fill bubble interior holes (preserves the bright glare point in bubble center)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_bubble_mask = np.zeros_like(closed)
    cv2.drawContours(filled_bubble_mask, contours, -1, 255, thickness=cv2.FILLED)

    # 5. Expand boundary by margin_radius_px (preserves the subtle outer defocus blur)
    if margin_radius_px > 0:
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * margin_radius_px + 1, 2 * margin_radius_px + 1)
        )
        expanded_bubble_mask = cv2.dilate(filled_bubble_mask, dilate_kernel)
    else:
        expanded_bubble_mask = filled_bubble_mask

    # 6. Apply mask: retain original raw pixels inside the expanded bubble envelope,
    #    and set background pixels to clean uniform intensity.
    cleaned_img = np.full_like(raw_img, fill_background_value)
    cleaned_img[expanded_bubble_mask == 255] = raw_img[expanded_bubble_mask == 255]

    return cleaned_img


def run_batch_background_removal(
    input_dir: Path,
    output_dir: Path,
    bg_source: Optional[Path] = None,
    sample_size: int = 50,
    diff_threshold: int = 15,
    margin_px: int = 10,
    bg_value: int = 255,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}
    image_files = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in valid_exts]
    )

    if not image_files:
        raise FileNotFoundError(f"No image files found in {input_dir}")

    # 1. Resolve or compute background reference
    bg_ref = load_or_compute_background(
        input_images=image_files,
        bg_source=bg_source,
        sample_size=sample_size,
    )
    bg_save_path = output_dir / "static_background_reference.png"
    cv2.imwrite(str(bg_save_path), bg_ref)
    print(f"[SAVED] Background reference saved to: {bg_save_path}")

    # 2. Process all frames
    print(f"Cleaning background across {len(image_files)} frames -> '{output_dir}'...")
    for i, p in enumerate(image_files, 1):
        raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue

        cleaned = process_background_subtraction(
            raw,
            bg_ref,
            diff_threshold=diff_threshold,
            margin_radius_px=margin_px,
            fill_background_value=bg_value,
        )

        out_path = output_dir / p.name
        cv2.imwrite(str(out_path), cleaned)

        if i % 100 == 0 or i == len(image_files):
            print(f"  Processed [{i}/{len(image_files)}]")

    print(
        f"\n[DONE] Background removal complete. Cleaned frames saved to: {output_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean static background from flow boiling videos while protecting bubble cores and blur"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input folder with raw cropped frames",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output folder for cleaned frames",
    )
    parser.add_argument(
        "--bg-source",
        "--bg",
        type=Path,
        default=None,
        help="Optional path to a dedicated background image file OR a folder containing background-only frames. "
        "If not provided, computes temporal median automatically from --input.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of frames sampled when computing background median (default: 50)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=int,
        default=15,
        help="Intensity difference threshold (0-255) for bubble edge detection (default: 15)",
    )
    parser.add_argument(
        "--margin",
        "-m",
        type=int,
        default=10,
        help="Padding margin in pixels around bubble to preserve defocus blur (default: 10)",
    )
    parser.add_argument(
        "--bg-value",
        type=int,
        default=255,
        help="Fill intensity for removed background (default: 255 = white, 0 = black)",
    )
    args = parser.parse_args()

    run_batch_background_removal(
        input_dir=args.input,
        output_dir=args.output,
        bg_source=args.bg_source,
        sample_size=args.sample_size,
        diff_threshold=args.threshold,
        margin_px=args.margin,
        bg_value=args.bg_value,
    )
