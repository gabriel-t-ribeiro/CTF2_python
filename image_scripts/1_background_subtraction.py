#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_sample_range(
    total_count: int, sample_range: str | None = None, sample_size: int = 50
) -> list[int]:
    """
    Determine the frame indices to sample for background median estimation.
    If sample_range is provided (e.g. '0,80' or '100,50'):
      - 'start,count': takes 'count' consecutive frames starting at index 'start'.
      - 'start:end': takes frames from index 'start' to 'end'.
    If sample_range is None:
      - Takes 'sample_size' evenly spaced frames across the sequence.
    """
    if total_count <= 0:
        return []

    if sample_range:
        val = sample_range.strip()
        if ":" in val:
            parts = [int(p.strip()) for p in val.split(":")]
            start = max(0, min(total_count - 1, parts[0]))
            end = max(start + 1, min(total_count, parts[1]))
            indices = list(range(start, end))
        elif "," in val:
            parts = [int(p.strip()) for p in val.split(",")]
            start = max(0, min(total_count - 1, parts[0]))
            count = max(1, parts[1])
            end = min(total_count, start + count)
            indices = list(range(start, end))
        else:
            # Single integer: treat as count from start 0
            count = max(1, min(total_count, int(val)))
            indices = list(range(0, count))
        return indices

    # Default: evenly spaced frames
    if total_count <= sample_size or sample_size <= 0:
        return list(range(total_count))
    step = max(1, total_count // sample_size)
    return list(range(0, total_count, step))[:sample_size]


def compute_temporal_median_background(
    image_paths: list[Path], sample_size: int = 50, sample_range: str | None = None
) -> np.ndarray:
    """
    Computes a clean static background image by calculating the temporal median
    across a sample of video frames.

    Why Median? Moving bubbles occupy any single pixel only momentarily.
    The median intensity cleanly filters out moving bubbles without leaving ghost shadows.
    """
    if not image_paths:
        raise ValueError("Image paths list is empty.")

    indices = parse_sample_range(
        len(image_paths), sample_range=sample_range, sample_size=sample_size
    )
    sampled_paths = [image_paths[idx] for idx in indices]

    if sample_range:
        print(
            f"Estimating static background from {len(sampled_paths)} frames using range '{sample_range}' "
            f"(frames {indices[0]} to {indices[-1]})..."
        )
    else:
        print(f"Estimating static background from {len(sampled_paths)} evenly-spaced frames...")

    frames = []
    for p in sampled_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            frames.append(img)

    if not frames:
        raise FileNotFoundError("Could not read any sample images.")

    if len(frames) == 1:
        return frames[0]

    # Enforce consistent shape across sample frames to avoid ValueError in np.stack
    ref_shape = frames[0].shape
    frames = [
        f if f.shape == ref_shape else cv2.resize(f, (ref_shape[1], ref_shape[0])) for f in frames
    ]

    # Stack along 3rd dimension and compute median per pixel
    stack = np.stack(frames, axis=2)
    bg_median = np.median(stack, axis=2).astype(np.uint8)
    return bg_median


def load_or_compute_background(
    input_images: list[Path],
    bg_source: Path | None = None,
    sample_size: int = 50,
    sample_range: str | None = None,
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
            bg_files = sorted([p for p in bg_source.iterdir() if p.suffix.lower() in valid_exts])
            if not bg_files:
                raise FileNotFoundError(
                    f"No valid background images found in directory: {bg_source}"
                )
            print(
                f"Computing background reference from dedicated background folder: {bg_source} ({len(bg_files)} images)"
            )
            return compute_temporal_median_background(
                bg_files, sample_size=sample_size, sample_range=sample_range
            )
        else:
            raise FileNotFoundError(f"Specified background path does not exist: {bg_source}")

    # Fallback to temporal median across video sequence
    print(
        f"No external background provided. Computing temporal median from sequence ({len(input_images)} frames)..."
    )
    return compute_temporal_median_background(
        input_images, sample_size=sample_size, sample_range=sample_range
    )


def process_background_subtraction(
    raw_img: np.ndarray,
    bg_img: np.ndarray,
    diff_threshold: int = 15,
    margin_radius_px: int = 10,
    fill_background_value: int = 255,
    max_area_ratio: float = 0.65,
    blur_kernel_size: int = 0,
    normalize_lighting: bool = False,
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
      max_area_ratio: Maximum allowable contour area as fraction of image size (default: 0.65).
                      Prevents whole-frame false background shifts from being treated as giant bubbles.
      blur_kernel_size: Optional Gaussian blur kernel size (odd integer, e.g. 3, 5) applied to difference
                        image to tolerate minor focus/vibration discrepancies. (default: 0 = disabled).
      normalize_lighting: If True, scales raw frame illumination to match background before differencing.
    """
    # Resize background if dimensions differ slightly
    if raw_img.shape != bg_img.shape:
        bg_img = cv2.resize(bg_img, (raw_img.shape[1], raw_img.shape[0]))

    raw_proc = raw_img
    if normalize_lighting:
        raw_mean = float(np.mean(raw_img))
        bg_mean = float(np.mean(bg_img))
        if raw_mean > 1e-3:
            gain = bg_mean / raw_mean
            raw_proc = np.clip(raw_img.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    # 1. Compute absolute intensity difference
    abs_diff = cv2.absdiff(raw_proc, bg_img)

    # Optional pre-blur to smooth out minor focus/grain differences
    if blur_kernel_size > 1:
        k = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        abs_diff = cv2.GaussianBlur(abs_diff, (k, k), 0)

    # 2. Threshold differences to isolate bubble boundary rings
    _, thresh = cv2.threshold(abs_diff, diff_threshold, 255, cv2.THRESH_BINARY)

    # 3. Morphological close to bridge tiny gaps in the contour
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

    # 4. Fill bubble interior holes while discarding spurious full-image background contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = raw_img.shape[0] * raw_img.shape[1]
    max_allowed_area = max_area_ratio * img_area

    valid_contours = [c for c in contours if cv2.contourArea(c) <= max_allowed_area]

    filled_bubble_mask = np.zeros_like(closed)
    if valid_contours:
        cv2.drawContours(filled_bubble_mask, valid_contours, -1, 255, thickness=cv2.FILLED)

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
    bg_source: Path | None = None,
    sample_size: int = 50,
    sample_range: str | None = None,
    diff_threshold: int = 15,
    margin_px: int = 10,
    bg_value: int = 255,
    max_area_ratio: float = 0.65,
    blur_kernel_size: int = 0,
    normalize_lighting: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}
    image_files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in valid_exts])

    if not image_files:
        raise FileNotFoundError(f"No image files found in {input_dir}")

    # 1. Resolve or compute background reference
    bg_ref = load_or_compute_background(
        input_images=image_files,
        bg_source=bg_source,
        sample_size=sample_size,
        sample_range=sample_range,
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
            max_area_ratio=max_area_ratio,
            blur_kernel_size=blur_kernel_size,
            normalize_lighting=normalize_lighting,
        )

        out_path = output_dir / p.name
        cv2.imwrite(str(out_path), cleaned)

        if i % 100 == 0 or i == len(image_files):
            print(f"  Processed [{i}/{len(image_files)}]")

    print(f"\n[DONE] Background removal complete. Cleaned frames saved to: {output_dir}")


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
        "--sample-range",
        "--sample-direction",
        type=str,
        default=None,
        help="Consecutive frame range for background estimation as 'start,count' (e.g. '0,80' for first 80 frames) "
        "or 'start:end' (e.g. '0:80'). Overrides evenly spaced sampling.",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=int,
        default=15,
        help="Intensity difference threshold (0-255) for bubble edge detection (default: 15; try 35-50 for noisy/shifted frames)",
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
    parser.add_argument(
        "--blur",
        type=int,
        default=0,
        help="Gaussian pre-blur kernel size (e.g. 3, 5) to tolerate slight focus/grain differences (default: 0 = disabled)",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=0.65,
        help="Max allowable bubble contour area as fraction of image size to prevent whole-frame mismatch blowout (default: 0.65)",
    )
    parser.add_argument(
        "--normalize-lighting",
        action="store_true",
        help="Automatically scale global frame brightness to match background before differencing",
    )
    args = parser.parse_args()

    run_batch_background_removal(
        input_dir=args.input,
        output_dir=args.output,
        bg_source=args.bg_source,
        sample_size=args.sample_size,
        sample_range=args.sample_range,
        diff_threshold=args.threshold,
        margin_px=args.margin,
        bg_value=args.bg_value,
        max_area_ratio=args.max_area_ratio,
        blur_kernel_size=args.blur,
        normalize_lighting=args.normalize_lighting,
    )
