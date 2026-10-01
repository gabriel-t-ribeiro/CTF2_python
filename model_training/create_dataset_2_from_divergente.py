#!/usr/bin/env python3
"""
Create dataset_2 using original background images from images/background1/divergente_40kHz/
paired with pixel-aligned ground truth masks from dataset_2_no_background/masks/.
"""

import time
from pathlib import Path

import cv2
import numpy as np

FRAME_MAPPING = {
    "frame_0029.png": "Video_lente_divergente_40kHz_v2_S0001000001.png",
    "frame_0030.png": "Video_lente_divergente_40kHz_v2_S0001000015.png",
    "frame_0031.png": "Video_lente_divergente_40kHz_v2_S0001000109.png",
    "frame_0032.png": "Video_lente_divergente_40kHz_v2_S0001000241.png",
    "frame_0033.png": "Video_lente_divergente_40kHz_v2_S0001000379.png",
    "frame_0034.png": "Video_lente_divergente_40kHz_v2_S0001000529.png",
    "frame_0035.png": "Video_lente_divergente_40kHz_v2_S0001000721.png",
    "frame_0036.png": "Video_lente_divergente_40kHz_v2_S0001000967.png",
    "frame_0037.png": "Video_lente_divergente_40kHz_v2_S0001001153.png",
    "frame_0038.png": "Video_lente_divergente_40kHz_v2_S0001001393.png",
    "frame_0039.png": "Video_lente_divergente_40kHz_v2_S0001001561.png",
    "frame_0040.png": "Video_lente_divergente_40kHz_v2_S0001001657.png",
    "frame_0041.png": "Video_lente_divergente_40kHz_v2_S0001001669.png",
    "frame_0042.png": "Video_lente_divergente_40kHz_v2_S0001001693.png",
    "frame_0043.png": "Video_lente_divergente_40kHz_v2_S0001001719.png",
    "frame_0044.png": "Video_lente_divergente_40kHz_v2_S0001001879.png",
    "frame_0045.png": "Video_lente_divergente_40kHz_v2_S0001001953.png",
    "frame_0046.png": "Video_lente_divergente_40kHz_v2_S0001002029.png",
    "frame_0047.png": "Video_lente_divergente_40kHz_v2_S0001002151.png",
    "frame_0048.png": "Video_lente_divergente_40kHz_v2_S0001002173.png",
}

# Verified spatial translation: x_bg = x_nobg + 15, y_bg = y_nobg - 6
# Therefore: mask_bg[y_bg, x_bg] = mask_nobg[y_bg + 6, x_bg - 15]
DX = 15
DY = -6


def create_dataset_2(
    src_images_dir: Path,
    src_masks_dir: Path,
    dest_dataset_dir: Path,
) -> dict:
    t0 = time.perf_counter()
    dest_img_dir = dest_dataset_dir / "images"
    dest_mask_dir = dest_dataset_dir / "masks"
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_mask_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for frame_name, video_file in FRAME_MAPPING.items():
        src_img_path = src_images_dir / video_file
        src_mask_path = src_masks_dir / frame_name

        if not src_img_path.exists():
            raise FileNotFoundError(f"Missing source image: {src_img_path}")
        if not src_mask_path.exists():
            raise FileNotFoundError(f"Missing source mask: {src_mask_path}")

        # 1. Read authentic image with background
        img_bg = cv2.imread(str(src_img_path))
        h_bg, w_bg = img_bg.shape[:2]

        # 2. Read mask from dataset_2_no_background
        mask_nobg = cv2.imread(str(src_mask_path), cv2.IMREAD_GRAYSCALE)
        h_nobg, w_nobg = mask_nobg.shape[:2]

        # 3. Align mask to match img_bg pixel-for-pixel
        mask_aligned = np.zeros((h_bg, w_bg), dtype=np.uint8)
        for y_bg in range(h_bg):
            y_nobg = y_bg - DY  # y_bg + 6
            if 0 <= y_nobg < h_nobg:
                for x_bg in range(w_bg):
                    x_nobg = x_bg - DX  # x_bg - 15
                    if 0 <= x_nobg < w_nobg:
                        mask_aligned[y_bg, x_bg] = mask_nobg[y_nobg, x_nobg]

        # 4. Save to dataset_2
        cv2.imwrite(str(dest_img_dir / frame_name), img_bg)
        cv2.imwrite(str(dest_mask_dir / frame_name), mask_aligned)
        count += 1

    elapsed = time.perf_counter() - t0
    stats = {
        "count": count,
        "elapsed_s": elapsed,
        "ms_per_frame": (elapsed / count) * 1000 if count else 0,
        "image_shape": (h_bg, w_bg),
        "mask_shape": (h_bg, w_bg),
    }
    print(
        f"[DONE] Created dataset_2 with {count} frames in {elapsed:.3f} s ({stats['ms_per_frame']:.2f} ms/frame)"
    )
    print(f"       Images saved to: {dest_img_dir} (shape {h_bg}x{w_bg})")
    print(f"       Masks saved to:  {dest_mask_dir} (shape {h_bg}x{w_bg})")
    return stats


def generate_diagnostic_overlay(
    dataset_dir: Path,
    output_path: Path,
    sample_frame: str = "frame_0029.png",
):
    img_path = dataset_dir / "images" / sample_frame
    mask_path = dataset_dir / "masks" / sample_frame

    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    # Find contours from mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = img.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)  # Bright green contour

    # Mask in 3 channels for visualization
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Stack side by side: Raw Frame (with BG), Mask, Aligned Contour Overlay
    h, w = img.shape[:2]

    # Add title headers
    def add_label(image, text):
        out = image.copy()
        cv2.rectangle(out, (0, 0), (w, 35), (20, 20, 20), -1)
        cv2.putText(out, text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        return out

    p1 = add_label(img, "Raw Frame with Background (divergente_40kHz)")
    p2 = add_label(mask_bgr, "Aligned Ground Truth Mask")
    p3 = add_label(overlay, "Contour Verification (Green = Mask Outline)")

    canvas = np.vstack([p1, p2, p3])
    cv2.imwrite(str(output_path), canvas)
    print(f"[DIAGNOSTIC] Saved verification panel to {output_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    src_images = base_dir / "images" / "background1" / "divergente_40kHz"
    src_masks = base_dir / "dataset_2_no_background" / "masks"
    dest_dataset = base_dir / "dataset_2"

    stats = create_dataset_2(src_images, src_masks, dest_dataset)
    diag_path = base_dir / "models" / "dataset_2_alignment_verification.png"
    generate_diagnostic_overlay(dest_dataset, diag_path)
