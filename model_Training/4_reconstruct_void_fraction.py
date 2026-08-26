import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt


def compute_3d_gas_volume(
    mask_2d: np.ndarray, pipe_radius_px: float, smoothing_lambda: float = 0.05
) -> float:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_2d, connectivity=8
    )
    total_volume_px3 = 0.0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        if area < 3:
            continue

        blob_mask = (labels == i).astype(np.uint8)

        # Small Bubbles (< 5000 px): Robust Ellipsoid Fitting
        if area < 5000:
            contours, _ = cv2.findContours(
                blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if len(contours) > 0 and len(contours[0]) >= 5:
                try:
                    (_, (d_minor, d_major), _) = cv2.fitEllipse(contours[0])
                    a = min(d_major / 2.0, max(w, h))  # Bound semi-major axis
                    b = min(d_minor / 2.0, min(w, h))  # Bound semi-minor axis
                except Exception:
                    r = np.sqrt(area / np.pi)
                    a, b = r, r
            else:
                r = np.sqrt(area / np.pi)
                a, b = r, r

            # Gas depth cannot exceed pipe radius
            c = min(b, pipe_radius_px)
            vol_i = (4.0 / 3.0) * np.pi * a * b * c
            total_volume_px3 += vol_i

        # Large Bubbles / Slugs (>= 5000 px): Distance Transform + Laplacian Depth
        else:
            dist_map = distance_transform_edt(blob_mask)
            # Smooth extrusion profile bounded by pipe radius
            depth_map = dist_map * np.exp(-1.0 / (smoothing_lambda * dist_map + 1e-5))
            depth_map = np.clip(depth_map, 0, pipe_radius_px)
            vol_i = 2.0 * np.sum(depth_map)
            total_volume_px3 += vol_i

    return total_volume_px3


def run_reconstruction(
    mask_dir: Path, output_csv: Path, pixel_size_mm: float, pipe_dia_mm: float
):
    mask_files = sorted(list(mask_dir.glob("*.png")))
    if not mask_files:
        raise FileNotFoundError(f"No masks found in {mask_dir}")

    pipe_radius_px = (pipe_dia_mm / 2.0) / pixel_size_mm
    records = []

    print(
        f"Processing {len(mask_files)} masks with D = {pipe_dia_mm} mm ({pipe_radius_px:.1f} px radius)..."
    )

    for f in mask_files:
        mask = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        h, w = mask.shape

        # Pipe volume is dynamically computed from actual image width (flow direction)
        pipe_length_px = float(w)
        v_pipe_px3 = np.pi * (pipe_radius_px**2) * pipe_length_px

        v_gas_px3 = compute_3d_gas_volume(mask, pipe_radius_px)

        # Alpha is physically bounded between 0.0 and 1.0
        alpha = float(np.clip(v_gas_px3 / (v_pipe_px3 + 1e-8), 0.0, 1.0))

        records.append(
            {
                "frame": f.stem,
                "gas_volume_px3": v_gas_px3,
                "pipe_volume_px3": v_pipe_px3,
                "void_fraction_alpha": alpha,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    print("\n" + "=" * 45)
    print("      VOID FRACTION RECONSTRUCTION SUMMARY      ")
    print("=" * 45)
    print(f"Frames Analyzed  : {len(df)}")
    print(f"Mean Alpha (α)   : {df['void_fraction_alpha'].mean():.4f}")
    print(f"Min Alpha        : {df['void_fraction_alpha'].min():.4f}")
    print(f"Max Alpha        : {df['void_fraction_alpha'].max():.4f}")
    print(f"Standard Dev     : {df['void_fraction_alpha'].std():.4f}")
    print("=" * 45)
    print(f"Results saved to : {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D Bubble Reconstruction and Void Fraction"
    )
    parser.add_argument(
        "--masks",
        "-m",
        type=Path,
        required=True,
        help="Input folder with binary masks",
    )
    parser.add_argument(
        "--output-csv",
        "-o",
        type=Path,
        default=Path("results/void_fraction_results.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        required=True,
        help="Pixel size in mm/pixel",
    )
    parser.add_argument(
        "--diameter",
        type=float,
        required=True,
        help="Channel inner diameter in mm",
    )
    args = parser.parse_args()

    run_reconstruction(args.masks, args.output_csv, args.pixel_size, args.diameter)
