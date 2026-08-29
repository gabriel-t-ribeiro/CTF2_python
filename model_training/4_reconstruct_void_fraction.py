import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt


def calculate_rouhani_axelsson_void_fraction(
    x: float,
    mass_flux_G: float,
    rho_l: float = 958.4,
    rho_v: float = 0.597,
    sigma: float = 0.0589,
    g: float = 9.81,
) -> float:
    """
    Computes theoretical volumetric void fraction (epsilon) using the
    Rouhani & Axelsson (1970) drift-flux correlation for flow boiling.

    Parameters:
        x: Vapor quality (0.0 to 1.0)
        mass_flux_G: Total mass flux G in kg/(m^2 s)
        rho_l: Liquid density in kg/m^3 (default: water at 100°C)
        rho_v: Vapor density in kg/m^3 (default: steam at 100°C)
        sigma: Surface tension in N/m (default: water-steam at 100°C)
        g: Gravitational acceleration in m/s^2
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Term 1: Distribution parameter effect C_0 * (x / rho_v + (1-x) / rho_l)
    c0 = 1.0 + 0.12 * (1.0 - x)
    kinematic_term = c0 * ((x / rho_v) + ((1.0 - x) / rho_l))

    # Term 2: Drift velocity component V_gj / G
    buoyancy = (g * sigma * (rho_l - rho_v)) ** 0.25
    drift_term = (1.18 * (1.0 - x) * buoyancy) / (mass_flux_G * np.sqrt(rho_l))

    epsilon = (x / rho_v) / (kinematic_term + drift_term)
    return float(np.clip(epsilon, 0.0, 1.0))


def compute_3d_gas_volume(
    mask_2d: np.ndarray,
    pipe_radius_px: float,
    pipe_center_y_px: float,
    smoothing_lambda: float = 0.05,
) -> float:
    """
    Reconstructs 3D gas volumes from 2D projected binary masks using a dual-regime approach:
      1. Small Bubbles (< 5000 px): 3D ellipsoid model bounded by local circular tube chord depth.
      2. Large Slugs (>= 5000 px): Euclidean Distance Transform with Laplacian smoothing,
         clipped against the exact circular cylinder depth profile z_chord(y) = sqrt(R^2 - delta_y^2).

    Parameters:
        mask_2d: Binary mask array (H, W) where 255/1 is gas and 0 is liquid.
        pipe_radius_px: Pipe internal radius in pixels.
        pipe_center_y_px: Vertical y-coordinate of the pipe centerline in the image.
        smoothing_lambda: Boundary smoothing factor for large slug extrusion.
    """
    h_img, w_img = mask_2d.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_2d, connectivity=8
    )
    total_volume_px3 = 0.0

    # Pre-calculate 1D vertical chord depth profile for the cylindrical tube:
    # z_chord(y) = sqrt(max(0, R^2 - (y - y_center)^2))
    y_indices = np.arange(h_img, dtype=np.float64)
    delta_y = np.abs(y_indices - pipe_center_y_px)
    r2_minus_y2 = np.maximum(0.0, (pipe_radius_px**2) - (delta_y**2))
    z_chord_profile = np.sqrt(r2_minus_y2)  # Shape (H,)
    # 2D broadcasted tube depth map: Shape (H, W)
    z_chord_map_2d = np.repeat(z_chord_profile[:, np.newaxis], w_img, axis=1)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        if area < 3:
            continue

        blob_mask = (labels == i).astype(np.uint8)
        c_y = centroids[i][1]

        # Local cylindrical depth available at the bubble's transverse position
        delta_y_c = abs(c_y - pipe_center_y_px)
        z_local_max = np.sqrt(max(0.0, (pipe_radius_px**2) - (delta_y_c**2)))

        # Regime 1: Small Bubbles (< 5000 px) -> 3D Ellipsoid
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

            # Constrain out-of-plane depth semi-axis c by local cylindrical chord depth
            c = min(b, z_local_max)
            vol_i = (4.0 / 3.0) * np.pi * a * b * c
            total_volume_px3 += vol_i

        # Regime 2: Large Slugs (>= 5000 px) -> EDT + Cylindrical Curvature Clipping
        else:
            dist_map = distance_transform_edt(blob_mask)
            # Apply exponential boundary Laplacian smoothing
            depth_map = dist_map * np.exp(-1.0 / (smoothing_lambda * dist_map + 1e-5))
            # Clip depth against the exact circular cross-section chord depth map
            depth_map_curved = np.minimum(depth_map, z_chord_map_2d)
            # Multiply by 2 for front + back symmetric halves
            vol_i = 2.0 * np.sum(depth_map_curved[blob_mask == 1])
            total_volume_px3 += vol_i

    return total_volume_px3


def run_reconstruction(
    mask_dir: Path,
    output_csv: Path,
    pixel_size_mm: float,
    pipe_dia_mm: float,
    vapor_quality: Optional[float] = None,
    mass_flux: Optional[float] = None,
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
        if mask is None:
            continue
        h, w = mask.shape
        pipe_center_y_px = h / 2.0  # Centerline of the cropped pipe

        # Reference cylinder volume (pi * R^2 * L)
        pipe_length_px = float(w)
        v_pipe_px3 = np.pi * (pipe_radius_px**2) * pipe_length_px

        # Reconstruct gas volume accounting for pipe curvature
        v_gas_px3 = compute_3d_gas_volume(mask, pipe_radius_px, pipe_center_y_px)

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

    mean_alpha = df["void_fraction_alpha"].mean()

    print("\n" + "=" * 50)
    print("      VOID FRACTION RECONSTRUCTION SUMMARY      ")
    print("=" * 50)
    print(f"Frames Analyzed     : {len(df)}")
    print(f"Mean Alpha (α_exp)  : {mean_alpha:.4f}")
    print(f"Min Alpha           : {df['void_fraction_alpha'].min():.4f}")
    print(f"Max Alpha           : {df['void_fraction_alpha'].max():.4f}")
    print(f"Standard Dev        : {df['void_fraction_alpha'].std():.4f}")

    # Theoretical Rouhani-Axelsson benchmark comparison if parameters are provided
    if vapor_quality is not None and mass_flux is not None:
        epsilon_th = calculate_rouhani_axelsson_void_fraction(vapor_quality, mass_flux)
        rel_diff = abs(mean_alpha - epsilon_th) / max(epsilon_th, 1e-5) * 100.0
        print("-" * 50)
        print("     ROUHANI-AXELSSON THEORETICAL BENCHMARK     ")
        print("-" * 50)
        print(f"Vapor Quality (x)   : {vapor_quality:.4f}")
        print(f"Mass Flux (G)       : {mass_flux:.1f} kg/(m² s)")
        print(f"Theoretical Void (ε): {epsilon_th:.4f}")
        print(f"Relative Difference : {rel_diff:.2f} %")

    print("=" * 50)
    print(f"Results saved to    : {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D Bubble Reconstruction and Void Fraction Analysis"
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
    parser.add_argument(
        "--vapor-quality",
        "-x",
        type=float,
        default=None,
        help="Optional vapor quality (x) for Rouhani-Axelsson benchmark",
    )
    parser.add_argument(
        "--mass-flux",
        "-G",
        type=float,
        default=None,
        help="Optional mass flux G in kg/(m^2 s) for Rouhani-Axelsson benchmark",
    )
    args = parser.parse_args()

    run_reconstruction(
        args.masks,
        args.output_csv,
        args.pixel_size,
        args.diameter,
        args.vapor_quality,
        args.mass_flux,
    )
