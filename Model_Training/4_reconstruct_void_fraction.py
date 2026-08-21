from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

# Physical & Channel Parameters (Set according to your rig)
PIXEL_SIZE_MM = 0.067  # mm per pixel
PIPE_DIAMETER_MM = 2.96  # Pipe internal diameter[cite: 8]
PIPE_RADIUS_PX = (PIPE_DIAMETER_MM / 2.0) / PIXEL_SIZE_MM
PIPE_LENGTH_MM = 10.0  # Visualized axial test segment length
PIPE_LENGTH_PX = PIPE_LENGTH_MM / PIXEL_SIZE_MM
SMOOTHING_LAMBDA = 0.05  # Laplacian decay factor[cite: 8]


def rouhani_axelsson_void_fraction(x, G, rho_l, rho_v, sigma, D_h, g=9.81):
    """Rouhani-Axelsson (1970) drift-flux correlation for void fraction[cite: 8]."""
    term1 = (1.0 + 0.12 * (1.0 - x)) * ((x / rho_v) + ((1.0 - x) / rho_l))  # [cite: 8]
    term2 = (1.18 * (1.0 - x) * (g * sigma * (rho_l - rho_v)) ** 0.25) / (
        G * (rho_l**0.5)
    )  # [cite: 8]
    epsilon = (x / rho_v) / (term1 + term2)  # [cite: 8]
    return epsilon


def compute_3d_gas_volume(mask_2d: np.ndarray) -> float:
    """Calculates volumetric gas content using the dual-regime reconstruction[cite: 8]."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_2d, connectivity=8
    )  # [cite: 8]
    total_volume_px3 = 0.0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 3:
            continue

        blob_mask = (labels == i).astype(np.uint8)

        # Regime 1: Small Bubbles (Area < 5000 px) -> 3D Gaussian / Ellipsoid Model[cite: 8]
        if area < 5000:
            contours, _ = cv2.findContours(
                blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if len(contours) > 0 and len(contours[0]) >= 5:
                ellipse = cv2.fitEllipse(contours[0])
                (center), (d_minor, d_major), angle = ellipse
                a = (d_major / 2.0) if d_major > 0 else 1.0
                b = (d_minor / 2.0) if d_minor > 0 else 1.0
            else:
                r = np.sqrt(area / np.pi)
                a, b = r, r

            # Depth constraint bounded by pipe radius[cite: 8]
            c = min(b, PIPE_RADIUS_PX)
            vol_i = (4.0 / 3.0) * np.pi * a * b * c
            total_volume_px3 += vol_i

        # Regime 2: Large Bubbles / Slugs (Area >= 5000 px) -> Distance Transform + Laplacian[cite: 8]
        else:
            dist_map = distance_transform_edt(blob_mask)  # [cite: 8]
            # Exponential Laplacian depth extrusion[cite: 8]
            depth_map = dist_map * np.exp(
                -1.0 / (SMOOTHING_LAMBDA * dist_map + 1e-6)
            )  # [cite: 8]
            depth_map = np.clip(depth_map, 0, PIPE_RADIUS_PX)
            vol_i = 2.0 * np.sum(depth_map)
            total_volume_px3 += vol_i

    return total_volume_px3


def analyze_dataset():
    BASE_DIR = Path(__file__).resolve().parent
    MASK_DIR = BASE_DIR / "segmented_masks"
    mask_files = sorted(list(MASK_DIR.glob("*.png")))

    # Total pipe segment volume in px^3
    v_total_pipe_px3 = np.pi * (PIPE_RADIUS_PX**2) * PIPE_LENGTH_PX  # [cite: 8]

    records = []
    print(f"Computing 3D Void Fraction for {len(mask_files)} masks...")

    for f in mask_files:
        mask = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        v_gas_px3 = compute_3d_gas_volume(mask)
        alpha = v_gas_px3 / v_total_pipe_px3  # [cite: 8]

        records.append(
            {
                "frame": f.stem,
                "gas_volume_px3": v_gas_px3,
                "void_fraction_alpha": alpha,
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(BASE_DIR / "void_fraction_results.csv", index=False)

    mean_alpha = df["void_fraction_alpha"].mean()
    print(f"\nMean Experimental Void Fraction (alpha): {mean_alpha:.4f}")

    # Theoretical Validation Example: R245fa properties[cite: 8]
    # G=1000 kg/m^2s, x=0.32, rho_l=1200 kg/m^3, rho_v=25 kg/m^3, sigma=0.008 N/m[cite: 8]
    alpha_theoretical = rouhani_axelsson_void_fraction(
        x=0.32,
        G=1000.0,
        rho_l=1200.0,
        rho_v=25.0,
        sigma=0.008,
        D_h=PIPE_DIAMETER_MM * 1e-3,
    )
    print(f"Rouhani-Axelsson Theoretical Void Fraction: {alpha_theoretical:.4f}")


if __name__ == "__main__":
    analyze_dataset()
