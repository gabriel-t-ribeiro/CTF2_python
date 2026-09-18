"""
Depth from Defocus (DFD) Analysis Module in Standard SI Units.

Implements the single-camera Depth from Defocus technique according to:
  Rao, Sharma, Basu, & Tropea (2024). "Depth from defocus technique:
  a simple calibration-free approach for sizing and depth measurement of spherical particles."
  Experiments in Fluids, 65(9), 136.

Key Equations & Theory:
  1. Blur parameter: sigma = beta * |Delta z|
  2. Dimensionless parameters:
     - Reference gradient at g_t = 0.5: G_tilde = |r_t * dg_t / dr_t| = f2(sigma_tilde)
     - Dimensionless radius: rho_tilde = r_t / d_0 = f1(sigma_tilde)
     - In-focus diameter: d_0 = r_t / rho_tilde
     - Blur size: sigma = sigma_tilde * d_0
     - Physical depth: |Delta z| = sigma / beta_eff
  3. In-situ Calibration with tube walls:
     beta_eff = sigma_rear / D_int (meters/pixel or pixels/meter)
"""

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import i0e, i1e


def _compute_disc_convolved_profile(sigma_tilde: float, rho_tilde: float) -> tuple[float, float]:
    """
    Computes intensity g_t and radial gradient dg_t / drho_tilde for a circular disc of unit radius
    convolved with a 2D Gaussian PSF of standard deviation sigma_tilde.
    Using numerically stable exponential scaling:
      exp(-(r^2 + rho^2)/(2*s^2)) * I0(r*rho/s^2) = exp(-(r-rho)^2/(2*s^2)) * I0e(r*rho/s^2)
    """
    if sigma_tilde < 1e-4:
        g = 1.0 if rho_tilde < 0.5 else (0.5 if abs(rho_tilde - 0.5) < 1e-4 else 0.0)
        dg = 0.0
        return g, dg

    s2 = sigma_tilde**2

    def integrand(r):
        arg = (r * rho_tilde) / s2
        exp_factor = np.exp(-((r - rho_tilde) ** 2) / (2.0 * s2))
        return (r / s2) * exp_factor * i0e(arg)

    def integrand_deriv(r):
        arg = (r * rho_tilde) / s2
        exp_factor = np.exp(-((r - rho_tilde) ** 2) / (2.0 * s2))
        d_term = (-rho_tilde / s2) * i0e(arg) + (r / s2) * i1e(arg)
        return (r / s2) * exp_factor * d_term

    val, _ = quad(integrand, 0.0, 0.5, limit=50)
    deriv, _ = quad(integrand_deriv, 0.0, 0.5, limit=50)
    return float(val), float(deriv)


class DFDModel:
    """
    Lookup-table based fast inverter for Rao et al. (2024) functions f1(sigma_tilde) and f2(sigma_tilde).
    Precomputes relationship between dimensionless gradient G_tilde, dimensionless radius rho_tilde,
    and dimensionless blur sigma_tilde for sigma_tilde in [0.005, 0.35].
    """

    def __init__(self):
        # Precompute table over operative range sigma_tilde in [0.005, 0.35]
        sigmas = np.linspace(0.005, 0.35, 45)
        rhos = []
        g_tildes = []

        for s in sigmas:
            low, high = 0.20, 0.52
            for _ in range(16):
                mid = 0.5 * (low + high)
                g_val, _ = _compute_disc_convolved_profile(s, mid)
                if g_val > 0.5:
                    low = mid
                else:
                    high = mid
            rho_50 = 0.5 * (low + high)
            _, dg_50 = _compute_disc_convolved_profile(s, rho_50)
            g_tilde = abs(rho_50 * dg_50)

            rhos.append(rho_50)
            g_tildes.append(g_tilde)

        self._sigmas = sigmas
        self._rhos = np.array(rhos)
        self._g_tildes = np.array(g_tildes)

        # Monotonic interpolators:
        sort_idx = np.argsort(self._g_tildes)
        self._interp_sigma = interp1d(
            self._g_tildes[sort_idx],
            self._sigmas[sort_idx],
            kind="linear",
            bounds_error=False,
            fill_value=(self._sigmas[sort_idx[-1]], self._sigmas[sort_idx[0]]),
        )
        self._interp_rho = interp1d(
            self._sigmas,
            self._rhos,
            kind="linear",
            bounds_error=False,
            fill_value=(self._rhos[0], self._rhos[-1]),
        )

    def invert(self, g_tilde: float) -> tuple[float, float]:
        """
        Given dimensionless gradient G_tilde, returns:
          (sigma_tilde, rho_tilde)
        """
        s_tilde = float(self._interp_sigma(g_tilde))
        rho_tilde = float(self._interp_rho(s_tilde))
        return s_tilde, rho_tilde


def analyze_frame_dfd(
    raw_img: np.ndarray,
    binary_mask: np.ndarray,
    dfd_model: DFDModel,
    pixel_size_m: float = 6.7e-5,
    beta_eff_px_m: float = 2000.0,
    background_img: np.ndarray | None = None,
    near_wall_sigma_threshold: float = 0.05,
) -> list[dict[str, Any]]:
    """
    Extracts 3D coordinates (x_m, y_m, z_m) and in-focus physical diameters (d0_m)
    for each detected bubble in the frame using single-camera DFD.

    Includes near-wall regularization for sigma_tilde < near_wall_sigma_threshold
    to eliminate near-focus blind zone instabilities on front-wall nucleating bubbles.
    """
    h, w = raw_img.shape

    # Normalize intensity profile
    if background_img is not None:
        bg_f = background_img.astype(np.float32)
        raw_f = raw_img.astype(np.float32)
        diff = np.maximum(0.0, bg_f - raw_f)
        max_diff = np.max(diff) if np.max(diff) > 0 else 255.0
        norm_img = diff / max_diff
    else:
        norm_img = (255.0 - raw_img.astype(np.float32)) / 255.0

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    results = []

    # Compute spatial image gradients
    sobel_x = cv2.Sobel(norm_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(norm_img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    for i in range(1, num_labels):
        area_px = stats[i, cv2.CC_STAT_AREA]
        if area_px < 15:  # Ignore microscopic specks
            continue

        c_x_px, c_y_px = float(centroids[i][0]), float(centroids[i][1])
        r_t_px = np.sqrt(area_px / np.pi)

        blob_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        edge_pixels = contours[0].squeeze()
        if len(edge_pixels.shape) != 2:
            continue

        ys = np.clip(edge_pixels[:, 1], 0, h - 1)
        xs = np.clip(edge_pixels[:, 0], 0, w - 1)
        edge_grads = grad_mag[ys, xs]

        # Robust trimmed-mean gradient estimation along bubble boundary
        # Rejects outlier gradient spikes caused by bubble glare or caustic reflections
        if len(edge_grads) >= 8:
            p_low, p_high = np.percentile(edge_grads, [15, 85])
            valid_grads = edge_grads[(edge_grads >= p_low) & (edge_grads <= p_high)]
            mean_edge_grad = (
                float(np.mean(valid_grads)) if len(valid_grads) > 0 else float(np.mean(edge_grads))
            )
        else:
            mean_edge_grad = float(np.mean(edge_grads))

        # Dimensionless gradient: G_tilde = r_t * |dg_t / dr_t|
        g_tilde = r_t_px * mean_edge_grad

        # Deconvolve blur size sigma_tilde and scale rho_tilde
        s_tilde, rho_tilde = dfd_model.invert(g_tilde)

        # Near-Focus Blind Zone Regularization (Rao et al., 2024; s_tilde < 0.05):
        # In the near-wall focus regime, boundary gradient G_tilde is steep and vulnerable
        # to pixel discretization and noise. For bubbles sliding along the heated front wall,
        # optical blur is sub-pixel (z ~ 0, rho_tilde -> 0.5, d0 -> 2*r_t).
        # We apply a C1-continuous smoothstep ramp to eliminate noise-induced depth oscillations:
        is_near_wall = s_tilde < near_wall_sigma_threshold
        if is_near_wall and near_wall_sigma_threshold > 0:
            u = max(0.0, min(1.0, s_tilde / near_wall_sigma_threshold))
            reg_factor = (3.0 * u**2) - (2.0 * u**3)  # Smoothstep S(u) in [0, 1]
            rho_reg = 0.5 * (1.0 - reg_factor) + rho_tilde * reg_factor
            d0_px = (2.0 * r_t_px) / (2.0 * rho_reg)
            sigma_px = s_tilde * d0_px * reg_factor
        else:
            d0_px = (2.0 * r_t_px) / (2.0 * rho_tilde)
            sigma_px = s_tilde * d0_px

        # Physical units (SI)
        d0_m = d0_px * pixel_size_m
        x_m = c_x_px * pixel_size_m
        y_m = c_y_px * pixel_size_m

        # Out-of-plane depth z_m = sigma / beta_eff
        z_m = sigma_px / beta_eff_px_m

        results.append(
            {
                "bubble_id": i,
                "centroid_x_m": x_m,
                "centroid_y_m": y_m,
                "depth_z_m": z_m,
                "diameter_d0_m": d0_m,
                "blur_sigma_px": sigma_px,
                "sigma_tilde": s_tilde,
                "g_tilde": g_tilde,
                "near_wall_zone": bool(is_near_wall),
                "area_px": area_px,
                "centroid_x_px": c_x_px,
                "centroid_y_px": c_y_px,
            }
        )

    return results


def run_dfd_pipeline(
    raw_images_dir: Path,
    masks_dir: Path,
    output_csv: Path,
    pixel_size: float = 6.7e-5,
    beta_eff: float = 2000.0,
    background_path: Path | None = None,
    near_wall_threshold: float = 0.05,
):
    """
    Runs single-camera Depth from Defocus analysis across all paired images and masks.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    pixel_size_m = pixel_size * 1e-3 if pixel_size > 0.001 else pixel_size
    dfd_model = DFDModel()

    bg_img = None
    if background_path is not None and background_path.exists():
        bg_img = cv2.imread(str(background_path), cv2.IMREAD_GRAYSCALE)

    mask_files = sorted(masks_dir.glob("*.png"))
    if not mask_files:
        raise FileNotFoundError(f"No mask files found in: {masks_dir}")

    all_records = []
    print(f"Executing Depth from Defocus (DFD) analysis across {len(mask_files)} frames:")
    print(f"  Pixel size (SI)     : {pixel_size_m:.6e} m/px")
    print(f"  Beta calibration    : {beta_eff:.2f} px/m")
    print(f"  Near-wall threshold : {near_wall_threshold:.4f} (sigma_tilde)")

    for f_mask in mask_files:
        raw_candidates = [
            raw_images_dir / f_mask.name,
            raw_images_dir / f"{f_mask.stem.replace('_mask', '')}.png",
            raw_images_dir / f"{f_mask.stem.replace('_mask', '')}.jpg",
            raw_images_dir / f"{f_mask.stem.replace('_mask', '')}.tif",
        ]
        raw_path = next((p for p in raw_candidates if p.exists()), None)
        if raw_path is None:
            continue

        raw_img = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(f_mask), cv2.IMREAD_GRAYSCALE)
        if raw_img is None or mask is None:
            continue

        frame_results = analyze_frame_dfd(
            raw_img,
            mask,
            dfd_model,
            pixel_size_m=pixel_size_m,
            beta_eff_px_m=beta_eff,
            background_img=bg_img,
            near_wall_sigma_threshold=near_wall_threshold,
        )

        for r in frame_results:
            r["frame"] = f_mask.stem
            all_records.append(r)

    df = pd.DataFrame(all_records)
    df.to_csv(output_csv, index=False)

    print("\n" + "=" * 52)
    print("      DEPTH FROM DEFOCUS (DFD) SUMMARY (SI)      ")
    print("=" * 52)
    print(f"Total Bubbles Sized : {len(df)}")
    if not df.empty:
        print(
            f"Mean Diameter (d0)  : {df['diameter_d0_m'].mean():.6e} m ({df['diameter_d0_m'].mean() * 1e3:.3f} mm)"
        )
        print(
            f"Mean Depth (|z|)    : {df['depth_z_m'].mean():.6e} m ({df['depth_z_m'].mean() * 1e3:.3f} mm)"
        )
        print(
            f"Max Depth (|z|)     : {df['depth_z_m'].max():.6e} m ({df['depth_z_m'].max() * 1e3:.3f} mm)"
        )
        print(f"Mean Blur (sigma)   : {df['blur_sigma_px'].mean():.2f} px")
        if "near_wall_zone" in df:
            n_wall = int(df["near_wall_zone"].sum())
            print(f"Near-Wall Bubbles   : {n_wall} ({n_wall / len(df) * 100:.1f}%)")
    print("=" * 52)
    print(f"[DONE] Saved DFD results to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-Camera Depth from Defocus (DFD) Sizing & Depth in SI Units"
    )
    parser.add_argument("--images", "-i", type=Path, required=True, help="Path to raw image frames")
    parser.add_argument("--masks", "-m", type=Path, required=True, help="Path to binary masks")
    parser.add_argument(
        "--output-csv",
        "-o",
        type=Path,
        default=Path("results/dfd_results.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=8.6e-5,
        help="Pixel size in m/px (SI, default: 8.6e-5 m/px = 0.086 mm/px)",
    )
    parser.add_argument(
        "--beta-eff",
        type=float,
        default=2000.0,
        help="Effective DFD calibration constant in px/m",
    )
    parser.add_argument(
        "--near-wall-threshold",
        type=float,
        default=0.05,
        help="Dimensionless blur threshold (sigma_tilde) for near-wall/near-focus regularization (default: 0.05)",
    )
    parser.add_argument(
        "--background",
        "-bg",
        type=Path,
        default=None,
        help="Optional background image for normalization",
    )
    args = parser.parse_args()

    run_dfd_pipeline(
        args.images,
        args.masks,
        args.output_csv,
        pixel_size=args.pixel_size,
        beta_eff=args.beta_eff,
        background_path=args.background,
        near_wall_threshold=args.near_wall_threshold,
    )
