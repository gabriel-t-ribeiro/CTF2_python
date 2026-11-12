#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

try:
    from visualization.render_3d_flow import (
        reconstruct_3d_bubbles_si,
        render_3d_flow_visualization,
    )
except ModuleNotFoundError:  # executed directly: put the repository root on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from visualization.render_3d_flow import (
        reconstruct_3d_bubbles_si,
        render_3d_flow_visualization,
    )


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

    All quantities are strictly defined in standard SI units:
        x: Vapor quality (dimensionless, 0.0 to 1.0)
        mass_flux_G: Total mass flux G in kg/(m^2 s)
        rho_l: Liquid density in kg/m^3 (default: water at 100°C, 958.4 kg/m^3)
        rho_v: Vapor density in kg/m^3 (default: steam at 100°C, 0.597 kg/m^3)
        sigma: Surface tension in N/m (default: water-steam at 100°C, 0.0589 N/m)
        g: Gravitational acceleration in m/s^2 (default: 9.81 m/s^2)
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if mass_flux_G <= 0.0:
        raise ValueError(f"Mass flux G must be positive, got {mass_flux_G}")

    # Term 1: Distribution parameter effect C_0 * (x / rho_v + (1-x) / rho_l)
    c0 = 1.0 + 0.12 * (1.0 - x)
    kinematic_term = c0 * ((x / rho_v) + ((1.0 - x) / rho_l))

    # Term 2: Drift velocity component V_gj / G
    # Buoyancy term: [g * sigma * (rho_l - rho_v)]^0.25
    delta_rho = max(0.0, rho_l - rho_v)
    buoyancy = (g * sigma * delta_rho) ** 0.25
    drift_term = (1.18 * (1.0 - x) * buoyancy) / (mass_flux_G * np.sqrt(rho_l))

    epsilon = (x / rho_v) / (kinematic_term + drift_term)
    return float(np.clip(epsilon, 0.0, 1.0))


def compute_3d_gas_volume(
    mask_2d: np.ndarray,
    pipe_radius_px: float,
    pipe_center_y_px: float,
    smoothing_lambda: float = 0.15,
    pixel_size_m: float | None = None,
    regime_thresh_px: float | None = None,
) -> float:
    """
    Reconstructs 3D gas volumes from 2D projected binary masks using a dual-regime approach:
      1. Small Bubbles (d_eq < 2*R_pipe or area < regime_thresh_px): 3D ellipsoid model
         bounded by local circular tube chord depth.
      2. Large Slugs: Local circular cylinder chord depth profile z_chord(y) = sqrt(R^2 - delta_y^2)
         with exponential Laplacian boundary smoothing (Eduardo Reis, Eq 3.2):
         D'(p) = z_chord(y) * exp(-1 / (lambda * dist(p) + eps))

    If pixel_size_m is provided, calculation is performed and returned in SI cubic meters (m^3).
    Otherwise, returns volume in cubic pixels (px^3) for legacy compatibility.

    Parameters:
        mask_2d: Binary mask array (H, W) where >0 is gas and 0 is liquid.
        pipe_radius_px: Pipe internal radius in pixels.
        pipe_center_y_px: Vertical y-coordinate of the pipe centerline in the image.
        smoothing_lambda: Boundary smoothing factor for large slug extrusion.
        pixel_size_m: Optional pixel size in meters/pixel for direct SI output.
        regime_thresh_px: Optional custom pixel area threshold distinguishing regimes.
                          Defaults to physical threshold based on pipe radius.
    """
    h_img, w_img = mask_2d.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_2d, connectivity=8)
    total_volume_px3 = 0.0

    # Default regime threshold: bubble equivalent diameter reaching channel diameter (2 * R)
    # Area = pi * R^2 (in pixels)
    if regime_thresh_px is None:
        regime_thresh_px = np.pi * (pipe_radius_px**2)

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

        # Regime 1: Small Bubbles (area < regime_thresh_px) -> 3D Ellipsoid
        if area < regime_thresh_px:
            contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            fitted = False
            if len(contours) > 0 and len(contours[0]) >= 5:
                try:
                    box = cv2.fitEllipse(contours[0])
                    d1, d2 = box[1]
                    # Ensure major and minor axes are properly sorted
                    d_major = max(float(d1), float(d2))
                    d_minor = min(float(d1), float(d2))
                    # Guard against NaN/negative values from OpenCV fitEllipse
                    if not (np.isnan(d_major) or np.isnan(d_minor) or d_minor <= 0 or d_major <= 0):
                        a = min(d_major / 2.0, max(w, h))  # Semi-major axis
                        b = min(d_minor / 2.0, min(w, h))  # Semi-minor axis
                        fitted = True
                except (cv2.error, ValueError, TypeError):
                    fitted = False

            if not fitted:
                r = np.sqrt(area / np.pi)
                a, b = r, r
            else:
                # Area conservation guardrail (Week 37):
                # For irregular, concave, or odd-shaped non-convex bubbles, fitEllipse can
                # severely overestimate the 2D projected area (pi * a * b >> area).
                # We scale a and b to strictly conserve the actual measured 2D bubble area.
                ellipse_area = np.pi * a * b
                if ellipse_area > 0 and area > 0:
                    scale = min(1.0, np.sqrt(area / ellipse_area))
                    a *= scale
                    b *= scale

            # Constrain out-of-plane depth semi-axis c by local cylindrical chord depth
            c = min(b, z_local_max)
            vol_i = (4.0 / 3.0) * np.pi * a * b * c
            if not np.isnan(vol_i):
                total_volume_px3 += vol_i

        # Regime 2: Large Slugs (>= regime_thresh_px) -> Laplacian Extrusion of Tube Chord
        else:
            dist_map = distance_transform_edt(blob_mask)
            # Eduardo Reis (2025) Eq 3.2: D'(p) = D(p) * exp(-1 / (lambda * d(p)))
            # The local chord depth z_chord_map_2d is scaled by boundary attenuation
            depth_map = z_chord_map_2d * np.exp(-1.0 / (smoothing_lambda * dist_map + 1e-5))
            depth_map_curved = np.minimum(depth_map, z_chord_map_2d)
            # Multiply by 2 for front + back symmetric halves
            vol_i = 2.0 * np.sum(depth_map_curved[blob_mask == 1])
            if not np.isnan(vol_i):
                total_volume_px3 += vol_i

    if pixel_size_m is not None:
        # Convert px^3 to standard SI m^3
        return total_volume_px3 * (pixel_size_m**3)

    return total_volume_px3


def run_reconstruction(
    mask_dir: Path,
    output_csv: Path,
    pixel_size: float,
    diameter: float,
    vapor_quality: float | None = None,
    mass_flux: float | None = None,
    render_sample: bool = False,
    render_output: Path | None = None,
    pipe_center_y_px: float | None = None,
    regime_thresh_px: float | None = None,
    plot_timeline: bool = False,
    fps: float = 2000.0,
):
    """
    Executes frame-by-frame 3D void fraction reconstruction in standard SI units.
    """
    mask_files = sorted(mask_dir.glob("*.png"))
    if not mask_files:
        raise FileNotFoundError(f"No masks found in {mask_dir}")

    # Standardize inputs to SI units (meters)
    # If user passed pixel_size > 0.001, it was specified in mm -> convert to meters
    pixel_size_m = pixel_size * 1e-3 if pixel_size > 0.001 else pixel_size

    # If diameter > 0.05, it was likely passed in mm (e.g. 20.0 mm) -> convert to meters
    diameter_m = diameter * 1e-3 if diameter > 0.05 else diameter

    pipe_radius_m = diameter_m / 2.0
    pipe_radius_px = pipe_radius_m / pixel_size_m

    print("Executing 3D void fraction reconstruction in standard SI units:")
    print(f"  Masks count         : {len(mask_files)}")
    print(f"  Pixel size (SI)     : {pixel_size_m:.6e} m/px ({pixel_size_m * 1e3:.4f} mm/px)")
    print(f"  Channel diameter (SI): {diameter_m:.6e} m ({diameter_m * 1e3:.2f} mm)")
    print(f"  Channel radius (SI) : {pipe_radius_m:.6e} m ({pipe_radius_px:.1f} px)")

    records = []
    first_recon_data = None

    for idx, f in enumerate(mask_files):
        mask = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        h, w = mask.shape
        cy = pipe_center_y_px if pipe_center_y_px is not None else (h / 2.0)

        # Channel section length and cylindrical volume in SI (m, m^3)
        pipe_length_m = float(w) * pixel_size_m
        v_pipe_m3 = np.pi * (pipe_radius_m**2) * pipe_length_m

        # Pixel-space reference volume for cross-verification
        pipe_length_px = float(w)
        v_pipe_px3 = np.pi * (pipe_radius_px**2) * pipe_length_px

        # Reconstruct gas volume in SI (m^3)
        v_gas_m3 = compute_3d_gas_volume(
            mask,
            pipe_radius_px,
            cy,
            pixel_size_m=pixel_size_m,
            regime_thresh_px=regime_thresh_px,
        )
        v_gas_px3 = v_gas_m3 / (pixel_size_m**3)

        # Alpha is physically bounded between 0.0 and 1.0
        alpha = float(np.clip(v_gas_m3 / (v_pipe_m3 + 1e-12), 0.0, 1.0))

        records.append(
            {
                "frame": f.stem,
                "gas_volume_m3": v_gas_m3,
                "pipe_volume_m3": v_pipe_m3,
                "void_fraction_alpha": alpha,
                "gas_volume_px3": v_gas_px3,
                "pipe_volume_px3": v_pipe_px3,
            }
        )

        if idx == 0 and render_sample:
            first_recon_data = reconstruct_3d_bubbles_si(
                mask, pixel_size_m=pixel_size_m, pipe_dia_m=diameter_m
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    mean_alpha = float(df["void_fraction_alpha"].mean())

    print("\n" + "=" * 52)
    print("      VOID FRACTION RECONSTRUCTION SUMMARY (SI)      ")
    print("=" * 52)
    print(f"Frames Analyzed     : {len(df)}")
    print(f"Mean Alpha (alpha_exp): {mean_alpha:.4f} ({mean_alpha * 100:.3f}%)")
    print(f"Min Alpha           : {df['void_fraction_alpha'].min():.4f}")
    print(f"Max Alpha           : {df['void_fraction_alpha'].max():.4f}")
    print(f"Standard Dev        : {df['void_fraction_alpha'].std():.4f}")
    print(f"Mean Gas Volume     : {df['gas_volume_m3'].mean():.6e} m³")
    print(f"Mean Pipe Volume    : {df['pipe_volume_m3'].mean():.6e} m³")

    # Theoretical Rouhani-Axelsson benchmark comparison in SI
    if vapor_quality is not None and mass_flux is not None:
        epsilon_th = calculate_rouhani_axelsson_void_fraction(vapor_quality, mass_flux)
        rel_diff = abs(mean_alpha - epsilon_th) / max(epsilon_th, 1e-5) * 100.0
        print("-" * 52)
        print("     ROUHANI-AXELSSON THEORETICAL BENCHMARK (SI)     ")
        print("-" * 52)
        print(f"Vapor Quality (x)   : {vapor_quality:.4f} [-]")
        print(f"Mass Flux (G)       : {mass_flux:.1f} kg/(m² s)")
        print(f"Theoretical Void (ε): {epsilon_th:.4f} ({epsilon_th * 100:.2f}%)")
        print(f"Relative Difference : {rel_diff:.2f} %")

    print("=" * 52)
    print(f"Results saved to    : {output_csv}")

    # If 3D render is requested, render the sample frame
    if render_sample and first_recon_data is not None:
        render_path = (
            render_output
            if render_output is not None
            else output_csv.parent / "3d_reconstruction_render.png"
        )
        rendered = render_3d_flow_visualization(first_recon_data, render_path)
        print(f"Rendered 3D artifact: {rendered}")

    # If timeline plot is requested, generate publication plot and HTML dashboard
    if plot_timeline:
        from visualization.plot_void_fraction_timeline import run_timeline_pipeline

        timeline_plot_path = output_csv.parent / f"{output_csv.stem}_timeline.png"
        timeline_html_path = output_csv.parent / f"{output_csv.stem}_timeline.html"
        run_timeline_pipeline(
            input_paths=[output_csv],
            output_plot=timeline_plot_path,
            output_html=timeline_html_path,
            fps=fps,
            time_unit="ms",
            vapor_quality=vapor_quality,
            mass_flux=mass_flux,
            multi_panel=True,
        )
        print(f"Generated timeline plot     : {timeline_plot_path}")
        print(f"Generated interactive viewer: {timeline_html_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D Bubble Reconstruction and Void Fraction Analysis in SI Units"
    )
    parser.add_argument(
        "--masks",
        "-m",
        type=Path,
        required=True,
        help="Path to folder containing binary mask PNGs",
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
        default=None,
        help="Pixel size in m/px (or mm/px). Required unless a 0_*calibration.txt sits next to the masks",
    )
    parser.add_argument(
        "--diameter",
        type=float,
        default=0.020,
        help="Channel inner diameter in meters (SI, default: 0.020 m = 20 mm) or mm",
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
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render 3D flow reconstruction image artifact",
    )
    parser.add_argument(
        "--render-output",
        type=Path,
        default=None,
        help="Path for rendered 3D image artifact",
    )
    parser.add_argument(
        "--pipe-center-y",
        type=float,
        default=None,
        help="Optional vertical y-coordinate of pipe centerline in pixels (default: h / 2.0)",
    )
    parser.add_argument(
        "--regime-thresh-px",
        type=float,
        default=None,
        help="Optional pixel area threshold between small bubble and slug regimes (default: pi * R^2)",
    )
    parser.add_argument(
        "--plot-timeline",
        action="store_true",
        help="Automatically generate publication-quality void fraction timeline plot and interactive dashboard",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2000.0,
        help="Camera framerate in FPS for time mapping (default: 2000.0)",
    )
    args = parser.parse_args()

    # No default optical scale: take it from --pixel-size or from a calibration file.
    if str(Path(__file__).resolve().parents[1]) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from image_scripts.calibrate_pixel_size import resolve_pixel_size

    run_reconstruction(
        args.masks,
        args.output_csv,
        resolve_pixel_size(args.pixel_size, args.masks),
        args.diameter,
        args.vapor_quality,
        args.mass_flux,
        render_sample=args.render,
        render_output=args.render_output,
        pipe_center_y_px=args.pipe_center_y,
        regime_thresh_px=args.regime_thresh_px,
        plot_timeline=args.plot_timeline,
        fps=args.fps,
    )
