#!/usr/bin/env python3
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
    max_sigma_threshold: float = 0.35,
    wall_thickness_m: float = 0.003,
    inner_diameter_m: float = 0.020,
    pipe_center_y_px: float | None = None,
    max_bubble_area_px: float | None = None,
    min_circularity: float = 0.55,
) -> list[dict[str, Any]]:
    """
    Extracts 3D coordinates (x_m, y_m, z_m) and in-focus physical diameters (d0_m)
    for each detected bubble in the frame using single-camera DFD.

    Physical Confinement & Reliability Rules:
      1. Slug Exclusion: Slugs and highly deformed non-circular entities (aspect ratio > 2.5
         or circularity < min_circularity or area > max_bubble_area_px) are marked
         as 'slug_excluded_from_dfd'. Analytical DFD is strictly restricted to spherical/ellipsoidal bubbles.
      2. Glass Tube Wall Offset: The camera focal plane is set at the tube outer surface (z_focus = 0).
         The borosilicate glass wall has thickness wall_thickness_m (3 mm), so the fluid region spans
         z in [z_fluid_front, z_fluid_front + L_chord(y)], where z_fluid_front = 3 mm.
      3. Transverse Chord Confinement: Across vertical elevation y, depth is bounded by the circular tube chord:
         L_chord(y) = 2 * sqrt(max(0, R_tube^2 - (y - y_center)^2)).
         The physical rear wall boundary is z_max_phys(y) = wall_thickness_m + L_chord(y).
      4. Defocus Thresholding:
         - sigma_tilde < near_wall_sigma_threshold (0.05): Smoothstep regularization applied.
         - sigma_tilde > max_sigma_threshold (0.35): Marked as unreliable / saturated defocus.
    """
    h, w = raw_img.shape

    # Default pipe center if not specified: vertical image midpoint
    if pipe_center_y_px is None:
        pipe_center_y_px = float(h) / 2.0

    r_tube_m = inner_diameter_m / 2.0
    r_tube_px = r_tube_m / pixel_size_m
    if max_bubble_area_px is None:
        max_bubble_area_px = np.pi * (r_tube_px**2)

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

    # Compute spatial image gradients (scaled by 1/8 to yield true spatial derivative for ksize=3)
    sobel_x = cv2.Sobel(norm_img, cv2.CV_64F, 1, 0, ksize=3, scale=1.0 / 8.0)
    sobel_y = cv2.Sobel(norm_img, cv2.CV_64F, 0, 1, ksize=3, scale=1.0 / 8.0)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    for i in range(1, num_labels):
        area_px = stats[i, cv2.CC_STAT_AREA]
        if area_px < 15:  # Ignore microscopic specks
            continue

        c_x_px, c_y_px = float(centroids[i][0]), float(centroids[i][1])
        w_blob = stats[i, cv2.CC_STAT_WIDTH]
        h_blob = stats[i, cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(w_blob, h_blob) / max(1, min(w_blob, h_blob))
        r_t_px = np.sqrt(area_px / np.pi)

        blob_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        # Morphological circularity check
        perimeter_px = cv2.arcLength(contours[0], True)
        circularity = (4.0 * np.pi * area_px) / (perimeter_px**2) if perimeter_px > 0 else 0.0

        # Physical chord length available inside the circular tube at this transverse elevation y
        delta_y_px = abs(c_y_px - pipe_center_y_px)
        delta_y_m = delta_y_px * pixel_size_m
        if delta_y_m < r_tube_m:
            l_chord_m = 2.0 * np.sqrt(r_tube_m**2 - delta_y_m**2)
        else:
            l_chord_m = 0.0

        z_chord_half_m = 0.5 * l_chord_m
        z_front_inner_m = wall_thickness_m + (r_tube_m - z_chord_half_m)
        z_rear_inner_m = wall_thickness_m + (r_tube_m + z_chord_half_m)

        x_m = c_x_px * pixel_size_m
        y_m = c_y_px * pixel_size_m

        # Rule 1: Exclude large slugs or highly deformed bodies from DFD
        is_slug = (
            (area_px >= max_bubble_area_px)
            or (aspect_ratio > 2.5)
            or (circularity < min_circularity)
        )
        if is_slug:
            results.append(
                {
                    "bubble_id": i,
                    "centroid_x_m": x_m,
                    "centroid_y_m": y_m,
                    "depth_z_m": float(wall_thickness_m + r_tube_m),
                    "depth_fluid_z_m": float(0.5 * l_chord_m),
                    "diameter_d0_m": float(2.0 * r_t_px * pixel_size_m),
                    "blur_sigma_px": 0.0,
                    "sigma_tilde": 0.0,
                    "g_tilde": 0.0,
                    "near_wall_zone": False,
                    "is_reliable_dfd": False,
                    "dfd_status": "slug_excluded",
                    "area_px": area_px,
                    "circularity": float(circularity),
                    "aspect_ratio": float(aspect_ratio),
                    "l_chord_m": float(l_chord_m),
                    "z_rear_wall_m": float(z_rear_inner_m),
                    "centroid_x_px": c_x_px,
                    "centroid_y_px": c_y_px,
                }
            )
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
        # to pixel discretization and noise.
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

        # Raw DFD distance from optical focus (outer wall): z_raw = sigma / beta_eff
        z_raw_m = sigma_px / beta_eff_px_m

        # Fluid depth measured from the front inner wall into the liquid domain
        depth_fluid_z_m = float(np.clip(z_raw_m, 0.0, l_chord_m))
        depth_z_m = float(z_front_inner_m + depth_fluid_z_m)

        # Categorize DFD reliability status:
        if s_tilde > max_sigma_threshold or z_raw_m >= l_chord_m:
            dfd_status = "saturated_rear_wall" if z_raw_m >= l_chord_m else "unreliable_defocus"
            is_reliable = False
        elif is_near_wall:
            dfd_status = "near_wall_regularized"
            is_reliable = True
        else:
            dfd_status = "valid"
            is_reliable = True

        results.append(
            {
                "bubble_id": i,
                "centroid_x_m": x_m,
                "centroid_y_m": y_m,
                "depth_z_m": depth_z_m,
                "depth_fluid_z_m": depth_fluid_z_m,
                "diameter_d0_m": d0_m,
                "blur_sigma_px": sigma_px,
                "sigma_tilde": s_tilde,
                "g_tilde": g_tilde,
                "near_wall_zone": bool(is_near_wall),
                "is_reliable_dfd": is_reliable,
                "dfd_status": dfd_status,
                "area_px": area_px,
                "circularity": float(circularity),
                "aspect_ratio": float(aspect_ratio),
                "l_chord_m": float(l_chord_m),
                "z_rear_wall_m": float(z_rear_inner_m),
                "centroid_x_px": c_x_px,
                "centroid_y_px": c_y_px,
            }
        )

    return results


def render_dfd_diagnostic(
    raw_img: np.ndarray,
    binary_mask: np.ndarray,
    results: list[dict[str, Any]],
    output_path: Path,
    pipe_center_y_px: float,
    r_tube_px: float,
    pixel_size_m: float,
    wall_thickness_m: float = 0.003,
    inner_diameter_m: float = 0.020,
    show_labels: bool = False,
):
    """
    Renders publication-grade 2-panel diagnostic figure:
      Panel (a): Front View (x-y) overlaying raw image with segmented bubbles color-coded by DFD status.
      Panel (b): Transverse Cross-Section (y-z) showing the circular tube profile, glass wall,
                 and the reconstructed out-of-plane depth of each bubble.
    """
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, (ax_front, ax_side) = plt.subplots(
        1, 2, figsize=(11.5, 4.2), dpi=300, gridspec_kw={"width_ratios": [2.25, 1.0]}
    )

    # Color mapping for statuses
    status_colors = {
        "valid": "#00E676",  # Bright green
        "near_wall_regularized": "#00B0FF",  # Cyan / blue
        "saturated_rear_wall": "#FF5252",  # Coral / red
        "unreliable_defocus": "#FF9100",  # Orange
        "slug_excluded": "#9E9E9E",  # Grey
    }

    # Soft translucent color tint on bubble silhouettes in front view
    def hex_to_rgb(hex_str: str) -> tuple[float, float, float]:
        hex_clean = hex_str.lstrip("#")
        return tuple(int(hex_clean[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

    status_colors_rgb = {k: hex_to_rgb(v) for k, v in status_colors.items()}

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    res_by_id = {r["bubble_id"]: r for r in results}

    if len(raw_img.shape) == 2:
        base_rgb = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB).astype(float) / 255.0
    else:
        base_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB).astype(float) / 255.0

    overlay = base_rgb.copy()
    for b_id, res in res_by_id.items():
        st = res["dfd_status"]
        c_rgb = np.array(status_colors_rgb.get(st, (1.0, 1.0, 1.0)))
        b_mask = labels == b_id
        overlay[b_mask] = 0.72 * overlay[b_mask] + 0.28 * c_rgb

    ax_front.imshow(overlay, aspect="equal")

    # Draw tube inner walls on front view
    h_img, w_img = raw_img.shape[:2]
    y_top_wall = pipe_center_y_px - r_tube_px
    y_bot_wall = pipe_center_y_px + r_tube_px
    ax_front.axhline(
        y_top_wall, color="#FFC107", ls="--", lw=1.2, alpha=0.85, label="Upper tube wall"
    )
    ax_front.axhline(
        y_bot_wall, color="#FFC107", ls="--", lw=1.2, alpha=0.85, label="Lower tube wall"
    )
    ax_front.axhline(
        pipe_center_y_px, color="#17A2B8", ls=":", lw=1.0, alpha=0.6, label="Tube centerline"
    )

    for b_id, res in res_by_id.items():
        st = res["dfd_status"]
        color = status_colors.get(st, "#FFFFFF")
        cx = res["centroid_x_px"]
        cy = res["centroid_y_px"]
        d0_mm = res["diameter_d0_m"] * 1e3
        z_mm = res["depth_fluid_z_m"] * 1e3
        res["sigma_tilde"]

        # Contour
        b_mask = (labels == b_id).astype(np.uint8)
        cnts, _ = cv2.findContours(b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            for cnt in cnts:
                pts = cnt.squeeze()
                if len(pts.shape) == 2:
                    ax_front.plot(pts[:, 0], pts[:, 1], color=color, lw=1.4)

        if show_labels:
            if st == "slug_excluded":
                lbl_txt = f"ID {b_id} (Slug)\n[EDT 3D]"
            elif st == "saturated_rear_wall":
                lbl_txt = f"ID {b_id}\nSAT: z={z_mm:.1f}mm"
            elif st == "near_wall_regularized":
                lbl_txt = f"ID {b_id}\nNear-wall: z={z_mm:.2f}mm"
            else:
                lbl_txt = f"ID {b_id}\nz={z_mm:.1f}mm | d={d0_mm:.2f}mm"

            ax_front.text(
                cx,
                cy,
                lbl_txt,
                ha="center",
                va="center",
                fontsize=6.8,
                fontweight="bold",
                color=color,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="#000000")],
            )

    ax_front.set_title(
        r"$\mathbf{(a)}$ Front Projection ($x-y$): DFD Reliability Classification",
        fontsize=9.5,
        color="#1B365D",
    )
    ax_front.set_xlabel("Axial coordinate $x$ [px]")
    ax_front.set_ylabel("Transverse coordinate $y$ [px]")
    ax_front.set_xlim(0, w_img)
    ax_front.set_ylim(h_img, 0)  # Invert for image coordinates

    # Panel (b): Transverse Cross-Section (y-z)
    # y in [-R, R] (mm), z in [-wall, 2R + wall] (mm)
    r_tube_mm = (inner_diameter_m / 2.0) * 1e3
    wall_mm = wall_thickness_m * 1e3
    d_tube_mm = inner_diameter_m * 1e3

    # Draw outer circular tube wall
    theta = np.linspace(0, 2 * np.pi, 200)
    # Center of tube in z: z_center = wall_mm + r_tube_mm
    z_center_mm = wall_mm + r_tube_mm
    y_center_mm = 0.0

    z_inner = z_center_mm + r_tube_mm * np.cos(theta)
    y_inner = y_center_mm + r_tube_mm * np.sin(theta)
    z_outer = z_center_mm + (r_tube_mm + wall_mm) * np.cos(theta)
    y_outer = y_center_mm + (r_tube_mm + wall_mm) * np.sin(theta)

    ax_side.plot(
        z_outer, y_outer, color="#888888", lw=1.2, ls="--", label="Outer wall (focus $z=0$)"
    )
    ax_side.plot(z_inner, y_inner, color="#1B365D", lw=1.8, label="Inner tube wall")
    ax_side.fill(
        np.append(z_outer, z_inner[::-1]),
        np.append(y_outer, y_inner[::-1]),
        color="#E0E0E0",
        alpha=0.5,
    )

    # Optical focus plane at z = 0
    ax_side.axvline(0.0, color="#E91E63", ls=":", lw=1.3, label="Focal plane ($z_f=0$)")
    # Front inner wall at z = wall_mm
    ax_side.axvline(wall_mm, color="#00BCD4", ls=":", lw=1.1, alpha=0.7)

    # Plot bubble positions in cross section
    for b_id, res in res_by_id.items():
        st = res["dfd_status"]
        color = status_colors.get(st, "#FFFFFF")
        cy_px = res["centroid_y_px"]
        delta_y_mm = (cy_px - pipe_center_y_px) * (pixel_size_m * 1e3)
        y_elev_mm = -delta_y_mm

        # Bound elevation within circular tube profile
        y_elev_clamped_mm = float(np.clip(y_elev_mm, -r_tube_mm * 0.995, r_tube_mm * 0.995))
        z_chord_half_mm = float(np.sqrt(max(0.0, r_tube_mm**2 - y_elev_clamped_mm**2)))
        z_front_inner_mm = z_center_mm - z_chord_half_mm
        z_rear_inner_mm = z_center_mm + z_chord_half_mm

        # Fluid depth measured from front inner wall into fluid
        z_fluid_mm = float(res.get("depth_fluid_z_m", 0.0)) * 1e3
        d0_mm = float(res.get("diameter_d0_m", 0.0)) * 1e3
        r0_mm = max(0.3, d0_mm / 2.0)

        # Optical depth coordinate
        if st == "slug_excluded":
            z_bubble_mm = z_center_mm
        else:
            z_bubble_raw_mm = z_front_inner_mm + z_fluid_mm
            z_bubble_mm = float(
                np.clip(z_bubble_raw_mm, z_front_inner_mm + 0.15, z_rear_inner_mm - 0.15)
            )

        # Check distance from tube center (z_center_mm, 0.0)
        dist_from_tube_center = np.sqrt((z_bubble_mm - z_center_mm) ** 2 + y_elev_clamped_mm**2)
        max_allowed_center_dist = r_tube_mm - 0.25
        if dist_from_tube_center > max_allowed_center_dist:
            scale_center = max_allowed_center_dist / dist_from_tube_center
            z_bubble_mm = z_center_mm + (z_bubble_mm - z_center_mm) * scale_center
            y_elev_clamped_mm = y_elev_clamped_mm * scale_center
            dist_from_tube_center = max_allowed_center_dist

        # Compute exact radial clearance to the circular tube wall
        radial_clearance_mm = max(0.18, r_tube_mm - dist_from_tube_center)

        # Scale visual patch radius so it remains strictly inside tube bounds in all directions
        r0_plot_mm = min(r0_mm, max(0.18, radial_clearance_mm * 0.92))

        # Plot circle representing bubble
        c_patch = Circle(
            (z_bubble_mm, y_elev_clamped_mm),
            r0_plot_mm,
            facecolor=color,
            edgecolor="#000000",
            lw=0.8,
            alpha=0.85,
        )
        ax_side.add_patch(c_patch)
        if show_labels:
            ax_side.text(
                z_bubble_mm,
                y_elev_clamped_mm,
                str(b_id),
                ha="center",
                va="center",
                fontsize=6.5,
                fontweight="bold",
                color="#FFFFFF",
                path_effects=[pe.withStroke(linewidth=1.8, foreground="#000000")],
            )

    ax_side.set_title(r"$\mathbf{(b)}$ Transverse Profile ($z-y$)", fontsize=9.5, color="#1B365D")
    ax_side.set_xlabel("Optical depth $z$ [mm]")
    ax_side.set_ylabel("Transverse elevation $y$ [mm]")
    ax_side.set_xlim(-1.0, wall_mm * 2 + d_tube_mm + 1.0)
    ax_side.set_ylim(-r_tube_mm - wall_mm - 1.0, r_tube_mm + wall_mm + 1.0)
    ax_side.grid(True, ls=":", alpha=0.4)
    ax_side.set_box_aspect(1)
    ax_side.set_aspect("equal", adjustable="box")

    # Add custom legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#00E676",
            markeredgecolor="k",
            markersize=7,
            label=r"Valid ($\tilde{\sigma} \leq 0.35$)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#00B0FF",
            markeredgecolor="k",
            markersize=7,
            label=r"Near-wall ($\tilde{\sigma} < 0.05$)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#FF5252",
            markeredgecolor="k",
            markersize=7,
            label=r"Saturated Rear Wall",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#FF9100",
            markeredgecolor="k",
            markersize=7,
            label=r"Unreliable Blur ($\tilde{\sigma} > 0.35$)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#9E9E9E",
            markeredgecolor="k",
            markersize=7,
            label=r"Slug (Excluded)",
        ),
    ]
    ax_front.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=7.2,
        framealpha=0.95,
        facecolor="#FFFFFF",
        edgecolor="#B0BEC5",
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[DONE] Saved DFD diagnostic visualization to: {output_path}")


def run_dfd_pipeline(
    raw_images_dir: Path,
    masks_dir: Path,
    output_csv: Path,
    pixel_size: float = 6.7e-5,
    beta_eff: float = 2000.0,
    background_path: Path | None = None,
    near_wall_threshold: float = 0.05,
    max_sigma_threshold: float = 0.35,
    wall_thickness_m: float = 0.003,
    inner_diameter_m: float = 0.020,
    pipe_center_y_px: float | None = None,
    render_diagnostic: bool = False,
    diagnostic_output_dir: Path | None = None,
):
    """
    Runs single-camera Depth from Defocus analysis across all paired images and masks.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if render_diagnostic and diagnostic_output_dir is not None:
        diagnostic_output_dir.mkdir(parents=True, exist_ok=True)

    pixel_size_m = pixel_size * 1e-3 if pixel_size > 0.001 else pixel_size
    r_tube_px = (inner_diameter_m / 2.0) / pixel_size_m
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
    print(f"  Wall thickness (SI) : {wall_thickness_m * 1e3:.1f} mm")
    print(f"  Channel diameter    : {inner_diameter_m * 1e3:.1f} mm")
    print(
        f"  Reliability bounds  : sigma_tilde in [{near_wall_threshold:.2f}, {max_sigma_threshold:.2f}]"
    )

    for idx, f_mask in enumerate(mask_files):
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

        frame_center_y = (
            pipe_center_y_px if pipe_center_y_px is not None else float(raw_img.shape[0]) / 2.0
        )

        frame_results = analyze_frame_dfd(
            raw_img,
            mask,
            dfd_model,
            pixel_size_m=pixel_size_m,
            beta_eff_px_m=beta_eff,
            background_img=bg_img,
            near_wall_sigma_threshold=near_wall_threshold,
            max_sigma_threshold=max_sigma_threshold,
            wall_thickness_m=wall_thickness_m,
            inner_diameter_m=inner_diameter_m,
            pipe_center_y_px=frame_center_y,
        )

        for r in frame_results:
            r["frame"] = f_mask.stem
            all_records.append(r)

        if (
            render_diagnostic
            and diagnostic_output_dir is not None
            and (idx < 5 or idx == len(mask_files) - 1)
        ):
            diag_png = diagnostic_output_dir / f"dfd_diag_{f_mask.stem}.png"
            render_dfd_diagnostic(
                raw_img,
                mask,
                frame_results,
                diag_png,
                pipe_center_y_px=frame_center_y,
                r_tube_px=r_tube_px,
                pixel_size_m=pixel_size_m,
                wall_thickness_m=wall_thickness_m,
                inner_diameter_m=inner_diameter_m,
            )

    df = pd.DataFrame(all_records)
    df.to_csv(output_csv, index=False)

    print("\n" + "=" * 56)
    print("      DEPTH FROM DEFOCUS (DFD) SUMMARY (SI)      ")
    print("=" * 56)
    print(f"Total Entities Detected      : {len(df)}")
    if not df.empty:
        n_valid = int((df["dfd_status"] == "valid").sum())
        n_near = int((df["dfd_status"] == "near_wall_regularized").sum())
        n_sat = int((df["dfd_status"] == "saturated_rear_wall").sum())
        n_unreliable = int((df["dfd_status"] == "unreliable_defocus").sum())
        n_slug = int((df["dfd_status"] == "slug_excluded").sum())

        print(f"  Valid Bubbles (Core DFD)   : {n_valid} ({n_valid / len(df) * 100:.1f}%)")
        print(f"  Near-Wall Regularized      : {n_near} ({n_near / len(df) * 100:.1f}%)")
        print(f"  Saturated Rear Wall Bound  : {n_sat} ({n_sat / len(df) * 100:.1f}%)")
        print(
            f"  Unreliable Defocus (>0.35) : {n_unreliable} ({n_unreliable / len(df) * 100:.1f}%)"
        )
        print(f"  Slugs Excluded (Regime 2)  : {n_slug} ({n_slug / len(df) * 100:.1f}%)")

        valid_df = df[df["is_reliable_dfd"]]
        if not valid_df.empty:
            print(f"\nReliable Bubble Metrics (N = {len(valid_df)}):")
            print(f"  Mean Diameter (d0)         : {valid_df['diameter_d0_m'].mean() * 1e3:.3f} mm")
            print(
                f"  Mean Depth in Fluid (z)    : {valid_df['depth_fluid_z_m'].mean() * 1e3:.3f} mm"
            )
            print(
                f"  Max Depth in Fluid (z)     : {valid_df['depth_fluid_z_m'].max() * 1e3:.3f} mm"
            )
    print("=" * 56)
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
        default=6.116e-5,
        help="Pixel size in m/px (SI, default: 6.116e-5 m/px)",
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
        "--max-sigma-threshold",
        type=float,
        default=0.35,
        help="Upper dimensionless blur threshold (sigma_tilde) for DFD validity (default: 0.35)",
    )
    parser.add_argument(
        "--wall-thickness",
        type=float,
        default=0.003,
        help="Borosilicate glass wall thickness in meters (default: 0.003 m = 3.0 mm)",
    )
    parser.add_argument(
        "--diameter",
        "-d",
        type=float,
        default=0.020,
        help="Channel inner diameter in meters (default: 0.020 m = 20.0 mm)",
    )
    parser.add_argument(
        "--pipe-center-y",
        type=float,
        default=None,
        help="Pipe centerline y [px]",
    )
    parser.add_argument(
        "--background",
        "-bg",
        type=Path,
        default=None,
        help="Optional background image for normalization",
    )
    parser.add_argument(
        "--render-diagnostic",
        action="store_true",
        help="Render 2-panel diagnostic visualization showing bubble status and depth profile",
    )
    parser.add_argument(
        "--diagnostic-dir",
        type=Path,
        default=Path("results/dfd_diagnostics"),
        help="Output directory for diagnostic images",
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
        max_sigma_threshold=args.max_sigma_threshold,
        wall_thickness_m=args.wall_thickness,
        inner_diameter_m=args.diameter,
        pipe_center_y_px=args.pipe_center_y,
        render_diagnostic=args.render_diagnostic,
        diagnostic_output_dir=args.diagnostic_dir,
    )
