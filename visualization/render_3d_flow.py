#!/usr/bin/env python3
"""
3D Bubble Reconstruction and Rendering Engine in Standard SI Units.

Reconstructs 3D gas flow structures from 2D planar high-speed binary masks
and renders publication-grade 3D visualizations matching the reference specifications:
  - Dual-regime geometry: 3D ellipsoids for small bubbles, distance transform + Laplacian
    depth extrusion for large slugs.
  - Cylindrical confinement: local circular chord depth z_chord(y) = sqrt(max(0, R^2 - delta_y^2)).
  - Strict SI Units: all physical measurements, lengths, diameters, areas, volumes,
    and coordinates are computed in meters (m), square meters (m^2), and cubic meters (m^3).
  - High-fidelity visual styling: wireframe cylindrical pipe (dark navy), 3D bubble isosurfaces
    (deep maroon with Gouraud/specular lighting), MATLAB-style boxed canvas.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from scipy.ndimage import distance_transform_edt


def reconstruct_3d_bubbles_si(
    mask_2d: np.ndarray,
    pixel_size_m: float,
    pipe_dia_m: float = 0.020,
    area_thresh_px: float = 5000.0,
    smoothing_lambda_px: float = 0.15,
    dfd_records: list[dict[str, Any]] | None = None,
    y_center_px: float | None = None,
) -> dict[str, Any]:
    """
    Reconstructs 3D bubble volumes from a 2D binary mask strictly in SI units.

    Parameters:
        mask_2d: Binary mask array (H, W) where >0 is gas and 0 is liquid.
        pixel_size_m: Optical scale in meters/pixel (required; there is no default).
        pipe_dia_m: Channel internal diameter in meters (default: 0.020 m = 20.0 mm).
        area_thresh_px: Threshold area in pixels distinguishing small vs large bubbles (default 5000 px).
        smoothing_lambda_px: Laplacian boundary smoothing factor (default 0.15).
        dfd_records: Optional single-camera DFD depth estimates.
        y_center_px: Optional optical channel centerline coordinate in pixels.

    Returns:
        Dictionary containing reconstructed volumes, geometries, and void fraction in SI units.
    """
    mask_binary = (mask_2d > 0).astype(np.uint8)
    h_px, w_px = mask_binary.shape

    # Physical dimensions in standard SI units (meters)
    pipe_radius_m = pipe_dia_m / 2.0
    pipe_length_m = w_px * pixel_size_m
    v_pipe_m3 = np.pi * (pipe_radius_m**2) * pipe_length_m

    # Pixel coordinate scale (centered vertically by default or calibration axis)
    if y_center_px is None:
        y_center_px = h_px / 2.0
    y_center_m = y_center_px * pixel_size_m

    # Scale-consistent SI smoothing parameter: lambda_si * D_si = lambda_px * D_px
    smoothing_lambda_si = smoothing_lambda_px / pixel_size_m

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_binary, connectivity=8
    )

    # Pre-calculate 1D vertical chord depth profile: z_chord(y) = sqrt(max(0, R^2 - delta_y^2))
    row_indices = np.arange(h_px, dtype=np.float64)
    delta_y_m = np.abs((row_indices - y_center_px) * pixel_size_m)
    r2_minus_y2 = np.maximum(0.0, (pipe_radius_m**2) - (delta_y_m**2))
    z_chord_profile_m = np.sqrt(r2_minus_y2)
    z_chord_map_2d_m = np.repeat(z_chord_profile_m[:, np.newaxis], w_px, axis=1)

    total_v_gas_m3 = 0.0
    bubble_entities: list[dict[str, Any]] = []

    for i in range(1, num_labels):
        area_px = stats[i, cv2.CC_STAT_AREA]
        if area_px < 3:
            continue

        blob_mask = (labels == i).astype(np.uint8)
        bx_px = int(stats[i, cv2.CC_STAT_LEFT])
        by_px = int(stats[i, cv2.CC_STAT_TOP])
        bw_px = int(stats[i, cv2.CC_STAT_WIDTH])
        bh_px = int(stats[i, cv2.CC_STAT_HEIGHT])

        c_x_px = float(centroids[i][0])
        c_y_px = float(centroids[i][1])

        # Centroid position in SI units
        c_x_m = c_x_px * pixel_size_m
        c_y_m = c_y_px * pixel_size_m

        # Local tube chord depth available at centroid transverse location
        delta_yc_m = abs(c_y_m - y_center_m)
        z_local_max_m = np.sqrt(max(0.0, pipe_radius_m**2 - delta_yc_m**2))

        area_m2 = float(area_px * (pixel_size_m**2))

        # Regime 1: Small Bubbles (< 5000 px / < area_thresh_m2) -> 3D Ellipsoid
        if area_px < area_thresh_px:
            contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            angle_rad = 0.0
            fitted = False

            if len(contours) > 0 and len(contours[0]) >= 5:
                try:
                    box = cv2.fitEllipse(contours[0])
                    d1, d2 = box[1]
                    angle_deg = box[2]
                    d_major = max(float(d1), float(d2))
                    d_minor = min(float(d1), float(d2))
                    if not (np.isnan(d_major) or np.isnan(d_minor) or d_minor <= 0 or d_major <= 0):
                        a_px = min(d_major / 2.0, max(bw_px, bh_px))
                        b_px = min(d_minor / 2.0, min(bw_px, bh_px))
                        angle_rad = np.radians(angle_deg)
                        fitted = True
                except (cv2.error, ValueError, TypeError):
                    fitted = False

            if not fitted:
                r_px = np.sqrt(area_px / np.pi)
                a_px, b_px = r_px, r_px
                angle_rad = 0.0
            else:
                # Area conservation guardrail (Week 37):
                # For irregular, concave, or odd-shaped non-convex bubbles, fitEllipse can
                # severely overestimate the 2D projected area (pi * a * b >> area_px).
                # We scale a and b to strictly conserve the actual measured 2D bubble area.
                ellipse_area_px = np.pi * a_px * b_px
                if ellipse_area_px > 0 and area_px > 0:
                    area_scale = min(1.0, np.sqrt(area_px / ellipse_area_px))
                    a_px *= area_scale
                    b_px *= area_scale

            # Convert axes to meters
            a_m = a_px * pixel_size_m
            b_m = b_px * pixel_size_m
            c_m = min(b_m, z_local_max_m)

            vol_i_m3 = (4.0 / 3.0) * np.pi * a_m * b_m * c_m
            total_v_gas_m3 += vol_i_m3

            # Incorporate physical depth offset from single-camera DFD if available
            depth_offset_m = 0.0
            if dfd_records:
                best_dist = 20.0
                best_rec = None
                for r in dfd_records:
                    rec_cx = r.get("centroid_x_px", -999.0)
                    rec_cy = r.get("centroid_y_px", -999.0)
                    dist = float(np.hypot(c_x_px - rec_cx, c_y_px - rec_cy))
                    if dist < best_dist:
                        best_dist = dist
                        best_rec = r
                if best_rec is not None and "depth_fluid_z_m" in best_rec:
                    # Fluid depth is measured from front inner wall into the fluid at local chord height:
                    # Front wall at centroid elevation is -z_local_max_m:
                    raw_offset = -z_local_max_m + float(best_rec["depth_fluid_z_m"])
                    depth_offset_m = float(
                        np.clip(raw_offset, -z_local_max_m + c_m, z_local_max_m - c_m)
                    )

            bubble_entities.append(
                {
                    "id": i,
                    "regime": "small_ellipsoid",
                    "area_m2": area_m2,
                    "volume_m3": vol_i_m3,
                    "centroid_m": (c_x_m, c_y_m, depth_offset_m),
                    "semi_axes_m": (a_m, b_m, c_m),
                    "orientation_rad": angle_rad,
                    "bbox_px": (bx_px, by_px, bw_px, bh_px),
                }
            )

        # Regime 2: Large Deformed Slugs (>= 5000 px) -> EDT + Laplacian Extrusion
        else:
            dist_map_px = distance_transform_edt(blob_mask)
            dist_map_m = dist_map_px * pixel_size_m

            # Laplacian boundary attenuation applied to cylindrical chord depth
            # D'(p) = D(p) * exp(-1 / (lambda * d(p) + eps)) (Eduardo Reis, Eq 3.2)
            depth_map_m = z_chord_map_2d_m * np.exp(
                -1.0 / (smoothing_lambda_si * dist_map_m + 1e-5)
            )

            # Strict cylindrical wall chord boundary constraint
            depth_map_bounded_m = np.minimum(depth_map_m, z_chord_map_2d_m)

            # Volumetric surface integration in m^3: 2 * sum(z * dA)
            dA_m2 = pixel_size_m**2
            vol_i_m3 = 2.0 * np.sum(depth_map_bounded_m[blob_mask == 1]) * dA_m2
            total_v_gas_m3 += vol_i_m3

            bubble_entities.append(
                {
                    "id": i,
                    "regime": "large_slug",
                    "area_m2": area_m2,
                    "volume_m3": vol_i_m3,
                    "centroid_m": (c_x_m, c_y_m, 0.0),
                    "depth_map_m": depth_map_bounded_m,
                    "blob_mask": blob_mask,
                    "bbox_px": (bx_px, by_px, bw_px, bh_px),
                }
            )

    alpha = float(np.clip(total_v_gas_m3 / (v_pipe_m3 + 1e-12), 0.0, 1.0))

    return {
        "mask_shape_px": (h_px, w_px),
        "pixel_size_m": pixel_size_m,
        "pipe_diameter_m": pipe_dia_m,
        "pipe_radius_m": pipe_radius_m,
        "pipe_length_m": pipe_length_m,
        "pipe_volume_m3": v_pipe_m3,
        "gas_volume_m3": total_v_gas_m3,
        "void_fraction_alpha": alpha,
        "bubble_count": len(bubble_entities),
        "bubbles": bubble_entities,
    }


def render_3d_flow_visualization(
    recon_data: dict[str, Any],
    output_png_path: Path,
    view_elev: float = 24.0,
    view_azim: float = -55.0,
    dpi: int = 180,
    title_suffix: str = "",
) -> Path:
    """
    Renders the reconstructed 3D multi-phase flow geometry to a publication-grade image.
    Faithfully mirrors the visual styling and mathematical precision of the reference version:
      - Deep navy wireframe cylindrical conduit cage.
      - Deep maroon 3D bubble isosurfaces with Gouraud/specular lighting.
      - Standard MATLAB-styled white coordinate frame panes on off-white canvas.
      - Full SI metric axis scales and comprehensive multi-line heading.
    """
    output_png_path = Path(output_png_path)
    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    h_px, _ = recon_data["mask_shape_px"]
    pixel_size_m = recon_data["pixel_size_m"]
    R_m = recon_data["pipe_radius_m"]
    L_m = recon_data["pipe_length_m"]
    alpha = recon_data["void_fraction_alpha"]
    alpha_pct = alpha * 100.0

    y_center_px = h_px / 2.0
    y_center_m = y_center_px * pixel_size_m

    fig = plt.figure(figsize=(12, 7.2), facecolor="#ECECEC")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    # Elongate cylinder along flow axis (X) to match natural aspect ratio
    ax.set_box_aspect((2.8, 1.0, 0.8))

    # White panes with soft gray grid matching reference visualization
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.xaxis._axinfo["grid"]["color"] = (0.88, 0.88, 0.88, 1.0)
    ax.yaxis._axinfo["grid"]["color"] = (0.88, 0.88, 0.88, 1.0)
    ax.zaxis._axinfo["grid"]["color"] = (0.88, 0.88, 0.88, 1.0)

    ls = LightSource(azdeg=315, altdeg=45)

    # 1. Wireframe cylindrical pipe cage (Navy Blue)
    N_rings = 45
    N_lines = 16
    theta = np.linspace(0, 2 * np.pi, 60)
    for x_pos in np.linspace(0, L_m, N_rings):
        y_ring = y_center_m + R_m * np.cos(theta)
        z_ring = R_m * np.sin(theta)
        ax.plot(
            x_pos * np.ones_like(theta),
            y_ring,
            z_ring,
            color="#000088",
            linewidth=0.75,
            alpha=0.9,
        )

    for th in np.linspace(0, 2 * np.pi, N_lines, endpoint=False):
        y_line = y_center_m + R_m * np.cos(th)
        z_line = R_m * np.sin(th)
        ax.plot(
            [0.0, L_m],
            [y_line, y_line],
            [z_line, z_line],
            color="#000088",
            linewidth=0.75,
            alpha=0.9,
        )

    # 2. Render reconstructed bubbles in SI coordinates
    for b in recon_data["bubbles"]:
        if b["regime"] == "small_ellipsoid":
            c_x_m, c_y_m, c_depth_offset_m = b["centroid_m"]
            # Map vertical image row to physical Z-axis (buoyancy direction)
            c_z_m = (y_center_px - (c_y_m / pixel_size_m)) * pixel_size_m
            a_m, b_m, c_m = b["semi_axes_m"]
            angle_rad = b["orientation_rad"]

            # Parametric mesh for 3D ellipsoid
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 12)
            X_e = a_m * np.outer(np.cos(u), np.sin(v))
            Z_e = b_m * np.outer(np.sin(u), np.sin(v))
            Y_e = c_m * np.outer(np.ones_like(u), np.cos(v))

            # Rotate in (X, Z) plane according to fitted 2D orientation
            X_rot = c_x_m + X_e * np.cos(angle_rad) - Z_e * np.sin(angle_rad)
            Z_rot = c_z_m + X_e * np.sin(angle_rad) + Z_e * np.cos(angle_rad)
            Y_rot = (y_center_m + c_depth_offset_m) + Y_e

            ax.plot_surface(
                X_rot,
                Y_rot,
                Z_rot,
                color="#780000",
                shade=True,
                lightsource=ls,
                alpha=0.98,
                edgecolors="#380000",
                linewidth=0.25,
            )

        elif b["regime"] == "large_slug":
            bx_px, by_px, bw_px, bh_px = b["bbox_px"]
            depth_map_m = b["depth_map_m"]
            blob_mask = b["blob_mask"]

            step = max(1, round(2.0))
            sub_y = slice(by_px, by_px + bh_px, step)
            sub_x = slice(bx_px, bx_px + bw_px, step)

            x_mesh_px = np.arange(bx_px, bx_px + bw_px, step)
            y_mesh_px = np.arange(by_px, by_px + bh_px, step)
            X_g, Y_mesh_2d = np.meshgrid(x_mesh_px * pixel_size_m, y_mesh_px)
            Z_g = (y_center_px - Y_mesh_2d) * pixel_size_m

            y_depth_crop = depth_map_m[sub_y, sub_x]
            mask_crop = blob_mask[sub_y, sub_x]

            Y_top = y_center_m + y_depth_crop
            Y_bot = y_center_m - y_depth_crop

            Y_top[mask_crop == 0] = np.nan
            Y_bot[mask_crop == 0] = np.nan

            ax.plot_surface(
                X_g,
                Y_top,
                Z_g,
                color="#780000",
                shade=True,
                lightsource=ls,
                rstride=1,
                cstride=1,
                alpha=0.98,
                edgecolors="#380000",
                linewidth=0.2,
            )
            ax.plot_surface(
                X_g,
                Y_bot,
                Z_g,
                color="#780000",
                shade=True,
                lightsource=ls,
                rstride=1,
                cstride=1,
                alpha=0.98,
                edgecolors="#380000",
                linewidth=0.2,
            )

    # Multi-line title matching reference specification
    title_str = (
        f"3D Bubble Reconstruction\n"
        f"Gaussian (small) + Laplacian Smoothing (large)\n"
        f"Void Fraction: {alpha_pct:.3f}%"
    )
    if title_suffix:
        title_str += f" | {title_suffix}"

    ax.set_title(title_str, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("X (flow direction) [m]", fontsize=11, labelpad=10)
    ax.set_ylabel("Y [m]", fontsize=11, labelpad=10)
    ax.set_zlabel("Z [m]", fontsize=11, labelpad=10)

    # Axis bounds strictly enforcing channel volume
    ax.set_xlim(0, L_m)
    ax.set_ylim(0, 300 * pixel_size_m)
    ax.set_zlim(-R_m, R_m)

    ax.view_init(elev=view_elev, azim=view_azim)

    plt.tight_layout()
    plt.savefig(str(output_png_path), dpi=dpi)
    plt.close(fig)

    return output_png_path


def extract_slug_mesh_data(
    blob_mask: np.ndarray,
    depth_map_m: np.ndarray,
    pixel_size_m: float,
    y_center_px: float,
    step: int = 3,
    pipe_radius_m: float | None = None,
) -> dict[str, Any]:
    """
    Constructs a watertight 3D surface mesh (vertices and triangle indices)
    from a 2.5D depth field for Three.js WebGL rendering.
    Generates top (+Z) and bottom (-Z) surface grids plus connecting boundary skirt walls.
    """
    by, bx = np.where(blob_mask)
    if len(by) == 0:
        return {"vertices": [], "indices": []}

    y_min, y_max = int(by.min()), int(by.max())
    x_min, x_max = int(bx.min()), int(bx.max())

    rows = np.arange(y_min, y_max + 1, step)
    cols = np.arange(x_min, x_max + 1, step)
    nr, nc = len(rows), len(cols)

    grid_top = np.full((nr, nc), -1, dtype=np.int32)
    grid_bot = np.full((nr, nc), -1, dtype=np.int32)

    vertices: list[float] = []
    v_idx = 0

    for r_idx, r in enumerate(rows):
        for c_idx, c in enumerate(cols):
            if blob_mask[r, c]:
                # Mapping to Three.js coordinates:
                # X: flow axis (meters)
                # Y: vertical buoyancy axis (meters, centered at tube midplane)
                # Z: depth axis (meters, perpendicular to viewing plane, centered at 0)
                x_m = float(c * pixel_size_m)
                y_m = float((y_center_px - r) * pixel_size_m)
                z_depth = float(depth_map_m[r, c])

                # Physical containment guardrail: strictly enforce cylindrical pipe envelope
                if pipe_radius_m is not None and pipe_radius_m > 0:
                    y_m = float(np.clip(y_m, -pipe_radius_m, pipe_radius_m))
                    z_chord_limit = float(np.sqrt(max(0.0, pipe_radius_m**2 - y_m**2)))
                    z_depth = min(z_depth, z_chord_limit)

                x_m = round(x_m, 6)
                y_m = round(y_m, 6)
                z_depth = round(z_depth, 6)

                # Top vertex (+z_depth)
                vertices.extend([x_m, y_m, z_depth])
                grid_top[r_idx, c_idx] = v_idx
                v_idx += 1

                # Bottom vertex (-z_depth)
                vertices.extend([x_m, y_m, -z_depth])
                grid_bot[r_idx, c_idx] = v_idx
                v_idx += 1

    indices: list[int] = []
    for r in range(nr - 1):
        for c in range(nc - 1):
            t_tl = int(grid_top[r, c])
            t_tr = int(grid_top[r, c + 1])
            t_bl = int(grid_top[r + 1, c])
            t_br = int(grid_top[r + 1, c + 1])

            b_tl = int(grid_bot[r, c])
            b_tr = int(grid_bot[r, c + 1])
            b_bl = int(grid_bot[r + 1, c])
            b_br = int(grid_bot[r + 1, c + 1])

            # Top surface triangles (CCW facing +Z)
            if t_tl >= 0 and t_tr >= 0 and t_bl >= 0:
                indices.extend([t_tl, t_bl, t_tr])
            if t_tr >= 0 and t_bl >= 0 and t_br >= 0:
                indices.extend([t_tr, t_bl, t_br])

            # Bottom surface triangles (CCW facing -Z)
            if b_tl >= 0 and b_tr >= 0 and b_bl >= 0:
                indices.extend([b_tl, b_tr, b_bl])
            if b_tr >= 0 and b_bl >= 0 and b_br >= 0:
                indices.extend([b_tr, b_br, b_bl])

            # Boundary skirts / walls closing the top and bottom sheets
            if t_tl >= 0 and t_tr >= 0 and (r == 0 or grid_top[r - 1, c] < 0):
                indices.extend([t_tl, t_tr, b_tr, t_tl, b_tr, b_tl])
            if t_bl >= 0 and t_br >= 0 and (r + 1 == nr - 1 or grid_top[r + 2, c] < 0):
                indices.extend([t_br, t_bl, b_bl, t_br, b_bl, b_br])
            if t_tl >= 0 and t_bl >= 0 and (c == 0 or grid_top[r, c - 1] < 0):
                indices.extend([t_bl, t_tl, b_tl, t_bl, b_tl, b_bl])
            if t_tr >= 0 and t_br >= 0 and (c + 1 == nc - 1 or grid_top[r, c + 2] < 0):
                indices.extend([t_tr, t_br, b_br, t_tr, b_br, t_tr])

    return {"vertices": vertices, "indices": indices}


def render_interactive_html(recon_data: dict[str, Any], output_html_path: Path) -> Path:
    """
    Generates a standalone, interactive 3D WebGL HTML visualization of the reconstructed tube
    and bubbles (both small ellipsoids and large slugs) using Three.js via an embedded browser widget.
    """
    import json

    output_html_path = Path(output_html_path)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    R_m = recon_data["pipe_radius_m"]
    L_m = recon_data["pipe_length_m"]
    pixel_size_m = recon_data["pixel_size_m"]
    alpha_pct = recon_data["void_fraction_alpha"] * 100.0
    y_center_px = recon_data["mask_shape_px"][0] / 2.0

    spheres = []
    slugs = []

    for b in recon_data["bubbles"]:
        if b["regime"] == "small_ellipsoid":
            cx, cy, _ = b["centroid_m"]
            cz = (
                recon_data["mask_shape_px"][0] / 2.0 - cy / recon_data["pixel_size_m"]
            ) * recon_data["pixel_size_m"]
            a, b_axis, c = b["semi_axes_m"]
            spheres.append(
                {
                    "x": float(cx),
                    "y": float(cz),
                    "z": 0.0,
                    "rx": float(a),
                    "ry": float(b_axis),
                    "rz": float(c),
                    "rotation_rad": float(b.get("orientation_rad", 0.0)),
                }
            )
        elif b["regime"] == "large_slug":
            bw_px = b["bbox_px"][2]
            bh_px = b["bbox_px"][3]
            step = max(2, int(round(max(bw_px, bh_px) / 100.0)))
            slug_mesh = extract_slug_mesh_data(
                b["blob_mask"],
                b["depth_map_m"],
                pixel_size_m=pixel_size_m,
                y_center_px=y_center_px,
                step=step,
            )
            if slug_mesh["vertices"] and slug_mesh["indices"]:
                slugs.append(slug_mesh)

    spheres_json = json.dumps(spheres)
    slugs_json = json.dumps(slugs)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>3D Bubble Reconstruction (SI Units)</title>
    <style>
        body {{ margin: 0; background-color: #ECECEC; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; overflow: hidden; }}
        #header {{
            position: absolute; top: 12px; width: 100%; text-align: center;
            pointer-events: none; color: #111; text-shadow: 0 1px 2px rgba(255,255,255,0.8);
        }}
        #header h2 {{ margin: 2px; font-size: 20px; font-weight: 700; }}
        #header p {{ margin: 3px; font-size: 14px; color: #333; }}
        #hud {{
            position: absolute; bottom: 15px; left: 15px; background: rgba(255,255,255,0.92);
            padding: 12px 16px; border-radius: 8px; font-size: 13px; line-height: 1.5;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15); border: 1px solid #ccc;
        }}
        #hud b.title {{ font-size: 14px; color: #000088; }}
        #legend {{
            position: absolute; top: 15px; right: 15px; background: rgba(255,255,255,0.92);
            padding: 10px 14px; border-radius: 8px; font-size: 12px; line-height: 1.5;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15); border: 1px solid #ccc;
        }}
        .legend-item {{ display: flex; align-items: center; margin-bottom: 4px; }}
        .color-box {{ width: 14px; height: 14px; margin-right: 8px; border-radius: 3px; }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="header">
        <h2>3D Bubble Reconstruction (SI Units)</h2>
        <p>Gaussian (small) + Laplacian Smoothing (large) | Void Fraction: <b>{alpha_pct:.3f}%</b></p>
    </div>
    <div id="legend">
        <div class="legend-item"><div class="color-box" style="background:#000088; opacity:0.7;"></div> Pipe Conduit Wireframe</div>
        <div class="legend-item"><div class="color-box" style="background:#780000;"></div> 3D Vapor Bubbles / Slugs</div>
    </div>
    <div id="hud">
        <b class="title">SI Flow Parameters</b><br>
        Pixel scale (s_px): {pixel_size_m * 1e3:.3f} mm/px ({pixel_size_m:.4e} m/px)<br>
        Channel Diameter (D): {(R_m * 2) * 1e3:.2f} mm ({(R_m * 2):.5f} m)<br>
        Channel Length (L): {L_m * 1e3:.2f} mm ({L_m:.5f} m)<br>
        Total Gas Volume: {recon_data["gas_volume_m3"]:.3e} m³ ({recon_data["gas_volume_m3"] * 1e9:.2f} mm³)<br>
        Total Pipe Volume: {recon_data["pipe_volume_m3"]:.3e} m³ ({recon_data["pipe_volume_m3"] * 1e9:.2f} mm³)<br>
        Void Fraction (α): <b>{alpha_pct:.3f}%</b> ({recon_data["void_fraction_alpha"]:.4f})<br>
        Structures: {len(spheres)} small ellipsoids, {len(slugs)} large slugs<br>
        <small style="color:#555;">Rotate: Left-Click + Drag | Pan: Right-Click + Drag | Zoom: Scroll</small>
    </div>
    <script>
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xECECEC);
        const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.0005, 10);
        camera.position.set({L_m * 0.5}, {R_m * 1.6}, {R_m * 3.4});

        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.05;
        document.body.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set({L_m * 0.5}, 0, 0);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.update();

        // 3D Illumination
        scene.add(new THREE.AmbientLight(0xffffff, 0.75));

        const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.75);
        dirLight1.position.set(1, 2, 2).normalize();
        scene.add(dirLight1);

        const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.45);
        dirLight2.position.set(-1, -1, -2).normalize();
        scene.add(dirLight2);

        const dirLight3 = new THREE.DirectionalLight(0xffffff, 0.3);
        dirLight3.position.set(0, 3, 0).normalize();
        scene.add(dirLight3);

        // Cylindrical pipe wireframe (Navy Blue)
        const cylGeom = new THREE.CylinderGeometry({R_m}, {R_m}, {L_m}, 64, 40, true);
        cylGeom.rotateZ(Math.PI / 2);
        cylGeom.translate({L_m / 2}, 0, 0);
        const wireframe = new THREE.WireframeGeometry(cylGeom);
        const line = new THREE.LineSegments(wireframe);
        line.material.color.setHex(0x000088);
        line.material.opacity = 0.55;
        line.material.transparent = true;
        scene.add(line);

        // Material for all bubbles (Deep maroon with Gouraud/Phong specular reflection)
        const bubbleMat = new THREE.MeshPhongMaterial({{
            color: 0x780000,
            emissive: 0x180000,
            specular: 0x663333,
            shininess: 75,
            side: THREE.DoubleSide
        }});

        // 1. Render Large Slugs (3D Watertight Meshes)
        const slugs = {slugs_json};
        slugs.forEach(slug => {{
            if (slug.vertices && slug.vertices.length > 0 && slug.indices && slug.indices.length > 0) {{
                const geom = new THREE.BufferGeometry();
                geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(slug.vertices), 3));
                geom.setIndex(slug.indices);
                geom.computeVertexNormals();
                const mesh = new THREE.Mesh(geom, bubbleMat);
                scene.add(mesh);
            }}
        }});

        // 2. Render Small Bubbles (Ellipsoids)
        const spheres = {spheres_json};
        spheres.forEach(s => {{
            const geom = new THREE.SphereGeometry(1, 20, 16);
            geom.scale(s.rx, s.ry, s.rz);
            const mesh = new THREE.Mesh(geom, bubbleMat);
            mesh.position.set(s.x, s.y, s.z);
            if (s.rotation_rad) {{
                mesh.rotation.z = s.rotation_rad;
            }}
            scene.add(mesh);
        }});

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();

        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_html_path


def main():
    parser = argparse.ArgumentParser(
        description="3D Bubble Flow Reconstruction & Render Engine in SI Units"
    )
    parser.add_argument(
        "--mask",
        "-m",
        type=Path,
        required=True,
        help="Path to 2D binary segmentation mask image (.png, .tif, etc.)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("results/3d_bubble_render.png"),
        help="Path for rendered 3D output image",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Optical pixel scale in m/px. Required (or --pixel-size-mm) unless a "
        "0_*calibration.txt sits next to the mask",
    )
    parser.add_argument(
        "--pixel-size-mm",
        type=float,
        default=None,
        help="Optical pixel scale in mm/pixel (convenience alternative to --pixel-size)",
    )
    parser.add_argument(
        "--diameter",
        type=float,
        default=None,
        help="Channel inner diameter in meters (SI unit, e.g. 0.020)",
    )
    parser.add_argument(
        "--diameter-mm",
        type=float,
        default=20.0,
        help="Channel inner diameter in mm (convenience, default 20.0 mm)",
    )
    parser.add_argument(
        "--interactive-html",
        type=Path,
        default=None,
        help="Optional path to export interactive 3D WebGL HTML visualization",
    )
    parser.add_argument(
        "--smoothing-lambda",
        "--lambda",
        dest="smoothing_lambda",
        type=float,
        default=0.15,
        help="Laplacian boundary smoothing parameter lambda in pixels (default 0.15)",
    )
    args = parser.parse_args()

    # Determine SI values strictly
    # No default optical scale: take it from --pixel-size or from a calibration file.
    if str(Path(__file__).resolve().parents[1]) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from image_scripts.calibrate_pixel_size import resolve_pixel_size

    explicit_m = args.pixel_size
    if explicit_m is None and args.pixel_size_mm is not None:
        explicit_m = args.pixel_size_mm * 1e-3
    pixel_size_m = resolve_pixel_size(explicit_m, args.mask)
    dia_m = args.diameter if args.diameter is not None else args.diameter_mm * 1e-3

    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask at: {args.mask}")

    print("Reconstructing 3D multi-phase flow geometry in SI units...")
    print(f"  Pixel size (SI) : {pixel_size_m:.6e} m/px ({pixel_size_m * 1e3:.4f} mm/px)")
    print(f"  Diameter (SI)   : {dia_m:.6e} m ({dia_m * 1e3:.2f} mm)")
    print(f"  Smoothing Lambda: {args.smoothing_lambda:.4f} px")

    recon_data = reconstruct_3d_bubbles_si(
        mask, pixel_size_m, dia_m, smoothing_lambda_px=args.smoothing_lambda
    )

    print(f"  Reconstructed Bubbles: {recon_data['bubble_count']}")
    print(f"  Gas Volume (SI)      : {recon_data['gas_volume_m3']:.6e} m³")
    print(f"  Pipe Volume (SI)     : {recon_data['pipe_volume_m3']:.6e} m³")
    print(
        f"  Void Fraction (alpha): {recon_data['void_fraction_alpha']:.4f} ({recon_data['void_fraction_alpha'] * 100.0:.3f}%)"
    )

    rendered_path = render_3d_flow_visualization(recon_data, args.output)
    print(f"[DONE] Rendered 3D visualization saved to: {rendered_path}")

    if args.interactive_html:
        html_path = render_interactive_html(recon_data, args.interactive_html)
        print(f"[DONE] Interactive 3D HTML saved to: {html_path}")


if __name__ == "__main__":
    main()
