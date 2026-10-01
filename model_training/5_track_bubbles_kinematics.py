#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def extract_bubble_blobs(mask_2d: np.ndarray, pixel_size_m: float = 6.7e-5) -> list[dict[str, Any]]:
    """
    Identifies 8-connected bubble components and extracts their spatial properties
    in both image pixel coordinates and standard SI units (meters, square meters):
      - ID
      - Centroid (x_m, y_m) and (x_px, y_px)
      - Area (m^2) and (px)
      - Bounding box (x, y, w, h) in px and meters
    """
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask_2d, connectivity=8)
    blobs = []
    for i in range(1, num_labels):
        area_px = stats[i, cv2.CC_STAT_AREA]
        if area_px < 5:  # Ignore microscopic noise specks (< 5 pixels)
            continue
        bx_px, by_px, bw_px, bh_px = (
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
        )
        c_x_px, c_y_px = float(centroids[i][0]), float(centroids[i][1])

        # Convert to SI units (meters, m^2)
        c_x_m = c_x_px * pixel_size_m
        c_y_m = c_y_px * pixel_size_m
        area_m2 = float(area_px * (pixel_size_m**2))

        blobs.append(
            {
                "id": i,
                "centroid_px": (c_x_px, c_y_px),
                "centroid_m": (c_x_m, c_y_m),
                "area_px": float(area_px),
                "area_m2": area_m2,
                "bbox_px": (bx_px, by_px, bw_px, bh_px),
                "bbox_m": (
                    bx_px * pixel_size_m,
                    by_px * pixel_size_m,
                    bw_px * pixel_size_m,
                    bh_px * pixel_size_m,
                ),
            }
        )
    return blobs


def run_tracking(
    mask_dir: Path,
    output_dir: Path,
    fps: float,
    pixel_size: float,
    max_disp_px: float,
    image_dir: Path | None = None,
    render_animation: bool = False,
    playback_fps: float = 5.0,
    vector_step_px: float = 12.0,
):
    """
    Performs nearest-neighbor kinematic tracking and hydrodynamic breakup detection across consecutive frames.
    All physical outputs are calculated and saved in standard SI units (m/s, m, m^2, s).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_files = sorted(mask_dir.glob("*.png"))
    if not mask_files:
        raise FileNotFoundError(f"No mask files found in {mask_dir}")

    # Standardize pixel size to SI (meters/pixel)
    pixel_size_m = pixel_size * 1e-3 if pixel_size > 0.001 else pixel_size

    # Inter-frame temporal resolution in SI seconds (e.g. 0.0005 s at 2000 FPS)
    dt_s = 1.0 / fps
    max_disp_m = max_disp_px * pixel_size_m

    velocity_records = []
    breakup_events = []

    print(f"Tracking bubble kinematics at {fps} FPS across {len(mask_files)} frames in SI units...")
    print(f"  Temporal resolution (dt)  : {dt_s:.6e} s ({dt_s * 1e3:.3f} ms)")
    print(f"  Pixel size (s_px)         : {pixel_size_m:.6e} m/px")
    print(f"  Max search displacement   : {max_disp_m:.6e} m ({max_disp_px:.1f} px)")

    for t in range(len(mask_files) - 1):
        mask_t = cv2.imread(str(mask_files[t]), cv2.IMREAD_GRAYSCALE)
        mask_t1 = cv2.imread(str(mask_files[t + 1]), cv2.IMREAD_GRAYSCALE)

        if mask_t is None or mask_t1 is None:
            continue

        blobs_t = extract_bubble_blobs(mask_t, pixel_size_m=pixel_size_m)
        blobs_t1 = extract_bubble_blobs(mask_t1, pixel_size_m=pixel_size_m)

        if not blobs_t or not blobs_t1:
            continue

        # Coalescence Guardrail (Phase 3 Rule):
        # Frame transitions with unequal bubble counts indicate ongoing breakup, merging, or boundary entry/exit.
        is_stable_pair = len(blobs_t) == len(blobs_t1)

        pts_t = np.array([b["centroid_px"] for b in blobs_t])
        pts_t1 = np.array([b["centroid_px"] for b in blobs_t1])

        # Pairwise Euclidean distance matrix in pixels
        dist_matrix_px = cdist(pts_t, pts_t1)

        # Estimate global mean translation displacement for frame t -> t+1
        # using preliminary mutual nearest neighbors within max_disp_px
        disp_estimates_px = []
        for idx_t, _b_t in enumerate(blobs_t):
            best_col = int(np.argmin(dist_matrix_px[idx_t]))
            if (
                dist_matrix_px[idx_t, best_col] <= max_disp_px
                and int(np.argmin(dist_matrix_px[:, best_col])) == idx_t
            ):
                disp_estimates_px.append(
                    (
                        pts_t1[best_col, 0] - pts_t[idx_t, 0],
                        pts_t1[best_col, 1] - pts_t[idx_t, 1],
                    )
                )
        if disp_estimates_px:
            mean_dx_px = float(np.mean([d[0] for d in disp_estimates_px]))
            mean_dy_px = float(np.mean([d[1] for d in disp_estimates_px]))
        else:
            mean_dx_px, mean_dy_px = 0.0, 0.0

        # Detect breakup events first with unique child assignment
        breakup_parent_indices = set()
        claimed_children_indices = set()

        for idx_t, b_t in enumerate(blobs_t):
            bx, by, bw, bh = b_t["bbox_px"]
            margin = 5.0
            shifted_x_min = bx + mean_dx_px - margin
            shifted_x_max = bx + bw + mean_dx_px + margin
            shifted_y_min = by + mean_dy_px - margin
            shifted_y_max = by + bh + mean_dy_px + margin

            candidate_children = [
                (idx_c, c)
                for idx_c, c in enumerate(blobs_t1)
                if idx_c not in claimed_children_indices
                and (shifted_x_min <= c["centroid_px"][0] <= shifted_x_max)
                and (shifted_y_min <= c["centroid_px"][1] <= shifted_y_max)
            ]

            if len(candidate_children) >= 2:
                total_child_area_m2 = sum(c["area_m2"] for _, c in candidate_children)
                parent_area_m2 = b_t["area_m2"]
                area_error = abs(total_child_area_m2 - parent_area_m2) / parent_area_m2

                if area_error < 0.35:
                    breakup_parent_indices.add(idx_t)
                    for idx_c, _ in candidate_children:
                        claimed_children_indices.add(idx_c)

                    breakup_events.append(
                        {
                            "frame_t": mask_files[t].stem,
                            "frame_t1": mask_files[t + 1].stem,
                            "parent_id": b_t["id"],
                            "parent_area_m2": parent_area_m2,
                            "total_child_area_m2": total_child_area_m2,
                            "area_error_rel": area_error,
                            "num_children": len(candidate_children),
                            "parent_dx_m": mean_dx_px * pixel_size_m,
                            "parent_dy_m": mean_dy_px * pixel_size_m,
                        }
                    )

        # Optimal 1-to-1 bipartite assignment for non-breaking bubbles
        cost_matrix = np.copy(dist_matrix_px)
        large_cost = max_disp_px * 1e4
        for p_idx in breakup_parent_indices:
            cost_matrix[p_idx, :] = large_cost
        for c_idx in claimed_children_indices:
            cost_matrix[:, c_idx] = large_cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            if r in breakup_parent_indices or c in claimed_children_indices:
                continue
            min_dist_px = float(dist_matrix_px[r, c])
            if min_dist_px <= max_disp_px:
                b_t = blobs_t[r]
                dx_px = pts_t1[c, 0] - pts_t[r, 0]
                dy_px = pts_t1[c, 1] - pts_t[r, 1]
                disp_m = min_dist_px * pixel_size_m
                dx_m = dx_px * pixel_size_m
                dy_m = dy_px * pixel_size_m
                vel_m_s = disp_m / dt_s

                velocity_records.append(
                    {
                        "frame_t": mask_files[t].stem,
                        "frame_t1": mask_files[t + 1].stem,
                        "bubble_id": b_t["id"],
                        "velocity_m_s": vel_m_s,
                        "displacement_m": disp_m,
                        "dx_m": dx_m,
                        "dy_m": dy_m,
                        "area_m2": b_t["area_m2"],
                        "displacement_px": min_dist_px,
                        "is_stable_count_pair": is_stable_pair,
                    }
                )

    vel_path = output_dir / "bubble_velocities_si.csv"
    breakup_path = output_dir / "bubble_breakups_si.csv"

    df_vel = pd.DataFrame(velocity_records)
    df_break = pd.DataFrame(breakup_events)

    df_vel.to_csv(vel_path, index=False)
    df_break.to_csv(breakup_path, index=False)

    print("\n" + "=" * 52)
    print("     KINEMATICS & BREAKUP ANALYSIS SUMMARY (SI)     ")
    print("=" * 52)
    if not df_vel.empty:
        stable_vels = df_vel[df_vel["is_stable_count_pair"]]
        if not stable_vels.empty:
            print(f"Total Trajectory Links (Stable Pairs): {len(stable_vels)}")
            print(
                f"Mean Velocity (Stable Pairs)        : {stable_vels['velocity_m_s'].mean():.4f} m/s"
            )
            print(
                f"Max Velocity (Stable Pairs)         : {stable_vels['velocity_m_s'].max():.4f} m/s"
            )
        else:
            print(f"Total Trajectory Links (All)        : {len(df_vel)}")
            print(f"Mean Velocity (All)                 : {df_vel['velocity_m_s'].mean():.4f} m/s")
        print(f"Mean Bubble Displacement            : {df_vel['displacement_m'].mean():.6e} m")
    print(f"Total Breakup Events Detected       : {len(df_break)}")
    print("=" * 52)
    print(f"[DONE] Saved velocity logs (SI) to: {vel_path}")
    print(f"[DONE] Saved breakup logs (SI) to : {breakup_path}")

    if render_animation:
        from model_training.visualize_flow_animation import render_flow_animation

        anim_out = output_dir / "animation"
        print(f"\n[ANIMATION] Generating flow animation & velocity fields in: {anim_out}")
        render_flow_animation(
            image_dir=image_dir,
            mask_dir=mask_dir,
            output_dir=anim_out,
            fps=fps,
            pixel_size_m=pixel_size_m,
            max_disp_px=max_disp_px,
            vector_step_px=vector_step_px,
            playback_fps=playback_fps,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="2000 FPS Bubble Kinematics and Breakup Tracking in SI Units"
    )
    parser.add_argument(
        "--masks",
        "-m",
        type=Path,
        required=True,
        help="Input folder with binary masks",
    )
    parser.add_argument(
        "--images",
        "-i",
        type=Path,
        default=None,
        help="Optional input folder with raw images for overlay",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("results/kinematics"),
        help="Output folder for CSV results",
    )
    parser.add_argument("--fps", type=float, default=2000.0, help="Camera frame rate in 1/s (Hz)")
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=8.6e-5,
        help="Pixel size in meters/pixel (SI, default 8.6e-5 m/px = 0.086 mm/px)",
    )
    parser.add_argument(
        "--max-disp",
        type=float,
        default=50.0,
        help="Max centroid search radius in px",
    )
    parser.add_argument(
        "--render-animation",
        action="store_true",
        help="Render slow-motion MP4 animation, PNG figures, and interactive HTML viewer",
    )
    parser.add_argument(
        "--playback-fps",
        type=float,
        default=5.0,
        help="Playback framerate for generated MP4 animation (default: 5.0)",
    )
    parser.add_argument(
        "--vector-step",
        type=float,
        default=12.0,
        help="Contour sampling interval in pixels for boundary vector arrows (default: 12.0)",
    )
    args = parser.parse_args()

    run_tracking(
        mask_dir=args.masks,
        output_dir=args.output_dir,
        fps=args.fps,
        pixel_size=args.pixel_size,
        max_disp_px=args.max_disp,
        image_dir=args.images,
        render_animation=args.render_animation,
        playback_fps=args.playback_fps,
        vector_step_px=args.vector_step,
    )
