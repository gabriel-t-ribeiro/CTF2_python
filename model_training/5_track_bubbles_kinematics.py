import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def extract_bubble_blobs(mask_2d: np.ndarray) -> list[dict]:
    """
    Identifies 8-connected bubble components and extracts their spatial properties:
    ID, centroid (x, y), area (px), and bounding box (x, y, w, h).
    """
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask_2d, connectivity=8
    )
    blobs = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 5:  # Ignore microscopic noise specks (< 5 pixels)
            continue
        x, y, w, h = (
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
        )
        blobs.append(
            {
                "id": i,
                "centroid": centroids[i],
                "area": float(area),
                "bbox": (x, y, w, h),
            }
        )
    return blobs


def run_tracking(
    mask_dir: Path,
    output_dir: Path,
    fps: float,
    pixel_size_mm: float,
    max_disp_px: float,
):
    """
    Performs nearest-neighbor kinematic tracking and hydrodynamic breakup detection across consecutive frames.
    Implements:
      1. Nearest-neighbor translation velocity vector calculation.
      2. Coalescence guardrail (stable bubble count filtering) to avoid tracking artifacts.
      3. Dynamic bounding box shift using the translation vector for accurate breakup detection in moving flows.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_files = sorted(list(mask_dir.glob("*.png")))
    if not mask_files:
        raise FileNotFoundError(f"No mask files found in {mask_dir}")

    dt = 1.0 / fps  # Temporal resolution (e.g. 0.0005 s at 2000 FPS)

    velocity_records = []
    breakup_events = []

    print(f"Tracking kinematics at {fps} FPS across {len(mask_files)} frames...")

    for t in range(len(mask_files) - 1):
        mask_t = cv2.imread(str(mask_files[t]), cv2.IMREAD_GRAYSCALE)
        mask_t1 = cv2.imread(str(mask_files[t + 1]), cv2.IMREAD_GRAYSCALE)

        if mask_t is None or mask_t1 is None:
            continue

        blobs_t = extract_bubble_blobs(mask_t)
        blobs_t1 = extract_bubble_blobs(mask_t1)

        if not blobs_t or not blobs_t1:
            continue

        # Coalescence Guardrail (Phase 3 Rule):
        # Frame transitions with unequal bubble counts indicate ongoing breakup, merging, or boundary entry/exit.
        is_stable_pair = len(blobs_t) == len(blobs_t1)

        pts_t = np.array([b["centroid"] for b in blobs_t])
        pts_t1 = np.array([b["centroid"] for b in blobs_t1])

        # Pairwise Euclidean distance matrix (N_t x N_t+1)
        dist_matrix = cdist(pts_t, pts_t1)

        # Store translation displacement vectors for parent blobs
        disp_vectors = {}
        valid_displacements = []

        for idx_t, b_t in enumerate(blobs_t):
            best_match_idx = int(np.argmin(dist_matrix[idx_t]))
            min_dist_px = float(dist_matrix[idx_t, best_match_idx])

            if min_dist_px <= max_disp_px:
                # Displacement vector (dx, dy) from t to t+1
                dx = pts_t1[best_match_idx, 0] - pts_t[idx_t, 0]
                dy = pts_t1[best_match_idx, 1] - pts_t[idx_t, 1]
                disp_vectors[idx_t] = (dx, dy)
                valid_displacements.append((dx, dy))

                disp_mm = min_dist_px * pixel_size_mm
                vel_m_s = (disp_mm * 1e-3) / dt

                velocity_records.append(
                    {
                        "frame_t": mask_files[t].stem,
                        "frame_t1": mask_files[t + 1].stem,
                        "bubble_id": b_t["id"],
                        "displacement_px": min_dist_px,
                        "dx_px": dx,
                        "dy_px": dy,
                        "velocity_m_s": vel_m_s,
                        "is_stable_count_pair": is_stable_pair,
                    }
                )

        # Calculate average frame translation vector (fallback for splitting bubbles)
        if valid_displacements:
            mean_dx = float(np.mean([d[0] for d in valid_displacements]))
            mean_dy = float(np.mean([d[1] for d in valid_displacements]))
        else:
            mean_dx, mean_dy = 0.0, 0.0

        # Breakup Detection: Shift parent bounding box by the translation vector
        for idx_t, b_t in enumerate(blobs_t):
            bx, by, bw, bh = b_t["bbox"]

            # Use bubble's own translation if available, otherwise use frame mean flow displacement
            dx, dy = disp_vectors.get(idx_t, (mean_dx, mean_dy))

            # Shift bounding box by translation vector and add a 5px deformation margin
            margin = 5.0
            shifted_x_min = bx + dx - margin
            shifted_x_max = bx + bw + dx + margin
            shifted_y_min = by + dy - margin
            shifted_y_max = by + bh + dy + margin

            # Find child candidates in frame t+1 within shifted parent envelope
            children = [
                c
                for c in blobs_t1
                if (shifted_x_min <= c["centroid"][0] <= shifted_x_max)
                and (shifted_y_min <= c["centroid"][1] <= shifted_y_max)
            ]

            # Breakup condition: at least 2 children and area conservation within 35%
            if len(children) >= 2:
                total_child_area = sum(c["area"] for c in children)
                area_error = abs(total_child_area - b_t["area"]) / b_t["area"]
                if area_error < 0.35:
                    breakup_events.append(
                        {
                            "frame_t": mask_files[t].stem,
                            "frame_t1": mask_files[t + 1].stem,
                            "parent_id": b_t["id"],
                            "parent_area_px": b_t["area"],
                            "total_child_area_px": total_child_area,
                            "area_error_rel": area_error,
                            "num_children": len(children),
                            "parent_dx_px": dx,
                            "parent_dy_px": dy,
                        }
                    )

    vel_path = output_dir / "bubble_velocities.csv"
    breakup_path = output_dir / "bubble_breakups.csv"

    df_vel = pd.DataFrame(velocity_records)
    df_break = pd.DataFrame(breakup_events)

    df_vel.to_csv(vel_path, index=False)
    df_break.to_csv(breakup_path, index=False)

    print("\n" + "=" * 50)
    print("        KINEMATICS & BREAKUP ANALYSIS SUMMARY        ")
    print("=" * 50)
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
            print(
                f"Mean Velocity (All)                 : {df_vel['velocity_m_s'].mean():.4f} m/s"
            )
    print(f"Total Breakup Events Detected       : {len(df_break)}")
    print("=" * 50)
    print(f"[DONE] Saved velocity logs to: {vel_path}")
    print(f"[DONE] Saved breakup logs to : {breakup_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="2000 FPS Bubble Kinematics and Breakup Tracking"
    )
    parser.add_argument(
        "--masks",
        "-m",
        type=Path,
        required=True,
        help="Input folder with binary masks",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("results/kinematics"),
        help="Output folder for CSV results",
    )
    parser.add_argument("--fps", type=float, default=2000.0, help="Camera frame rate")
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=0.067,
        help="Pixel size in mm/pixel",
    )
    parser.add_argument(
        "--max-disp",
        type=float,
        default=50.0,
        help="Max centroid search radius in px",
    )
    args = parser.parse_args()

    run_tracking(args.masks, args.output_dir, args.fps, args.pixel_size, args.max_disp)
