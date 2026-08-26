import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def extract_bubble_blobs(mask_2d: np.ndarray):
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask_2d, connectivity=8
    )
    blobs = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 5:
            continue
        x, y, w, h = (
            stats[i, cv2.CC_STAT_LEFT],
            stats[i, cv2.CC_STAT_TOP],
            stats[i, cv2.CC_STAT_WIDTH],
            stats[i, cv2.CC_STAT_HEIGHT],
        )
        blobs.append(
            {
                "id": i,
                "centroid": centroids[i],
                "area": area,
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
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_files = sorted(list(mask_dir.glob("*.png")))
    dt = 1.0 / fps

    velocity_records = []
    breakup_events = []

    print(f"Tracking kinematics at {fps} FPS across {len(mask_files)} frames...")

    for t in range(len(mask_files) - 1):
        mask_t = cv2.imread(str(mask_files[t]), cv2.IMREAD_GRAYSCALE)
        mask_t1 = cv2.imread(str(mask_files[t + 1]), cv2.IMREAD_GRAYSCALE)

        blobs_t = extract_bubble_blobs(mask_t)
        blobs_t1 = extract_bubble_blobs(mask_t1)

        if not blobs_t or not blobs_t1:
            continue

        pts_t = np.array([b["centroid"] for b in blobs_t])
        pts_t1 = np.array([b["centroid"] for b in blobs_t1])

        dist_matrix = cdist(pts_t, pts_t1)

        for idx_t, b_t in enumerate(blobs_t):
            best_match_idx = np.argmin(dist_matrix[idx_t])
            min_dist_px = dist_matrix[idx_t, best_match_idx]

            if min_dist_px <= max_disp_px:
                disp_mm = min_dist_px * pixel_size_mm
                vel_m_s = (disp_mm * 1e-3) / dt

                velocity_records.append(
                    {
                        "frame_t": mask_files[t].stem,
                        "bubble_id": b_t["id"],
                        "displacement_px": min_dist_px,
                        "velocity_m_s": vel_m_s,
                    }
                )

        # Breakup detection: child blobs in t+1 within parent bbox in t
        for b_t in blobs_t:
            bx, by, bw, bh = b_t["bbox"]
            children = [
                c
                for c in blobs_t1
                if (bx <= c["centroid"][0] <= bx + bw)
                and (by <= c["centroid"][1] <= by + bh)
            ]

            if len(children) >= 2:
                total_child_area = sum(c["area"] for c in children)
                if abs(total_child_area - b_t["area"]) / b_t["area"] < 0.35:
                    breakup_events.append(
                        {
                            "frame": mask_files[t].stem,
                            "parent_id": b_t["id"],
                            "parent_area": b_t["area"],
                            "num_children": len(children),
                        }
                    )

    vel_path = output_dir / "bubble_velocities.csv"
    breakup_path = output_dir / "bubble_breakups.csv"

    pd.DataFrame(velocity_records).to_csv(vel_path, index=False)
    pd.DataFrame(breakup_events).to_csv(breakup_path, index=False)

    print(f"[DONE] Saved tracking logs to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2000 FPS Bubble Kinematics")
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
        default=Path("tracking_results"),
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
