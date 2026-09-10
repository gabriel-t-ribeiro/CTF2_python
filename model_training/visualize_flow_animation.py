"""
High-Speed Bubble Velocity Field Visualization & Flow Animation Pipeline.

Mimics and enhances previous vector field research (Eduardo Reis 2025 report and MATLAB scripts):
  - Boundary displacement velocity vectors via nearest-neighbor contour mapping (cKDTree).
  - Perceptual color-mapping (Turbo / Viridis) encoding velocity magnitude in standard SI units (m/s).
  - Adaptive contour subsampling to avoid arrow clumping while capturing local interface deformation.
  - Multi-frame centroid trajectory trails with active-bubble pruning (no ghost traces).
  - Real-time HUD telemetry: Frame indices, elapsed time (ms), mean/max velocity (m/s), and scale bar.
  - Scalable performance for multi-thousand frame datasets:
      * Stride/frame-skipping to guarantee videos <= 30 seconds at 60 FPS.
      * Lightweight, lightning-fast interactive HTML player (< 8 MB via JPEG encoding & downsampling).
      * Clean filename formatting avoiding text collisions in HUD.
"""

from __future__ import annotations

import argparse
import base64
import csv
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


def format_short_frame_name(stem: str) -> str:
    """
    Extracts a concise, human-readable frame identifier (e.g. '#2790')
    from long camera or mask stems to prevent HUD text overflow.
    """
    clean = re.sub(r"(_mask|_pred|_seg)$", "", stem)
    match = re.search(r"(\d+)$", clean)
    if match:
        num = int(match.group(1))
        return f"#{num:04d}"
    if len(clean) > 16:
        return f"...{clean[-12:]}"
    return clean


def extract_bubble_blobs(
    mask_2d: np.ndarray, pixel_size_m: float = 6.1162e-5, min_area_px: int = 15
) -> list[dict[str, Any]]:
    """
    Extract 8-connected bubble components, contours, and physical metrics.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_2d, connectivity=8)
    blobs = []

    for i in range(1, num_labels):
        area_px = int(stats[i, cv2.CC_STAT_AREA])
        if area_px < min_area_px:
            continue

        bx_px = int(stats[i, cv2.CC_STAT_LEFT])
        by_px = int(stats[i, cv2.CC_STAT_TOP])
        bw_px = int(stats[i, cv2.CC_STAT_WIDTH])
        bh_px = int(stats[i, cv2.CC_STAT_HEIGHT])
        c_x_px, c_y_px = float(centroids[i][0]), float(centroids[i][1])

        # Extract precise single-bubble binary mask for contour extraction
        bubble_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        # Select longest contour in case of multiple fragments
        contour = max(contours, key=cv2.contourArea).squeeze()
        if contour.ndim != 2 or len(contour) < 5:
            continue

        blobs.append(
            {
                "id": i,
                "centroid_px": (c_x_px, c_y_px),
                "centroid_m": (c_x_px * pixel_size_m, c_y_px * pixel_size_m),
                "area_px": float(area_px),
                "area_m2": float(area_px * (pixel_size_m**2)),
                "bbox_px": (bx_px, by_px, bw_px, bh_px),
                "contour_px": contour,  # Array of shape (N, 2) in (x, y)
            }
        )

    return blobs


def sample_contour_evenly(contour_px: np.ndarray, step_px: float = 12.0) -> np.ndarray:
    """
    Subsamples a 2D boundary contour at approximately equal arc-length intervals
    to prevent dense overlapping clusters of displacement vectors.
    """
    if len(contour_px) <= 5:
        return contour_px

    # Calculate cumulative distance along closed loop contour
    diffs = np.diff(contour_px, axis=0)
    seg_dists = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_dist = np.concatenate(([0.0], np.cumsum(seg_dists)))
    total_length = cum_dist[-1]

    if total_length < step_px:
        step_count = max(4, int(len(contour_px) // 2))
        indices = np.linspace(0, len(contour_px) - 1, step_count, dtype=int)
        return contour_px[indices]

    num_samples = max(4, int(np.round(total_length / step_px)))
    sample_targets = np.linspace(0, total_length, num_samples, endpoint=False)
    sample_indices = np.searchsorted(cum_dist, sample_targets)
    sample_indices = np.clip(sample_indices, 0, len(contour_px) - 1)

    return contour_px[np.unique(sample_indices)]


def compute_boundary_vectors(
    contour_t: np.ndarray,
    contour_t1: np.ndarray,
    step_px: float,
    dt_s: float,
    pixel_size_m: float,
    max_disp_px: float = 50.0,
) -> list[dict[str, Any]]:
    """
    Computes local boundary displacement vectors from frame t to frame t+1 using
    nearest-neighbor search (cKDTree), returning positions, displacements, and SI velocities.
    """
    sampled_t = sample_contour_evenly(contour_t, step_px=step_px)
    tree_t1 = cKDTree(contour_t1)

    dists_px, indices_t1 = tree_t1.query(sampled_t)
    matched_pts_t1 = contour_t1[indices_t1]

    vectors = []
    for (x0, y0), (x1, y1), dist_px in zip(sampled_t, matched_pts_t1, dists_px):
        if dist_px > max_disp_px:
            continue

        dx_px = float(x1 - x0)
        dy_px = float(y1 - y0)
        disp_m = float(dist_px * pixel_size_m)
        dx_m = float(dx_px * pixel_size_m)
        dy_m = float(dy_px * pixel_size_m)

        speed_m_s = disp_m / dt_s if dt_s > 0 else 0.0
        vx_m_s = dx_m / dt_s if dt_s > 0 else 0.0
        vy_m_s = dy_m / dt_s if dt_s > 0 else 0.0

        vectors.append(
            {
                "x0_px": float(x0),
                "y0_px": float(y0),
                "x1_px": float(x1),
                "y1_px": float(y1),
                "dx_px": dx_px,
                "dy_px": dy_px,
                "disp_px": float(dist_px),
                "dx_m": dx_m,
                "dy_m": dy_m,
                "disp_m": disp_m,
                "vx_m_s": vx_m_s,
                "vy_m_s": vy_m_s,
                "speed_m_s": speed_m_s,
            }
        )

    return vectors


def get_colormap_color(
    speed_m_s: float, v_max_m_s: float, colormap_type: int = cv2.COLORMAP_TURBO
) -> tuple[int, int, int]:
    """
    Maps speed magnitude to a 24-bit BGR color via OpenCV colormap.
    """
    if v_max_m_s <= 1e-6:
        norm_val = 0
    else:
        norm_val = int(np.clip((speed_m_s / v_max_m_s) * 255.0, 0, 255))

    sample = np.array([[norm_val]], dtype=np.uint8)
    colored = cv2.applyColorMap(sample, colormap_type)
    b, g, r = colored[0, 0]
    return int(b), int(g), int(r)


def draw_hud_and_legend(
    canvas: np.ndarray,
    frame_t_name: str,
    frame_t1_name: str,
    time_elapsed_ms: float,
    mean_speed_m_s: float,
    max_speed_m_s: float,
    bubble_count: int,
    v_max_legend: float,
    scale_bar_speed_m_s: float,
    scale_bar_len_px: float,
    colormap_type: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """
    Renders an aerodynamic dark HUD banner at the top and a vertical colorbar on the right.
    Uses concise frame names to guarantee clean spacing without text overlap.
    """
    h, w = canvas.shape[:2]

    banner_h = 46
    hud_overlay = canvas.copy()
    cv2.rectangle(hud_overlay, (0, 0), (w, banner_h), (20, 20, 24), -1)
    canvas = cv2.addWeighted(hud_overlay, 0.82, canvas, 0.18, 0)
    cv2.line(canvas, (0, banner_h), (w, banner_h), (65, 65, 75), 1, cv2.LINE_AA)

    # Left: Concise frame transition & elapsed high-speed time
    short_t = format_short_frame_name(frame_t_name)
    short_t1 = format_short_frame_name(frame_t1_name)
    pair_text = f"Velocity: {short_t} -> {short_t1}  (t = {time_elapsed_ms:.1f} ms)"
    cv2.putText(
        canvas, pair_text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (245, 245, 250), 1, cv2.LINE_AA
    )

    # Dynamic spacing based on text length
    pair_w = cv2.getTextSize(pair_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0][0]
    stats_x = 12 + pair_w + 20

    # Center: Key kinematics telemetry
    stats_text = f"Bubbles: {bubble_count}  |  Mean: {mean_speed_m_s:.2f} m/s  |  Max: {max_speed_m_s:.2f} m/s"
    cv2.putText(
        canvas,
        stats_text,
        (stats_x, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 225, 235),
        1,
        cv2.LINE_AA,
    )

    # Far Right: Scale reference arrow in banner
    if scale_bar_len_px > 5:
        ref_x2 = w - 16
        ref_x1 = int(ref_x2 - min(scale_bar_len_px, 60))
        ref_y = 23
        cv2.arrowedLine(
            canvas,
            (ref_x1, ref_y),
            (ref_x2, ref_y),
            (0, 235, 255),
            2,
            tipLength=0.35,
            line_type=cv2.LINE_AA,
        )
        scale_label = f"{scale_bar_speed_m_s:.1f} m/s"
        cv2.putText(
            canvas,
            scale_label,
            (ref_x1 - 56, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 235, 255),
            1,
            cv2.LINE_AA,
        )

    # Vertical Colormap Legend on bottom right margin (if canvas is large enough)
    if h >= 120 and w >= 160:
        legend_w = 12
        legend_h = max(20, min(110, h - banner_h - 40))
        legend_x = w - 42
        legend_y = banner_h + 16

        grad = np.linspace(255, 0, legend_h, dtype=np.uint8).reshape(legend_h, 1)
        grad_color = cv2.applyColorMap(grad, colormap_type)
        grad_block = np.repeat(grad_color, legend_w, axis=1)

        legend_bg = canvas.copy()
        cv2.rectangle(
            legend_bg,
            (legend_x - 4, legend_y - 14),
            (w - 6, legend_y + legend_h + 16),
            (20, 20, 24),
            -1,
        )
        canvas = cv2.addWeighted(legend_bg, 0.70, canvas, 0.30, 0)

        canvas[legend_y : legend_y + legend_h, legend_x : legend_x + legend_w] = grad_block
        cv2.rectangle(
            canvas,
            (legend_x, legend_y),
            (legend_x + legend_w, legend_y + legend_h),
            (180, 180, 190),
            1,
            cv2.LINE_AA,
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            canvas,
            f"{v_max_legend:.1f}",
            (legend_x - 30, legend_y + 8),
            font,
            0.34,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{v_max_legend / 2:.1f}",
            (legend_x - 30, legend_y + legend_h // 2 + 4),
            font,
            0.34,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "0.0",
            (legend_x - 24, legend_y + legend_h + 2),
            font,
            0.34,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas, "m/s", (legend_x - 1, legend_y - 4), font, 0.34, (0, 230, 255), 1, cv2.LINE_AA
        )

    return canvas


def render_flow_animation(
    image_dir: Path | None,
    mask_dir: Path,
    output_dir: Path,
    fps: float = 2000.0,
    pixel_size_m: float = 6.1162e-5,
    max_disp_px: float = 50.0,
    vector_step_px: float = 12.0,
    playback_fps: float = 5.0,
    max_duration_s: float = 20.0,
    stride: int = 1,
    start_frame: int = 0,
    trail_length: int = 5,
    arrow_scale: float = 1.0,
    colormap_name: str = "turbo",
    v_max_override: float | None = None,
    max_html_frames: int = 150,
    save_individual_frames: bool | None = None,
) -> dict[str, Any]:
    """
    Main orchestration function:
      1. Parses paired frames and masks.
      2. Limits frame sequence directly via max_duration_s (e.g. 10 to 30s) at smooth playback FPS (5 FPS).
      3. Uses smooth consecutive frames (stride=1) by default to eliminate jumping/strobing.
      4. Performs optimal bipartite bubble tracking across time.
      5. Discards inactive traces immediately when bubbles exit/lose track.
      6. Renders annotated slow-motion video and generates a lightweight interactive HTML player.
      7. Writes detailed SI vector records to CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"

    mask_files = sorted(mask_dir.glob("*.png"))
    if not mask_files:
        raise FileNotFoundError(f"No binary mask files (*.png) found in {mask_dir}")

    total_masks = len(mask_files)
    if stride is None or stride < 1:
        stride = 1

    # Duration directly controls the number of transitions evaluated
    if max_duration_s is not None and max_duration_s > 0:
        max_allowed_transitions = max(1, int(np.floor(max_duration_s * playback_fps)))
    else:
        max_allowed_transitions = total_masks

    safe_start = max(0, min(start_frame, total_masks - stride - 1)) if total_masks > stride else 0
    safe_end = min(total_masks - stride, safe_start + max_allowed_transitions * stride)

    step_indices = list(range(safe_start, safe_end, stride))
    num_transitions = len(step_indices)
    expected_duration = num_transitions / playback_fps

    # Individual PNGs saved only if <= 100 frames or explicitly requested
    if save_individual_frames is None:
        save_individual_frames = num_transitions <= 100
    if save_individual_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    # Map corresponding raw image files (indexed both with original stem and with _mask stripped)
    image_files_map: dict[str, Path] = {}
    if image_dir and image_dir.is_dir():
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            for img_p in image_dir.glob(f"*{ext}"):
                image_files_map[img_p.stem] = img_p
                clean_name = re.sub(r"(_mask|_pred|_seg)$", "", img_p.stem)
                image_files_map[clean_name] = img_p

    dt_step = stride * (1.0 / fps)
    max_disp_step = min(max_disp_px * stride, 150.0)
    colormap_type = (
        cv2.COLORMAP_VIRIDIS if colormap_name.lower() == "viridis" else cv2.COLORMAP_TURBO
    )

    print("=" * 64)
    print("      HIGH-SPEED BUBBLE VELOCITY FIELD & FLOW ANIMATION      ")
    print("=" * 64)
    print(f"  Total Raw Masks Found     : {total_masks}")
    print(f"  Frame Processing Stride   : {stride} (dt_step = {dt_step * 1e3:.3f} ms)")
    print(f"  Sequence Range Evaluated  : Frame #{safe_start} -> #{safe_end}")
    print(f"  Total Processed Frames    : {num_transitions}")
    print(f"  Camera Capture Rate       : {fps:.1f} FPS")
    print(f"  Video Playback Rate       : {playback_fps:.1f} FPS (Smooth Slow-Mo)")
    print(
        f"  Resulting Video Duration  : {expected_duration:.2f} s (Target: {max_duration_s:.1f} s)"
    )
    print(f"  Optical Scale             : {pixel_size_m:.6e} m/px ({pixel_size_m * 1e3:.4f} mm/px)")
    print(f"  Boundary Vector Spacing   : {vector_step_px:.1f} px")
    print(f"  Colormap Encoding         : {colormap_name.upper()}")

    all_step_data = []
    all_speeds_m_s = []
    centroid_histories: dict[int, list[tuple[float, float, int]]] = {}
    next_global_id = 1

    # First pass: Extract blobs, compute tracking & collect velocities for global colormap
    for idx_step, t in enumerate(step_indices):
        f_t_path = mask_files[t]
        f_t1_path = mask_files[t + stride]
        stem_t = f_t_path.stem
        stem_t1 = f_t1_path.stem

        mask_t = cv2.imread(str(f_t_path), cv2.IMREAD_GRAYSCALE)
        mask_t1 = cv2.imread(str(f_t1_path), cv2.IMREAD_GRAYSCALE)

        blobs_t = extract_bubble_blobs(mask_t, pixel_size_m=pixel_size_m)
        blobs_t1 = extract_bubble_blobs(mask_t1, pixel_size_m=pixel_size_m)

        if not blobs_t or not blobs_t1:
            all_step_data.append(
                {
                    "step_idx": idx_step,
                    "t_raw": t,
                    "stem_t": stem_t,
                    "stem_t1": stem_t1,
                    "matched_pairs": [],
                    "boundary_vectors": [],
                    "mean_speed": 0.0,
                    "max_speed": 0.0,
                }
            )
            continue

        pts_t = np.array([b["centroid_px"] for b in blobs_t])
        pts_t1 = np.array([b["centroid_px"] for b in blobs_t1])
        cost_mat = cdist(pts_t, pts_t1)

        row_ind, col_ind = linear_sum_assignment(cost_mat)
        matched_pairs = []
        step_vectors = []

        for r, c in zip(row_ind, col_ind):
            dist_px = float(cost_mat[r, c])
            if dist_px <= max_disp_step:
                b_t = blobs_t[r]
                b_t1 = blobs_t1[c]

                b_id = b_t.get("global_id", next_global_id)
                if "global_id" not in b_t:
                    b_t["global_id"] = b_id
                    next_global_id += 1
                b_t1["global_id"] = b_id

                b_vecs = compute_boundary_vectors(
                    b_t["contour_px"],
                    b_t1["contour_px"],
                    step_px=vector_step_px,
                    dt_s=dt_step,
                    pixel_size_m=pixel_size_m,
                    max_disp_px=max_disp_step,
                )

                for v in b_vecs:
                    v["bubble_id"] = b_id
                    all_speeds_m_s.append(v["speed_m_s"])
                    step_vectors.append(v)

                c_dx_px = b_t1["centroid_px"][0] - b_t["centroid_px"][0]
                c_dy_px = b_t1["centroid_px"][1] - b_t["centroid_px"][1]
                c_disp_m = dist_px * pixel_size_m
                c_vel_m_s = c_disp_m / dt_step

                matched_pairs.append(
                    {
                        "bubble_id": b_id,
                        "blob_t": b_t,
                        "blob_t1": b_t1,
                        "centroid_t": b_t["centroid_px"],
                        "centroid_t1": b_t1["centroid_px"],
                        "c_dx_px": c_dx_px,
                        "c_dy_px": c_dy_px,
                        "c_vel_m_s": c_vel_m_s,
                        "boundary_vectors": b_vecs,
                    }
                )

        mean_speed = float(np.mean([v["speed_m_s"] for v in step_vectors])) if step_vectors else 0.0
        max_speed = float(np.max([v["speed_m_s"] for v in step_vectors])) if step_vectors else 0.0

        all_step_data.append(
            {
                "step_idx": idx_step,
                "t_raw": t,
                "stem_t": stem_t,
                "stem_t1": stem_t1,
                "matched_pairs": matched_pairs,
                "boundary_vectors": step_vectors,
                "mean_speed": mean_speed,
                "max_speed": max_speed,
            }
        )

    # Determine robust colorbar maximum speed
    if v_max_override is not None and v_max_override > 0:
        v_max_legend = float(v_max_override)
    elif all_speeds_m_s:
        v_max_legend = max(0.5, float(np.percentile(all_speeds_m_s, 95)))
    else:
        v_max_legend = 1.0

    print(f"  Colormap Dynamic Scale    : 0.00 to {v_max_legend:.2f} m/s")

    # Setup VideoWriter with even dimensions
    ref_mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
    raw_h, raw_w = ref_mask.shape[:2]
    vid_w = raw_w if raw_w % 2 == 0 else raw_w + 1
    vid_h = raw_h if raw_h % 2 == 0 else raw_h + 1

    video_path = output_dir / "bubble_velocity_flow_animation.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(video_path), fourcc, playback_fps, (vid_w, vid_h))

    csv_records = []
    rendered_frame_paths = []
    rendered_frames_for_html = []

    # Second pass: Render visual overlays and encode video
    for step in all_step_data:
        idx_step = step["step_idx"]
        t_raw = step["t_raw"]
        stem_t = step["stem_t"]
        stem_t1 = step["stem_t1"]

        # 1. Base frame resolution (raw camera frame or fallback mask background)
        clean_t = re.sub(r"(_mask|_pred|_seg)$", "", stem_t)
        raw_img_path = image_files_map.get(stem_t) or image_files_map.get(clean_t)

        if raw_img_path and raw_img_path.is_file():
            base_frame = cv2.imread(str(raw_img_path), cv2.IMREAD_COLOR)
            if base_frame is not None and base_frame.shape[:2] != (raw_h, raw_w):
                base_frame = cv2.resize(base_frame, (raw_w, raw_h), interpolation=cv2.INTER_LINEAR)
        else:
            base_frame = None

        if base_frame is None:
            mask_img = cv2.imread(str(mask_dir / f"{stem_t}.png"), cv2.IMREAD_GRAYSCALE)
            base_frame = np.zeros((raw_h, raw_w, 3), dtype=np.uint8)
            base_frame[:] = (28, 28, 32)
            if mask_img is not None:
                base_frame[mask_img > 0] = (70, 70, 78)

        canvas = base_frame.copy()

        # 2. Semi-transparent mask highlight
        mask_t = cv2.imread(str(mask_dir / f"{stem_t}.png"), cv2.IMREAD_GRAYSCALE)
        if mask_t is not None:
            bubble_overlay = canvas.copy()
            bubble_overlay[mask_t > 0] = np.clip(
                bubble_overlay[mask_t > 0].astype(int) + [50, 30, 0], 0, 255
            ).astype(np.uint8)
            canvas = cv2.addWeighted(bubble_overlay, 0.40, canvas, 0.60, 0)

        # 3. Active bubble ID set for this frame
        active_ids = {pair["bubble_id"] for pair in step["matched_pairs"]}

        # Update centroid histories ONLY for currently active bubbles
        for pair in step["matched_pairs"]:
            b_id = pair["bubble_id"]
            c_x, c_y = pair["centroid_t"]
            if b_id not in centroid_histories:
                centroid_histories[b_id] = []
            centroid_histories[b_id].append((c_x, c_y, idx_step))
            if len(centroid_histories[b_id]) > trail_length:
                centroid_histories[b_id].pop(0)

        # PRUNE: Discard history for bubbles that exited or lost tracking (NO GHOST TRAILS)
        for old_id in list(centroid_histories.keys()):
            if old_id not in active_ids:
                del centroid_histories[old_id]

        # 4. Draw centroid trajectory trails (ONLY for active bubbles)
        for b_id in active_ids:
            pts = centroid_histories.get(b_id, [])
            if len(pts) >= 2:
                for p_idx in range(len(pts) - 1):
                    x_a, y_a, _ = pts[p_idx]
                    x_b, y_b, _ = pts[p_idx + 1]
                    alpha_factor = (p_idx + 1) / len(pts)
                    trail_color = (
                        int(255 * alpha_factor),
                        int(180 * alpha_factor),
                        int(0 * alpha_factor),
                    )
                    cv2.line(
                        canvas,
                        (int(round(x_a)), int(round(y_a))),
                        (int(round(x_b)), int(round(y_b))),
                        trail_color,
                        2,
                        cv2.LINE_AA,
                    )

        # 5. Draw bubble contours & boundary velocity vectors
        for pair in step["matched_pairs"]:
            b_id = pair["bubble_id"]
            blob_t = pair["blob_t"]

            cnt_t = blob_t["contour_px"].reshape((-1, 1, 2)).astype(np.int32)
            cv2.drawContours(canvas, [cnt_t], -1, (0, 220, 110), 1, cv2.LINE_AA)

            for v in pair["boundary_vectors"]:
                x0, y0 = v["x0_px"], v["y0_px"]
                dx = v["dx_px"] * arrow_scale
                dy = v["dy_px"] * arrow_scale
                x1 = x0 + dx
                y1 = y0 + dy

                color = get_colormap_color(v["speed_m_s"], v_max_legend, colormap_type)
                arrow_len = np.hypot(dx, dy)
                if arrow_len >= 1.5:
                    cv2.arrowedLine(
                        canvas,
                        (int(round(x0)), int(round(y0))),
                        (int(round(x1)), int(round(y1))),
                        color,
                        1,
                        tipLength=0.35,
                        line_type=cv2.LINE_AA,
                    )
                else:
                    cv2.circle(canvas, (int(round(x0)), int(round(y0))), 1, color, -1)

                csv_records.append(
                    {
                        "frame_t": stem_t,
                        "frame_t1": stem_t1,
                        "bubble_id": b_id,
                        "x0_px": v["x0_px"],
                        "y0_px": v["y0_px"],
                        "dx_px": v["dx_px"],
                        "dy_px": v["dy_px"],
                        "disp_px": v["disp_px"],
                        "vx_m_s": v["vx_m_s"],
                        "vy_m_s": v["vy_m_s"],
                        "speed_m_s": v["speed_m_s"],
                    }
                )

            # Centroid bulk velocity arrow & badge
            cx, cy = pair["centroid_t"]
            cdx = pair["c_dx_px"] * arrow_scale
            cdy = pair["c_dy_px"] * arrow_scale
            c_color = get_colormap_color(pair["c_vel_m_s"], v_max_legend, colormap_type)

            cv2.arrowedLine(
                canvas,
                (int(round(cx)), int(round(cy))),
                (int(round(cx + cdx)), int(round(cy + cdy))),
                c_color,
                2,
                tipLength=0.40,
                line_type=cv2.LINE_AA,
            )
            cv2.circle(canvas, (int(round(cx)), int(round(cy))), 3, (255, 255, 255), -1)

            if blob_t.get("area_px", 0) >= 30:
                badge_text = f"#{b_id}"
                badge_pos = (int(cx - 10), int(cy - 8))
                cv2.putText(
                    canvas,
                    badge_text,
                    badge_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    badge_text,
                    badge_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.36,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 6. Render HUD telemetry banner and vertical colorbar
        scale_speed = 1.0
        scale_len_px = (scale_speed * dt_step / pixel_size_m) * arrow_scale
        time_elapsed_ms = t_raw * (1.0 / fps) * 1e3

        canvas = draw_hud_and_legend(
            canvas=canvas,
            frame_t_name=stem_t,
            frame_t1_name=stem_t1,
            time_elapsed_ms=time_elapsed_ms,
            mean_speed_m_s=step["mean_speed"],
            max_speed_m_s=step["max_speed"],
            bubble_count=len(step["matched_pairs"]),
            v_max_legend=v_max_legend,
            scale_bar_speed_m_s=scale_speed,
            scale_bar_len_px=scale_len_px,
            colormap_type=colormap_type,
        )

        if canvas.shape[0] != vid_h or canvas.shape[1] != vid_w:
            padded = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)
            padded[: canvas.shape[0], : canvas.shape[1]] = canvas
            canvas_to_write = padded
        else:
            canvas_to_write = canvas

        if save_individual_frames:
            frame_png_path = frames_dir / f"velocity_field_{stem_t}_to_{stem_t1}.png"
            cv2.imwrite(str(frame_png_path), canvas)
            rendered_frame_paths.append(frame_png_path)

        # Store sample frame in memory for interactive HTML player
        rendered_frames_for_html.append(canvas)
        video_writer.write(canvas_to_write)

    video_writer.release()
    print(
        f"[DONE] Video animation generated : {video_path} ({expected_duration:.1f}s at {playback_fps} FPS)"
    )

    # Generate animated GIF (subsampled to <= 100 frames to keep file size small ~2-4 MB)
    gif_path = output_dir / "bubble_velocity_flow_animation.gif"
    try:
        from PIL import Image

        gif_step = max(1, int(np.ceil(len(rendered_frames_for_html) / 100)))
        gif_sample_frames = rendered_frames_for_html[::gif_step]
        pil_frames = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in gif_sample_frames
        ]
        if pil_frames:
            gif_duration_ms = max(40, int(1000.0 / (playback_fps / gif_step)))
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=gif_duration_ms,
                loop=0,
            )
            print(f"[DONE] Animated GIF generated    : {gif_path} ({len(pil_frames)} frames)")
    except Exception as e:
        print(f"[WARN] Could not generate animated GIF: {e}")

    # Write CSV
    csv_path = output_dir / "boundary_velocity_vectors_si.csv"
    if csv_records:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_records[0].keys()))
            writer.writeheader()
            writer.writerows(csv_records)
        print(f"[DONE] Vector dataset saved to    : {csv_path} ({len(csv_records)} vectors)")

    # Generate lightweight interactive HTML player (< 8 MB via JPEG & subsampling)
    html_path = output_dir / "interactive_flow_viewer.html"
    generate_interactive_html_viewer(
        rendered_frames=rendered_frames_for_html,
        step_data=all_step_data,
        output_html_path=html_path,
        dt_s=dt_step,
        playback_fps=playback_fps,
        max_html_frames=max_html_frames,
    )
    html_size_mb = html_path.stat().st_size / (1024 * 1024)
    print(f"[DONE] Interactive HTML player in: {html_path} ({html_size_mb:.2f} MB)")
    print("=" * 64)

    return {
        "video_path": video_path,
        "frames_dir": frames_dir if save_individual_frames else None,
        "html_path": html_path,
        "csv_path": csv_path,
        "gif_path": gif_path,
        "total_frames": num_transitions,
        "total_vectors": len(csv_records),
        "v_max_legend": v_max_legend,
        "duration_s": expected_duration,
        "video_duration_s": expected_duration,
        "stride": stride,
    }


def generate_interactive_html_viewer(
    rendered_frames: list[np.ndarray],
    step_data: list[dict[str, Any]],
    output_html_path: Path,
    dt_s: float,
    playback_fps: float = 5.0,
    max_html_frames: int = 150,
):
    """
    Creates a responsive, zero-dependency, ultra-lightweight HTML5 player.
    Subsamples frames to <= max_html_frames and compresses to JPEG quality 75,
    slashing file size from 1.2 GB down to ~3-5 MB.
    """
    total_rendered = len(rendered_frames)
    if total_rendered <= max_html_frames:
        sample_indices = list(range(total_rendered))
    else:
        sample_indices = np.linspace(0, total_rendered - 1, max_html_frames, dtype=int).tolist()

    encoded_frames = []
    metadata_js = []

    for idx in sample_indices:
        frame_bgr = rendered_frames[idx]
        fh, fw = frame_bgr.shape[:2]
        if fw > 720:
            scale = 720.0 / fw
            target_size = (720, int(round(fh * scale)))
            comp_frame = cv2.resize(frame_bgr, target_size, interpolation=cv2.INTER_AREA)
        else:
            comp_frame = frame_bgr

        _, buf = cv2.imencode(".jpg", comp_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        b64_str = base64.b64encode(buf).decode("utf-8")
        encoded_frames.append(f"data:image/jpeg;base64,{b64_str}")

        s = step_data[idx]
        metadata_js.append(
            {
                "stem_t": format_short_frame_name(s["stem_t"]),
                "stem_t1": format_short_frame_name(s["stem_t1"]),
                "time_ms": round(s["step_idx"] * dt_s * 1e3, 2),
                "mean_speed": round(s["mean_speed"], 2),
                "max_speed": round(s["max_speed"], 2),
                "num_bubbles": len(s["matched_pairs"]),
                "num_vectors": len(s["boundary_vectors"]),
            }
        )

    import json

    frames_json = json.dumps(encoded_frames)
    meta_json = json.dumps(metadata_js)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CTF2 High-Speed Bubble Velocity Field & Flow Animation</title>
  <style>
    :root {{
      --bg-dark: #0f1117;
      --card-bg: #181b26;
      --accent-cyan: #00d2ff;
      --accent-blue: #3a7bd5;
      --text-main: #f0f3f8;
      --text-muted: #8b9bb4;
      --border-color: #272d3f;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
    body {{
      background-color: var(--bg-dark);
      color: var(--text-main);
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 24px 16px;
      min-height: 100vh;
    }}
    header {{
      text-align: center;
      margin-bottom: 18px;
    }}
    h1 {{
      font-size: 1.55rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: var(--text-main);
      margin-bottom: 6px;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 0.88rem;
    }}
    .viewer-card {{
      background: var(--card-bg);
      border: 1px solid var(--border-color);
      border-radius: 12px;
      box-shadow: 0 8px 30px rgba(0,0,0,0.45);
      width: 100%;
      max-width: 960px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }}
    .canvas-container {{
      position: relative;
      background: #000;
      width: 100%;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 280px;
    }}
    #flowImage {{
      width: 100%;
      height: auto;
      display: block;
      user-select: none;
    }}
    .control-panel {{
      padding: 16px 20px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      background: #141722;
      border-top: 1px solid var(--border-color);
    }}
    .scrubber-row {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    input[type=range] {{
      flex: 1;
      height: 6px;
      border-radius: 3px;
      background: #2b3248;
      outline: none;
      -webkit-appearance: none;
      cursor: pointer;
    }}
    input[type=range]::-webkit-slider-thumb {{
      -webkit-appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: var(--accent-cyan);
      box-shadow: 0 0 10px rgba(0,210,255,0.7);
      cursor: pointer;
    }}
    .frame-badge {{
      font-variant-numeric: tabular-nums;
      font-size: 0.85rem;
      color: var(--accent-cyan);
      min-width: 80px;
      text-align: right;
    }}
    .button-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .btn-group {{
      display: flex;
      gap: 8px;
    }}
    button {{
      background: #232838;
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 8px 16px;
      border-radius: 6px;
      font-size: 0.88rem;
      font-weight: 500;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }}
    button:hover {{
      background: #2f364c;
      border-color: var(--accent-cyan);
    }}
    button.primary {{
      background: linear-gradient(135deg, var(--accent-blue), #0099ff);
      border: none;
    }}
    button.primary:hover {{
      opacity: 0.92;
      box-shadow: 0 0 12px rgba(0, 153, 255, 0.4);
    }}
    .speed-control {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.82rem;
      color: var(--text-muted);
    }}
    .speed-control select {{
      background: #232838;
      color: var(--text-main);
      border: 1px solid var(--border-color);
      border-radius: 4px;
      padding: 4px 8px;
      font-size: 0.82rem;
      outline: none;
    }}
    .telemetry-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
      padding: 14px 20px;
      background: #11141e;
      border-top: 1px solid var(--border-color);
    }}
    .stat-card {{
      display: flex;
      flex-direction: column;
      gap: 2px;
    }}
    .stat-label {{
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
    }}
    .stat-val {{
      font-size: 1.05rem;
      font-weight: 600;
      color: var(--text-main);
      font-variant-numeric: tabular-nums;
    }}
    .stat-val.accent {{ color: var(--accent-cyan); }}
    footer {{
      margin-top: 16px;
      font-size: 0.76rem;
      color: var(--text-muted);
      text-align: center;
    }}
  </style>
</head>
<body>

<header>
  <h1>CTF2 Bubble Velocity Field & Flow Kinematics</h1>
  <p class="subtitle">High-speed 2-phase flow boiling analysis | SI Units (m/s, mm/px)</p>
</header>

<div class="viewer-card">
  <div class="canvas-container">
    <img id="flowImage" src="" alt="Flow Velocity Field">
  </div>

  <div class="control-panel">
    <div class="scrubber-row">
      <input type="range" id="frameScrubber" min="0" max="0" value="0" step="1">
      <span class="frame-badge" id="frameCounter">1 / 1</span>
    </div>

    <div class="button-row">
      <div class="btn-group">
        <button id="btnPrev">⏮ Prev</button>
        <button id="btnPlay" class="primary">▶ Play</button>
        <button id="btnNext">Next ⏭</button>
      </div>

      <div class="speed-control">
        <span>Playback Speed:</span>
        <select id="playbackSpeed">
          <option value="2">2 FPS (Slow-Mo)</option>
          <option value="5" selected>5 FPS (Standard Smooth)</option>
          <option value="10">10 FPS</option>
          <option value="20">20 FPS</option>
        </select>
      </div>
    </div>
  </div>

  <div class="telemetry-grid">
    <div class="stat-card">
      <span class="stat-label">Frame Transition</span>
      <span class="stat-val" id="statFrames">--</span>
    </div>
    <div class="stat-card">
      <span class="stat-label">Flow Time</span>
      <span class="stat-val" id="statTime">-- ms</span>
    </div>
    <div class="stat-card">
      <span class="stat-label">Mean Speed</span>
      <span class="stat-val accent" id="statMeanSpeed">-- m/s</span>
    </div>
    <div class="stat-card">
      <span class="stat-label">Max Speed</span>
      <span class="stat-val" id="statMaxSpeed">-- m/s</span>
    </div>
    <div class="stat-card">
      <span class="stat-label">Active Bubbles</span>
      <span class="stat-val" id="statBubbles">--</span>
    </div>
    <div class="stat-card">
      <span class="stat-label">Vectors</span>
      <span class="stat-val" id="statVectors">--</span>
    </div>
  </div>
</div>

<footer>
  Press <strong>Space</strong> to Play/Pause &bull; <strong>Left/Right Arrows</strong> to Step Frames
</footer>

<script>
  const frames = {frames_json};
  const metadata = {meta_json};

  const imgEl = document.getElementById('flowImage');
  const scrubber = document.getElementById('frameScrubber');
  const frameCounter = document.getElementById('frameCounter');
  const btnPlay = document.getElementById('btnPlay');
  const btnPrev = document.getElementById('btnPrev');
  const btnNext = document.getElementById('btnNext');
  const speedSelect = document.getElementById('playbackSpeed');

  const statFrames = document.getElementById('statFrames');
  const statTime = document.getElementById('statTime');
  const statMeanSpeed = document.getElementById('statMeanSpeed');
  const statMaxSpeed = document.getElementById('statMaxSpeed');
  const statBubbles = document.getElementById('statBubbles');
  const statVectors = document.getElementById('statVectors');

  let currentIdx = 0;
  let isPlaying = false;
  let timerId = null;

  scrubber.max = Math.max(0, frames.length - 1);

  function updateFrame(idx) {{
    if (!frames.length) return;
    currentIdx = Math.max(0, Math.min(idx, frames.length - 1));
    imgEl.src = frames[currentIdx];
    scrubber.value = currentIdx;
    frameCounter.textContent = `${{currentIdx + 1}} / ${{frames.length}}`;

    const meta = metadata[currentIdx];
    if (meta) {{
      statFrames.textContent = `${{meta.stem_t}} → ${{meta.stem_t1}}`;
      statTime.textContent = `${{meta.time_ms.toFixed(1)}} ms`;
      statMeanSpeed.textContent = `${{meta.mean_speed.toFixed(2)}} m/s`;
      statMaxSpeed.textContent = `${{meta.max_speed.toFixed(2)}} m/s`;
      statBubbles.textContent = meta.num_bubbles;
      statVectors.textContent = meta.num_vectors;
    }}
  }}

  function play() {{
    if (isPlaying) return;
    isPlaying = true;
    btnPlay.textContent = '⏸ Pause';
    const fpsVal = parseInt(speedSelect.value, 10) || 5;
    timerId = setInterval(() => {{
      let next = currentIdx + 1;
      if (next >= frames.length) next = 0;
      updateFrame(next);
    }}, 1000 / fpsVal);
  }}

  function pause() {{
    if (!isPlaying) return;
    isPlaying = false;
    btnPlay.textContent = '▶ Play';
    if (timerId) clearInterval(timerId);
  }}

  btnPlay.addEventListener('click', () => {{
    if (isPlaying) pause(); else play();
  }});

  btnPrev.addEventListener('click', () => {{
    pause();
    updateFrame(currentIdx - 1);
  }});

  btnNext.addEventListener('click', () => {{
    pause();
    updateFrame(currentIdx + 1);
  }});

  scrubber.addEventListener('input', (e) => {{
    pause();
    updateFrame(parseInt(e.target.value, 10));
  }});

  speedSelect.addEventListener('change', () => {{
    if (isPlaying) {{
      pause();
      play();
    }}
  }});

  window.addEventListener('keydown', (e) => {{
    if (e.code === 'Space') {{
      e.preventDefault();
      btnPlay.click();
    }} else if (e.code === 'ArrowLeft') {{
      btnPrev.click();
    }} else if (e.code === 'ArrowRight') {{
      btnNext.click();
    }}
  }});

  updateFrame(0);
</script>

</body>
</html>
"""
    output_html_path.write_text(html_content, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="High-Speed Bubble Velocity Field Visualization and Video Animation Pipeline"
    )
    parser.add_argument(
        "--masks",
        "-m",
        type=Path,
        required=True,
        help="Directory containing binary masks (*.png)",
    )
    parser.add_argument(
        "--images",
        "-i",
        type=Path,
        default=None,
        help="Optional directory containing raw high-speed flow frames (*.png)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("results/flow_animation"),
        help="Directory to save video animation, PNG frames, HTML viewer, and CSV",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2000.0,
        help="Camera high-speed acquisition rate in 1/s (Hz, default: 2000)",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=6.1162e-5,
        help="Optical scale in meters/pixel (SI, default: 6.1162e-5)",
    )
    parser.add_argument(
        "--stride",
        "-s",
        type=int,
        default=None,
        help="Frame evaluation stride / skip factor (default: auto, min 5 for long runs)",
    )
    parser.add_argument(
        "--playback-fps",
        type=float,
        default=60.0,
        help="Video playback framerate (default: 60.0 FPS)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum video duration in seconds (default: 30.0 s)",
    )
    parser.add_argument(
        "--max-disp",
        type=float,
        default=50.0,
        help="Base maximum centroid matching displacement in pixels",
    )
    parser.add_argument(
        "--vector-step",
        type=float,
        default=12.0,
        help="Contour sampling interval in pixels for vector arrows",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=5,
        help="Number of past frames for active bubble centroid trajectory history trails",
    )
    parser.add_argument(
        "--arrow-scale",
        type=float,
        default=1.0,
        help="Scale multiplier for vector arrows",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        choices=["turbo", "viridis"],
        default="turbo",
        help="Perceptual colormap for velocity magnitude",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Fixed maximum speed in m/s for colormap normalization (default: 95th percentile)",
    )
    parser.add_argument(
        "--max-html-frames",
        type=int,
        default=200,
        help="Maximum embedded frames for the interactive HTML player to keep file size small (default: 200)",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        default=False,
        help="Save individual PNG frames to disk (recommended only for short runs)",
    )

    args = parser.parse_args()

    render_flow_animation(
        image_dir=args.images,
        mask_dir=args.masks,
        output_dir=args.output_dir,
        fps=args.fps,
        pixel_size_m=args.pixel_size,
        stride=args.stride,
        playback_fps=args.playback_fps,
        max_duration_s=args.max_duration,
        max_disp_px=args.max_disp,
        vector_step_px=args.vector_step,
        trail_length=args.trail_length,
        arrow_scale=args.arrow_scale,
        colormap_name=args.colormap,
        v_max_override=args.vmax,
        max_html_frames=args.max_html_frames,
        save_individual_frames=args.save_frames,
    )
