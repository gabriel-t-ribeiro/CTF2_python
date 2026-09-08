"""
CTF2 Interactive Pixel Size Calibration Tool.

Enables interactive 2-point optical scale calibration on experimental flow images:
- Zoom and pan using the Matplotlib toolbar.
- Click two fiducial points (e.g., inner pipe walls, ruler markings).
- Live on-screen preview of distance (px), pixel size (mm/px and m/px), and optical scale (px/mm).
- Generates a standardized `<folder_name>_calibration.txt` report with human-readable
  details and machine-parsable parameters.
- Provides helper functions `find_and_load_calibration` for automated pipeline ingestion.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_images(folder: Path) -> list[Path]:
    """Return a sorted list of valid image paths within a folder."""
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder does not exist or is not a directory: {folder}")
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS)
    if not files:
        raise FileNotFoundError(f"No supported images ({', '.join(VALID_EXTS)}) found in {folder}")
    return files


def resolve_reference_image(input_path: Path, ref_arg: str | None = None) -> Path:
    """Resolve the single image to use for calibration."""
    if input_path.is_file():
        if input_path.suffix.lower() not in VALID_EXTS:
            raise ValueError(f"File {input_path.name} is not a supported image format")
        return input_path

    images = list_images(input_path)
    if ref_arg is None:
        return images[0]

    # Try numeric index
    try:
        idx = int(ref_arg)
        if idx < 0 or idx >= len(images):
            raise IndexError(f"Reference index {idx} out of range (0 to {len(images) - 1})")
        return images[idx]
    except ValueError:
        pass

    # Try direct filename in folder
    matches = [f for f in images if f.name == ref_arg or f.stem == ref_arg]
    if matches:
        return matches[0]

    direct_path = Path(ref_arg)
    if direct_path.is_file():
        return direct_path

    raise FileNotFoundError(f"Reference image '{ref_arg}' not found in {input_path}")


def prompt_real_world_dimension() -> tuple[float, str, float]:
    """
    Prompt user in the terminal for real-world dimension if omitted from CLI.
    Returns (dimension_value, unit, dimension_in_meters).
    """
    print("\n--- Real-World Dimension Input ---")
    print("Please specify the known physical length between the 2 points you will select.")
    print("Default unit is millimeters (mm). Examples: '20' or '20mm' or '0.02m'.")
    raw = input("Enter dimension [default: 20 mm]: ").strip()

    if not raw:
        return 20.0, "mm", 0.020

    raw_lower = raw.lower()
    if raw_lower.endswith("um") or raw_lower.endswith("µm"):
        num_str = raw_lower.rstrip("uµm").strip()
        val = float(num_str)
        return val, "um", val * 1e-6
    elif raw_lower.endswith("mm"):
        num_str = raw_lower.rstrip("m").rstrip("m").strip()
        val = float(num_str)
        return val, "mm", val * 1e-3
    elif raw_lower.endswith("m"):
        num_str = raw_lower.rstrip("m").strip()
        val = float(num_str)
        return val, "m", val
    else:
        # Default unit is mm
        val = float(raw)
        return val, "mm", val * 1e-3


def calculate_pixel_scale(
    pt1: tuple[float, float],
    pt2: tuple[float, float],
    real_size_m: float,
) -> dict[str, float]:
    """Calculate Euclidean distance, line angle, and pixel scales."""
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    distance_px = math.hypot(dx, dy)
    if distance_px <= 0.0:
        raise ValueError("Selected points must be distinct (distance > 0)")

    # Angle in image coordinates (-180 to 180 degrees)
    angle_deg = math.degrees(math.atan2(dy, dx))

    # Angle from horizontal between 0 and 90 degrees (0 = horizontal, 90 = vertical)
    abs_angle = abs(angle_deg)
    if abs_angle > 90.0:
        abs_angle = 180.0 - abs_angle

    pixel_size_m = real_size_m / distance_px
    pixel_size_mm = pixel_size_m * 1e3
    pixel_size_um = pixel_size_m * 1e6
    pixels_per_mm = 1.0 / pixel_size_mm if pixel_size_mm > 0 else 0.0

    return {
        "dx_px": dx,
        "dy_px": dy,
        "distance_px": distance_px,
        "angle_deg": angle_deg,
        "angle_from_horizontal_deg": abs_angle,
        "pixel_size_m": pixel_size_m,
        "pixel_size_mm": pixel_size_mm,
        "pixel_size_um": pixel_size_um,
        "pixels_per_mm": pixels_per_mm,
    }


def select_calibration_points(
    image_path: Path,
    real_size_m: float,
    real_size_label: str,
) -> tuple[tuple[float, float], tuple[float, float], dict[str, float]]:
    """
    Open interactive Matplotlib window for 2-point selection with live scale & angle preview,
    direct mouse dragging of points, arrow key nudging, and [V]/[H] alignment snapping.
    Returns (point1, point2, scale_metrics_dict).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise OSError(f"Could not load image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img_rgb.shape[:2]

    clicks: list[tuple[float, float]] = []
    markers: list[Any] = []
    line_holder: list[Any] = [None]
    halo_holder: list[Any] = [None]
    active_idx: list[int] = [0]
    dragging_idx: list[int | None] = [None]
    original_pt2: list[tuple[float, float] | None] = [None]
    confirmed: list[bool] = [False]

    fig, ax = plt.subplots(figsize=(13, 8.5))
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.06, right=0.96)
    ax.imshow(img_rgb)

    default_title = (
        f"CALIBRATION: Select 2 Points Across Known Dimension ({real_size_label})\n"
        "Left-click: Set Point 1 & Point 2  |  Drag points to adjust  |  Zoom/Pan: Toolbar\n"
        "[Awaiting Point 1...]"
    )
    title_obj = ax.set_title(default_title, fontsize=11, weight="bold", pad=12)

    def redraw():
        if line_holder[0] is not None:
            line_holder[0].remove()
            line_holder[0] = None
        if halo_holder[0] is not None:
            halo_holder[0].remove()
            halo_holder[0] = None

        if len(clicks) == 0:
            title_obj.set_text(default_title)
            title_obj.set_color("black")
        elif len(clicks) == 1:
            x1, y1 = clicks[0]
            halo = ax.plot(
                x1, y1, "o", markersize=18, fillstyle="none", color="yellow", markeredgewidth=2.5
            )[0]
            halo_holder[0] = halo
            title_obj.set_text(
                f"CALIBRATION: Select 2 Points Across Known Dimension ({real_size_label})\n"
                f"Point 1 set at ({x1:.1f}, {y1:.1f}) px  -->  [CLICK Point 2]\n"
                "Drag with mouse or use Arrow keys to nudge (Shift=fine) | Right-click: undo"
            )
            title_obj.set_color("darkblue")
        elif len(clicks) == 2:
            p1, p2 = clicks[0], clicks[1]
            line = ax.plot(
                [p1[0], p2[0]], [p1[1], p2[1]], color="lime", linewidth=2.5, linestyle="-"
            )[0]
            line_holder[0] = line

            curr_act = active_idx[0]
            if curr_act < len(clicks):
                act_pt = clicks[curr_act]
                halo = ax.plot(
                    act_pt[0],
                    act_pt[1],
                    "o",
                    markersize=18,
                    fillstyle="none",
                    color="yellow",
                    markeredgewidth=2.5,
                )[0]
                halo_holder[0] = halo

            metrics = calculate_pixel_scale(p1, p2, real_size_m)

            abs_ang = metrics["angle_from_horizontal_deg"]
            if abs(metrics["dx_px"]) < 0.05 or abs_ang >= 89.95:
                angle_desc = "PERFECT VERTICAL (90.0°, dx=0)"
            elif abs(metrics["dy_px"]) < 0.05 or abs_ang <= 0.05:
                angle_desc = "PERFECT HORIZONTAL (0.0°, dy=0)"
            else:
                angle_desc = f"Angle = {abs_ang:.1f}° from horiz (dx={metrics['dx_px']:.1f}px, dy={metrics['dy_px']:.1f}px)"

            act_pt = clicks[curr_act]
            act_label = f"Pt {curr_act + 1} ({act_pt[0]:.1f}, {act_pt[1]:.1f})"
            title_obj.set_text(
                f"PREVIEW: Dist = {metrics['distance_px']:.2f} px  |  {angle_desc}\n"
                f"Pixel Size = {metrics['pixel_size_mm']:.5f} mm/px ({metrics['pixel_size_m']:.5e} m/px) | Optical Scale = {metrics['pixels_per_mm']:.2f} px/mm\n"
                f"[Active: {act_label}] Drag mouse | Arrows: nudge (1px) | Shift+Arrows: fine (0.2px) | [Tab]: switch | [V]/[H]: snap | [ENTER]: ok"
            )
            title_obj.set_color("darkgreen")

        fig.canvas.draw_idle()

    def clear_markers():
        for m in markers:
            m.remove()
        markers.clear()

    def on_click(event):
        # Ignore clicks when zoom or pan toolbar modes are active
        toolbar = getattr(fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", "") != "":
            return
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return

        if event.button == 1:  # Left click
            # Check if clicked near an existing point (within 25 display pixels)
            for i, (px_x, px_y) in enumerate(clicks):
                disp_pt = ax.transData.transform((px_x, px_y))
                dist = math.hypot(event.x - disp_pt[0], event.y - disp_pt[1])
                if dist <= 25.0:
                    dragging_idx[0] = i
                    active_idx[0] = i
                    original_pt2[0] = None
                    redraw()
                    return

            if len(clicks) < 2:
                x = float(event.xdata)
                y = float(event.ydata)
                clicks.append((x, y))
                m = ax.plot(x, y, "r+", markersize=14, mew=2.5)[0]
                markers.append(m)
                active_idx[0] = len(clicks) - 1
                original_pt2[0] = None
                redraw()
            else:
                # 2 points already exist and clicked far away: start over
                clicks.clear()
                clear_markers()
                original_pt2[0] = None
                x = float(event.xdata)
                y = float(event.ydata)
                clicks.append((x, y))
                m = ax.plot(x, y, "r+", markersize=14, mew=2.5)[0]
                markers.append(m)
                active_idx[0] = 0
                redraw()

        elif event.button == 3:  # Right click (undo)
            if clicks:
                clicks.pop()
                original_pt2[0] = None
                if markers:
                    markers.pop().remove()
                if clicks:
                    active_idx[0] = len(clicks) - 1
                redraw()

    def on_motion(event):
        if (
            dragging_idx[0] is not None
            and event.inaxes == ax
            and event.xdata is not None
            and event.ydata is not None
        ):
            idx = dragging_idx[0]
            nx = max(0.0, min(float(width - 1), float(event.xdata)))
            ny = max(0.0, min(float(height - 1), float(event.ydata)))
            clicks[idx] = (nx, ny)
            markers[idx].set_data([nx], [ny])
            redraw()

    def on_release(event):
        dragging_idx[0] = None

    def on_key(event):
        if event.key in ("enter", " ", "escape"):
            if len(clicks) == 2:
                confirmed[0] = True
                plt.close(fig)
        elif event.key == "tab":
            if len(clicks) == 2:
                active_idx[0] = 1 - active_idx[0]
                redraw()
        elif event.key == "1":
            if len(clicks) >= 1:
                active_idx[0] = 0
                redraw()
        elif event.key == "2":
            if len(clicks) >= 2:
                active_idx[0] = 1
                redraw()
        elif event.key in ("v", "V"):
            if len(clicks) == 2:
                # Snap point 2 to strictly vertical: x2 = x1
                p1 = clicks[0]
                p2 = clicks[1]
                if original_pt2[0] is None:
                    original_pt2[0] = p2
                clicks[1] = (p1[0], p2[1])
                markers[1].set_data([p1[0]], [p2[1]])
                redraw()
        elif event.key in ("h", "H"):
            if len(clicks) == 2:
                # Snap point 2 to strictly horizontal: y2 = y1
                p1 = clicks[0]
                p2 = clicks[1]
                if original_pt2[0] is None:
                    original_pt2[0] = p2
                clicks[1] = (p2[0], p1[1])
                markers[1].set_data([p2[0]], [p1[1]])
                redraw()
        elif event.key in ("u", "U"):
            if len(clicks) == 2 and original_pt2[0] is not None:
                # Undo snap: restore original point 2
                p2_orig = original_pt2[0]
                clicks[1] = p2_orig
                markers[1].set_data([p2_orig[0]], [p2_orig[1]])
                original_pt2[0] = None
                redraw()
        else:
            # Check arrow key nudges
            key_str = event.key.lower() if event.key else ""
            step = 1.0
            if "shift+" in key_str:
                step = 0.2
            elif "ctrl+" in key_str or "alt+" in key_str:
                step = 5.0

            arrow = key_str.split("+")[-1]
            if arrow in ("up", "down", "left", "right") and clicks:
                idx = active_idx[0]
                if idx >= len(clicks):
                    idx = len(clicks) - 1
                cx, cy = clicks[idx]
                if arrow == "up":
                    cy -= step
                elif arrow == "down":
                    cy += step
                elif arrow == "left":
                    cx -= step
                elif arrow == "right":
                    cx += step

                cx = max(0.0, min(float(width - 1), cx))
                cy = max(0.0, min(float(height - 1), cy))
                clicks[idx] = (cx, cy)
                markers[idx].set_data([cx], [cy])
                original_pt2[0] = None
                redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()  # Blocks until user closes or confirms

    if len(clicks) < 2:
        raise ValueError("Calibration cancelled: exactly 2 points are required.")

    p1, p2 = clicks[0], clicks[1]
    metrics = calculate_pixel_scale(p1, p2, real_size_m)
    return p1, p2, metrics


def write_calibration_file(
    output_path: Path,
    folder_name: str,
    ref_image_name: str,
    img_width: int,
    img_height: int,
    pt1: tuple[float, float],
    pt2: tuple[float, float],
    metrics: dict[str, float],
    real_size: float,
    real_unit: str,
    real_size_m: float,
) -> Path:
    """Write comprehensive human and machine-readable calibration report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    content = f"""============================================================
CTF2 Two-Phase Flow Pixel Scale Calibration Report
============================================================
Folder Name       : {folder_name}
Reference Image   : {ref_image_name}
Image Dimensions  : {img_width} x {img_height} px
Generated On      : {timestamp}
------------------------------------------------------------
Point 1 (x, y)    : ({pt1[0]:.2f}, {pt1[1]:.2f}) px
Point 2 (x, y)    : ({pt2[0]:.2f}, {pt2[1]:.2f}) px
Pixel Distance    : {metrics["distance_px"]:.2f} px
Line Orientation  : {metrics["angle_from_horizontal_deg"]:.2f} deg from horizontal (dx: {metrics["dx_px"]:.2f} px, dy: {metrics["dy_px"]:.2f} px)
------------------------------------------------------------
Physical Dimension: {real_size:.4f} {real_unit} ({real_size_m:.6f} m)
Pixel Size (SI)   : {metrics["pixel_size_m"]:.6e} m/px
Pixel Size        : {metrics["pixel_size_mm"]:.6f} mm/px ({metrics["pixel_size_um"]:.2f} um/px)
Optical Scale     : {metrics["pixels_per_mm"]:.4f} px/mm
============================================================

# MACHINE-READABLE CALIBRATION PARAMETERS
folder_name={folder_name}
reference_image={ref_image_name}
image_width_px={img_width}
image_height_px={img_height}
point1_x={pt1[0]:.2f}
point1_y={pt1[1]:.2f}
point2_x={pt2[0]:.2f}
point2_y={pt2[1]:.2f}
dx_px={metrics["dx_px"]:.4f}
dy_px={metrics["dy_px"]:.4f}
angle_deg={metrics["angle_deg"]:.2f}
angle_from_horizontal_deg={metrics["angle_from_horizontal_deg"]:.2f}
distance_px={metrics["distance_px"]:.4f}
diameter_real={real_size:.6f}
diameter_unit={real_unit}
diameter_m={real_size_m:.6e}
pixel_size_m={metrics["pixel_size_m"]:.6e}
pixel_size_mm={metrics["pixel_size_mm"]:.6f}
pixels_per_mm={metrics["pixels_per_mm"]:.6f}
timestamp={timestamp}
"""

    output_path.write_text(content, encoding="utf-8")
    return output_path


def parse_calibration_file(filepath: Path) -> dict[str, Any]:
    """
    Parse a CTF2 calibration text file into a Python dictionary.
    Converts numeric fields to floats/ints.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"Calibration file not found: {filepath}")

    data: dict[str, Any] = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("="):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                try:
                    if "." in val or "e" in val.lower():
                        data[key] = float(val)
                    else:
                        data[key] = int(val)
                except ValueError:
                    data[key] = val

    return data


def find_and_load_calibration(search_path: Path) -> dict[str, Any] | None:
    """
    Search for a calibration file given an image, mask, or directory path.
    Searches:
    1. search_path itself (if it is a calibration text file)
    2. search_path directory: <dir_name>_calibration.txt, then any *calibration*.txt
    3. search_path parent directory
    4. sibling 'images' directory if search_path is 'masks' or 'inferred_masks'
    Returns parsed dictionary or None if not found.
    """
    p = search_path.resolve()
    if p.is_file():
        if p.name.endswith("_calibration.txt") or "calibration" in p.name:
            try:
                return parse_calibration_file(p)
            except Exception:
                pass
        p = p.parent

    # Candidate directories
    dirs_to_check: list[Path] = [p]
    if p.parent != p:
        dirs_to_check.append(p.parent)
    if p.name in ("masks", "inferred_masks"):
        sibling_images = p.parent / "images"
        if sibling_images.is_dir():
            dirs_to_check.append(sibling_images)

    for d in dirs_to_check:
        if not d.is_dir():
            continue
        # Check standard priority forms (0_ prefix sorts first in PyCharm/IDEs)
        candidates = [
            d / f"0_{d.name}_calibration.txt",
            d / f"{d.name}_calibration.txt",
            d / "0_calibration.txt",
            d / "calibration.txt",
        ]
        for c in candidates:
            if c.is_file():
                try:
                    cal = parse_calibration_file(c)
                    cal["_file_path"] = str(c)
                    return cal
                except Exception:
                    pass

        # Any file with 'calibration' and ending in '.txt'
        for f in sorted(d.glob("*calibration*.txt")):
            if f.is_file():
                try:
                    cal = parse_calibration_file(f)
                    cal["_file_path"] = str(f)
                    return cal
                except Exception:
                    pass

    return None


def run_calibration(
    input_path: Path,
    size_mm: float | None = None,
    size_m: float | None = None,
    ref_arg: str | None = None,
    output_path: Path | None = None,
    copy_to: list[Path] | Path | None = None,
) -> Path:
    """Full execution handler for calibration."""
    input_path = input_path.resolve()
    ref_image_path = resolve_reference_image(input_path, ref_arg)
    folder_name = input_path.name if input_path.is_dir() else input_path.parent.name

    # Determine real-world dimension
    if size_mm is not None and size_m is not None:
        print("Warning: Both --size-mm and --size were provided. Using --size-mm.")
        real_size = size_mm
        real_unit = "mm"
        real_size_m = size_mm * 1e-3
    elif size_mm is not None:
        real_size = size_mm
        real_unit = "mm"
        real_size_m = size_mm * 1e-3
    elif size_m is not None:
        real_size = size_m
        real_unit = "m"
        real_size_m = size_m
    else:
        real_size, real_unit, real_size_m = prompt_real_world_dimension()

    real_size_label = (
        f"{real_size:.2f} mm ({real_size_m:.4f} m)"
        if real_unit == "mm"
        else f"{real_size_m:.4f} m ({real_size_m * 1e3:.2f} mm)"
    )

    print("\n====================================================")
    print("      CTF2 INTERACTIVE PIXEL SCALE CALIBRATION      ")
    print("====================================================")
    print(f"  Target Image     : {ref_image_path.name}")
    print(f"  Directory        : {ref_image_path.parent}")
    print(f"  Known Dimension  : {real_size_label}")
    print("====================================================")
    print("Opening interactive window. Click 2 points across the known distance.\n")

    p1, p2, metrics = select_calibration_points(ref_image_path, real_size_m, real_size_label)

    img = cv2.imread(str(ref_image_path))
    h, w = img.shape[:2]

    # Determine output file path (prefixed with 0_ to always sort to the top in PyCharm/IDEs)
    if output_path is None:
        save_dir = input_path if input_path.is_dir() else input_path.parent
        output_file = save_dir / f"0_{folder_name}_calibration.txt"
    else:
        output_file = output_path.resolve()

    saved_path = write_calibration_file(
        output_path=output_file,
        folder_name=folder_name,
        ref_image_name=ref_image_path.name,
        img_width=w,
        img_height=h,
        pt1=p1,
        pt2=p2,
        metrics=metrics,
        real_size=real_size,
        real_unit=real_unit,
        real_size_m=real_size_m,
    )

    # Determine secondary / additional export directories
    dest_dirs: list[Path] = []
    if copy_to is not None:
        if isinstance(copy_to, Path):
            dest_dirs = [copy_to]
        elif isinstance(copy_to, list | tuple):
            dest_dirs = [Path(p) for p in copy_to if p]
    else:
        # Check if running interactively with a terminal attached
        if sys.stdin.isatty():
            try:
                suggested = ""
                if input_path.is_dir() and input_path.name == "images":
                    sibling_masks = input_path.parent / "masks"
                    if sibling_masks.is_dir():
                        suggested = str(sibling_masks)

                prompt_text = "\nExport calibration to a second folder (e.g. dataset/masks)?"
                if suggested:
                    prompt_text += f"\n[Press Enter for '{suggested}', or 'n' to skip]: "
                else:
                    prompt_text += "\n[Press Enter to skip, or enter folder path]: "

                ans = input(prompt_text).strip()
                if ans:
                    if ans.lower() not in ("n", "no", "skip"):
                        dest_dirs.append(Path(ans))
                elif suggested:
                    dest_dirs.append(Path(suggested))
            except (EOFError, KeyboardInterrupt):
                pass

    exported_paths = [saved_path]
    for dest in dest_dirs:
        dest = dest.resolve()
        if dest.suffix.lower() == ".txt":
            target_file = dest
            target_folder_name = dest.parent.name
        else:
            dest.mkdir(parents=True, exist_ok=True)
            target_folder_name = dest.name
            target_file = dest / f"0_{target_folder_name}_calibration.txt"

        write_calibration_file(
            output_path=target_file,
            folder_name=target_folder_name,
            ref_image_name=ref_image_path.name,
            img_width=w,
            img_height=h,
            pt1=p1,
            pt2=p2,
            metrics=metrics,
            real_size=real_size,
            real_unit=real_unit,
            real_size_m=real_size_m,
        )
        exported_paths.append(target_file)

    print("\n====================================================")
    print("                CALIBRATION COMPLETE                ")
    print("====================================================")
    print(f"  Point 1 (x, y)   : ({p1[0]:.2f}, {p1[1]:.2f}) px")
    print(f"  Point 2 (x, y)   : ({p2[0]:.2f}, {p2[1]:.2f}) px")
    print(
        f"  Line Orientation : {metrics['angle_from_horizontal_deg']:.2f}° from horizontal (dx: {metrics['dx_px']:.2f} px, dy: {metrics['dy_px']:.2f} px)"
    )
    print(f"  Pixel Distance   : {metrics['distance_px']:.2f} px")
    print(f"  Physical Size    : {real_size:.4f} {real_unit} ({real_size_m:.6e} m)")
    print(f"  Pixel Size (SI)  : {metrics['pixel_size_m']:.6e} m/px")
    print(
        f"  Pixel Size       : {metrics['pixel_size_mm']:.6f} mm/px ({metrics['pixel_size_um']:.2f} um/px)"
    )
    print(f"  Optical Scale    : {metrics['pixels_per_mm']:.4f} px/mm")
    print(f"  Saved Report To  : {saved_path}")
    for exp in exported_paths[1:]:
        print(f"  Also Exported To : {exp}")
    print("====================================================\n")

    return saved_path


def build_calibration_parser() -> argparse.ArgumentParser:
    """Build and return argument parser for calibration CLI."""
    parser = argparse.ArgumentParser(
        description="CTF2 Interactive Pixel Size Calibration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="Input image folder or single image file to calibrate on",
    )
    parser.add_argument(
        "--size-mm",
        "-smm",
        type=float,
        default=None,
        help="Known physical dimension in millimeters (e.g., -smm 20)",
    )
    parser.add_argument(
        "--size",
        "-s",
        type=float,
        default=None,
        help="Known physical dimension in meters (e.g., -s 0.020)",
    )
    parser.add_argument(
        "--ref",
        "-ref",
        "-r",
        type=str,
        default=None,
        help="Reference image index (0-based) or filename in folder (default: first image)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional custom output path for calibration report text file",
    )
    parser.add_argument(
        "--copy-to",
        "-c",
        "--export-dir",
        "-e",
        type=Path,
        nargs="*",
        default=None,
        help="Additional folder(s) to export calibration report to (e.g., -c dataset/masks)",
    )
    return parser


def main():
    parser = build_calibration_parser()
    args = parser.parse_args()
    if args.input is None:
        parser.error("the following arguments are required: --input / -i")

    try:
        run_calibration(
            input_path=args.input,
            size_mm=args.size_mm,
            size_m=args.size,
            ref_arg=args.ref,
            output_path=args.output,
            copy_to=args.copy_to,
        )
    except KeyboardInterrupt:
        print("\nCalibration aborted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
