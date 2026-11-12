#!/usr/bin/env python3
# ROI batch cropper for image sequences (e.g., high-speed video frames).
#
# Interactive ROI selection with zoom/pan support (matplotlib toolbar).
# Click two points on the first image to define opposite corners of the crop
# rectangle. Use the toolbar (magnifying glass) to zoom before clicking.
#
# Usage:
#     change the path and result path in the script, then run it. The cropped images will be saved in the result path.
#     - You can also use coordinates with the --roi argument to skip the GUI and crop all images with the same ROI.
#           -- needs code change to function
#     - --input, --output, --roi, --ref) and proper exit codes. available

import argparse
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
base_workspace = Path(__file__).resolve().parents[1]
# Repository root on sys.path so `image_scripts.*` resolves when this file runs as a script.
if str(base_workspace) not in sys.path:
    sys.path.insert(0, str(base_workspace))
photo_path = base_workspace / "images" / "1st_run" / "images_1" / "raw_1"
result_path = base_workspace / "images" / "1st_run" / "images_2"


def list_images(folder: Path) -> list[Path]:
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS)
    if not files:
        raise FileNotFoundError(f"No images found in {folder}")
    return files


def select_roi_interactive(
    image_path: Path, fixed_size: tuple[int, int] | None = None
) -> tuple[int, int, int, int]:
    """
    Show the image with matplotlib.
    If fixed_size=(w, h) is None:
      User clicks two opposite corners of the ROI freely.
    If fixed_size=(w, h) is provided:
      The box dimensions are locked to (w, h). User clicks once to place the top-left
      corner, drags the box with the mouse, or nudges with arrow keys.
    Returns (x, y, w, h) in original image pixel coordinates.
    """
    import math

    img = cv2.imread(str(image_path))
    if img is None:
        raise OSError(f"Could not read {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img_rgb.shape[:2]

    clicks: list[tuple[float, float]] = []
    current_patch: list[Rectangle | None] = [None]
    markers: list[Any] = []
    halo_holder: list[Any] = [None]
    active_idx: list[int] = [0]
    dragging_idx: list[int | None] = [None]
    dragging_box: list[bool] = [False]
    drag_offset: list[tuple[float, float]] = [(0.0, 0.0)]

    fig, ax = plt.subplots(figsize=(13, 8.5))
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.06, right=0.96)
    ax.imshow(img_rgb)

    if fixed_size is not None:
        fw, fh = fixed_size
        default_title = (
            f"FIXED-SIZE ROI ({fw} x {fh} px): Click to place top-left corner\n"
            "Left-click: Place / Drag box  |  Arrows: nudge (1px) | [ENTER]: confirm\n"
            "[Awaiting Placement...]"
        )
    else:
        default_title = (
            "ROI SELECTION: Click two opposite corners of the crop rectangle\n"
            "Left-click: Set Point 1 & Point 2  |  Drag points to adjust  |  Zoom/Pan: Toolbar\n"
            "[Awaiting Point 1...]"
        )
    title_obj = ax.set_title(default_title, fontsize=11, weight="bold", pad=12)

    def redraw():
        if current_patch[0] is not None:
            current_patch[0].remove()
            current_patch[0] = None
        if halo_holder[0] is not None:
            halo_holder[0].remove()
            halo_holder[0] = None

        if len(clicks) == 0:
            title_obj.set_text(default_title)
            title_obj.set_color("black")
        elif fixed_size is not None and len(clicks) >= 1:
            fw, fh = fixed_size
            x1, y1 = clicks[0]
            rx = max(0, min(width - fw, round(x1)))
            ry = max(0, min(height - fh, round(y1)))

            rect = Rectangle((rx, ry), fw, fh, linewidth=2.2, edgecolor="cyan", facecolor="none")
            ax.add_patch(rect)
            current_patch[0] = rect

            halo = ax.plot(
                rx, ry, "o", markersize=14, fillstyle="none", color="yellow", markeredgewidth=2.5
            )[0]
            halo_holder[0] = halo

            title_obj.set_text(
                f"LOCKED-SIZE ROI: Top-Left=({rx}, {ry}) | Size={fw} x {fh} px | Bottom-Right=({rx + fw}, {ry + fh})\n"
                "Drag box with mouse | Arrows: nudge (1px) | Shift+Arrows: fine (0.2px) | [ENTER]: confirm"
            )
            title_obj.set_color("darkgreen")
        elif len(clicks) == 1:
            x1, y1 = clicks[0]
            halo = ax.plot(
                x1, y1, "o", markersize=18, fillstyle="none", color="yellow", markeredgewidth=2.5
            )[0]
            halo_holder[0] = halo
            title_obj.set_text(
                "ROI SELECTION: Click opposite corner (Point 2)\n"
                f"Point 1 set at: (x={x1:.1f}, y={y1:.1f}) px  -->  [CLICK Point 2]\n"
                "Drag with mouse or use Arrow keys to nudge (Shift=fine) | Right-click: undo"
            )
            title_obj.set_color("darkblue")
        elif len(clicks) == 2:
            p1, p2 = clicks[0], clicks[1]
            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
            rx, ry = min(x1, x2), min(y1, y2)
            rw, rh = abs(x2 - x1), abs(y2 - y1)

            rect = Rectangle((rx, ry), rw, rh, linewidth=2, edgecolor="lime", facecolor="none")
            ax.add_patch(rect)
            current_patch[0] = rect

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

            act_pt = clicks[curr_act]
            act_label = f"Pt {curr_act + 1} (x={act_pt[0]:.1f}, y={act_pt[1]:.1f})"
            title_obj.set_text(
                f"ROI: x={rx:.1f}, y={ry:.1f} | width={rw:.1f} px, height={rh:.1f} px\n"
                f"Pt 1: (x={x1:.1f}, y={y1:.1f})  |  Pt 2: (x={x2:.1f}, y={y2:.1f})\n"
                f"[Active: {act_label}] Drag mouse | Arrows: nudge (1px) | Shift+Arrows: fine (0.2px) | [Tab]: switch | [ENTER]: confirm"
            )
            title_obj.set_color("darkgreen")

        fig.canvas.draw_idle()

    def clear_markers():
        for marker in markers:
            marker.remove()
        markers.clear()

    def on_click(event):
        # Ignore clicks that happen while a toolbar tool (zoom/pan) is active.
        toolbar = getattr(fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", "") != "":
            return
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return

        if event.button == 1:  # Left click
            if fixed_size is not None:
                fw, fh = fixed_size
                # If a box is already placed, check if clicked inside it to drag it
                if clicks:
                    bx, by = clicks[0]
                    if bx <= event.xdata <= bx + fw and by <= event.ydata <= by + fh:
                        dragging_box[0] = True
                        drag_offset[0] = (float(event.xdata) - bx, float(event.ydata) - by)
                        return
                # Otherwise place top-left corner here
                clicks.clear()
                clear_markers()
                bx = max(0.0, min(float(width - fw), float(event.xdata)))
                by = max(0.0, min(float(height - fh), float(event.ydata)))
                clicks.append((bx, by))
                m = ax.plot(bx, by, "r+", markersize=14, mew=2.5)[0]
                markers.append(m)
                dragging_box[0] = True
                drag_offset[0] = (float(event.xdata) - bx, float(event.ydata) - by)
                redraw()
                return

            # Check if clicked near an existing point (within 25 display pixels)
            for i, (px_x, px_y) in enumerate(clicks):
                disp_pt = ax.transData.transform((px_x, px_y))
                dist = math.hypot(event.x - disp_pt[0], event.y - disp_pt[1])
                if dist <= 25.0:
                    dragging_idx[0] = i
                    active_idx[0] = i
                    redraw()
                    return

            if len(clicks) < 2:
                x = float(event.xdata)
                y = float(event.ydata)
                clicks.append((x, y))
                marker = ax.plot(x, y, "r+", markersize=14, mew=2.5)[0]
                markers.append(marker)
                active_idx[0] = len(clicks) - 1
                redraw()
            else:
                # 2 points already exist and clicked far away: start over
                clicks.clear()
                clear_markers()
                x = float(event.xdata)
                y = float(event.ydata)
                clicks.append((x, y))
                marker = ax.plot(x, y, "r+", markersize=14, mew=2.5)[0]
                markers.append(marker)
                active_idx[0] = 0
                redraw()

        elif event.button == 3:  # Right click (undo)
            if clicks:
                clicks.pop()
                if markers:
                    markers.pop().remove()
                if clicks:
                    active_idx[0] = len(clicks) - 1
                redraw()

    def on_motion(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return

        if fixed_size is not None and dragging_box[0] and clicks:
            fw, fh = fixed_size
            ox, oy = drag_offset[0]
            nx = max(0.0, min(float(width - fw), float(event.xdata) - ox))
            ny = max(0.0, min(float(height - fh), float(event.ydata) - oy))
            clicks[0] = (nx, ny)
            markers[0].set_data([nx], [ny])
            redraw()
            return

        if dragging_idx[0] is not None:
            idx = dragging_idx[0]
            nx = max(0.0, min(float(width - 1), float(event.xdata)))
            ny = max(0.0, min(float(height - 1), float(event.ydata)))
            clicks[idx] = (nx, ny)
            markers[idx].set_data([nx], [ny])
            redraw()

    def on_release(event):
        dragging_idx[0] = None
        dragging_box[0] = False

    def on_key(event):
        if event.key in ("enter", " ", "escape"):
            if fixed_size is not None and len(clicks) >= 1 or len(clicks) == 2:
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
                if fixed_size is not None:
                    fw, fh = fixed_size
                    cx, cy = clicks[0]
                    if arrow == "up":
                        cy -= step
                    elif arrow == "down":
                        cy += step
                    elif arrow == "left":
                        cx -= step
                    elif arrow == "right":
                        cx += step
                    cx = max(0.0, min(float(width - fw), cx))
                    cy = max(0.0, min(float(height - fh), cy))
                    clicks[0] = (cx, cy)
                    markers[0].set_data([cx], [cy])
                    redraw()
                    return

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
                redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()  # blocks until the user closes the window or presses Enter

    if fixed_size is not None:
        if not clicks:
            raise ValueError("ROI selection cancelled: box was not placed.")
        fw, fh = fixed_size
        x = max(0, min(width - fw, round(clicks[0][0])))
        y = max(0, min(height - fh, round(clicks[0][1])))
        return x, y, fw, fh

    if len(clicks) != 2:
        raise ValueError("Need exactly 2 points to define the ROI.")

    p1, p2 = clicks[0], clicks[1]
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    x = round(min(x1, x2))
    y = round(min(y1, y2))
    w = round(abs(x2 - x1))
    h = round(abs(y2 - y1))

    if w == 0 or h == 0:
        raise ValueError("ROI has zero area.")

    return x, y, w, h


def parse_size_string_or_file(size_input: str) -> tuple[int, int]:
    """
    Extract (width, height) from either:
    1. A 'w,h' string (e.g. '1051,472')
    2. An 'x,y,w,h' string (e.g. '0,137,1051,472')
    3. A path to a crop metadata file (e.g. 0_crop_roi.txt) or folder containing one.
    """
    val = size_input.strip()
    path_candidate = Path(val)
    if path_candidate.is_dir() or path_candidate.is_file():
        _, _, w, h = parse_roi_string_or_file(val)
        return w, h

    parts = [p.strip() for p in val.split(",")]
    if len(parts) == 2:
        w, h = int(parts[0]), int(parts[1])
    elif len(parts) == 4:
        _, _, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    else:
        raise ValueError(
            f"Invalid size input '{size_input}'. Expected 'w,h' (e.g. '1051,472'), "
            "'x,y,w,h', or path to a crop metadata file/folder."
        )
    if w <= 0 or h <= 0:
        raise ValueError("Width and height must be positive")
    return w, h


def parse_roi_string_or_file(roi_input: str) -> tuple[int, int, int, int]:
    """
    Parse an ROI from either:
    1. A comma-separated string: "x,y,w,h" (e.g. "0,137,1051,472")
    2. A path to a file (e.g. "0_crop_roi.txt") or directory containing a crop report.
    Returns (x, y, w, h).
    """
    path_candidate = Path(roi_input.strip())
    # If a directory was provided, check for 0_crop_roi.txt or *crop*.txt inside it
    if path_candidate.is_dir():
        txt_files = sorted(path_candidate.glob("*crop*.txt"))
        if not txt_files:
            raise FileNotFoundError(
                f"No crop metadata (*crop*.txt) found in directory: {path_candidate}"
            )
        path_candidate = txt_files[0]

    # If it is a file, parse the ROI from it
    if path_candidate.is_file():
        content = path_candidate.read_text(encoding="utf-8")
        # 1. Look for 'roi_str=x,y,w,h'
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("roi_str="):
                val = line.split("=", 1)[1].strip()
                return parse_roi_string_or_file(val)
        # 2. Look for individual key-values
        kv = {}
        for line in content.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                kv[k.strip().lower()] = v.strip()
        if all(k in kv for k in ("roi_x", "roi_y", "roi_width", "roi_height")):
            return (
                int(kv["roi_x"]),
                int(kv["roi_y"]),
                int(kv["roi_width"]),
                int(kv["roi_height"]),
            )
        # 3. Look for plain 'x,y,w,h' line
        for line in content.splitlines():
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) == 4 and all(p.lstrip("-").isdigit() for p in parts):
                return tuple(int(p) for p in parts)  # type: ignore

        raise ValueError(f"Could not parse valid ROI parameters from file: {path_candidate}")

    # Otherwise treat as literal "x,y,w,h"
    parts = [p.strip() for p in roi_input.split(",")]
    if len(parts) != 4:
        raise ValueError(
            f"Invalid ROI input '{roi_input}'. Expected 'x,y,w,h' (e.g. '0,137,1051,472') "
            "or a path to a crop metadata file/folder."
        )
    try:
        x, y, w, h = (int(v) for v in parts)
    except ValueError as err:
        raise ValueError("ROI values must be integers") from err
    if x < 0 or y < 0:
        raise ValueError("ROI x and y must be non-negative")
    if w <= 0 or h <= 0:
        raise ValueError("ROI width and height must be positive")
    return x, y, w, h


def save_crop_metadata(
    output_dir: Path,
    roi: tuple[int, int, int, int],
    input_dir: Path | None = None,
    ref_image_name: str | None = None,
    total_images: int = 0,
) -> Path:
    """
    Save both human-readable and machine-readable crop ROI metadata to output_dir.
    Creates 0_crop_roi.txt (prefixed with 0_ so it sorts to the top in file explorers).
    Includes a dedicated, prominent copy-paste string and CLI command.
    """
    x, y, w, h = roi
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "0_crop_roi.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    input_str = str(input_dir) if input_dir else "N/A"
    ref_str = ref_image_name if ref_image_name else "N/A"
    roi_str = f"{x},{y},{w},{h}"

    content = f"""============================================================
CTF2 Image Sequence Crop & ROI Report
============================================================
Target Directory  : {output_dir.name}
Output Path       : {output_dir}
Input Directory   : {input_str}
Reference Image   : {ref_str}
Total Images      : {total_images}
Timestamp         : {timestamp}
------------------------------------------------------------
COPY-PASTE READY:
  ROI String      : {roi_str}
  CLI Flag        : --roi "{roi_str}"
------------------------------------------------------------
Crop ROI Breakdown:
  Top-Left (x, y) : ({x}, {y}) px
  Dimensions      : {w} (width) x {h} (height) px
  Bottom-Right    : ({x + w}, {y + h}) px
============================================================

# QUICK COPY-PASTE VALUE (NO LABELS)
{roi_str}

# MACHINE-READABLE CROP PARAMETERS
roi_str={roi_str}
roi_x={x}
roi_y={y}
roi_width={w}
roi_height={h}
roi_x2={x + w}
roi_y2={y + h}
reference_image={ref_str}
total_images={total_images}
timestamp={timestamp}
"""
    report_file.write_text(content, encoding="utf-8")
    print(f"  [metadata] Saved crop ROI info -> {report_file.name}")
    print(f"  [metadata] Quick copy-paste ROI: {roi_str}")
    return report_file


def crop_and_save(
    files: list[Path],
    roi: tuple[int, int, int, int],
    out_dir: Path,
    ref_image_name: str | None = None,
    input_dir: Path | None = None,
) -> None:
    x, y, w, h = roi
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok, n_fail = 0, 0

    for i, f in enumerate(files, 1):
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  [skip] Could not read {f.name}")
            n_fail += 1
            continue

        height, width = img.shape[:2]
        if x >= width or y >= height:
            print(f"  [skip] ROI outside image bounds for {f.name}")
            n_fail += 1
            continue

        x2 = min(x + w, width)
        y2 = min(y + h, height)
        cropped = img[max(0, y) : y2, max(0, x) : x2]

        if cropped.size == 0:
            print(f"  [skip] Empty crop for {f.name}")
            n_fail += 1
            continue

        ok = cv2.imwrite(str(out_dir / f.name), cropped)
        if not ok:
            print(f"  [fail] Could not write {f.name}")
            n_fail += 1
            continue

        n_ok += 1

        if i % 100 == 0 or i == len(files):
            print(f"  Processed {i}/{len(files)}")

    # Save human-readable and machine-readable metadata file
    save_crop_metadata(
        output_dir=out_dir,
        roi=roi,
        input_dir=input_dir if input_dir else (files[0].parent if files else None),
        ref_image_name=ref_image_name,
        total_images=n_ok,
    )

    print(f"\nDone. Saved: {n_ok}, Failed: {n_fail} -> {out_dir}")


def find_calibration_files(target_dir: Path) -> list[Path]:
    """Find all calibration text files directly within target_dir."""
    if not target_dir.is_dir():
        return []
    return sorted([f for f in target_dir.glob("*calibration*.txt") if f.is_file()])


def format_recalled_calibration(cal: dict[str, Any]) -> str:
    """Format previous calibration values for clear terminal display."""
    lines = [
        "\n" + "=" * 58,
        "           PREVIOUS CALIBRATION DATA RECALLED             ",
        "=" * 58,
    ]
    if "_file_path" in cal:
        lines.append(f"  Source File   : {cal['_file_path']}")
    if "diameter_real" in cal and "diameter_unit" in cal:
        dia_m = cal.get("diameter_m")
        m_str = f" ({dia_m:.4f} m)" if dia_m is not None else ""
        lines.append(f"  Diameter      : {cal['diameter_real']} {cal['diameter_unit']}{m_str}")
    elif "diameter_m" in cal:
        lines.append(
            f"  Diameter      : {cal['diameter_m']:.4f} m ({cal['diameter_m'] * 1e3:.2f} mm)"
        )

    if "pixel_size_mm" in cal:
        um = cal.get("pixel_size_um")
        um_str = f" ({um:.2f} um/px)" if um is not None else ""
        lines.append(f"  Pixel Size    : {cal['pixel_size_mm']:.6f} mm/px{um_str}")
    elif "pixel_size_m" in cal:
        lines.append(f"  Pixel Size    : {cal['pixel_size_m']:.6e} m/px")

    if "pixels_per_mm" in cal:
        lines.append(f"  Optical Scale : {cal['pixels_per_mm']:.4f} px/mm")
    if "distance_px" in cal:
        lines.append(f"  Measured Dist : {cal['distance_px']:.2f} px")
    lines.append("=" * 58)
    return "\n".join(lines)


def parse_custom_calibration_call(
    call_str: str,
    output_dir: Path,
    recalled_cal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse custom CLI arguments entered by the user for recalibration."""
    from image_scripts.calibrate_pixel_size import build_calibration_parser

    parser = build_calibration_parser()
    tokens = shlex.split(call_str.strip()) if call_str.strip() else []

    # Strip redundant prefixes if user typed 'python main.py calibrate' or 'calibrate.py'
    while tokens and (tokens[0] in ("python", "python3", "calibrate") or tokens[0].endswith(".py")):
        tokens = tokens[1:]

    # If user provided no dimension flags, fall back to recalled dimension
    has_size = any(t in ("--size-mm", "-smm", "--size", "-s") for t in tokens)
    if not has_size and recalled_cal:
        if (
            "diameter_real" in recalled_cal
            and str(recalled_cal.get("diameter_unit", "")).lower() == "mm"
        ):
            tokens.extend(["--size-mm", str(recalled_cal["diameter_real"])])
        elif "diameter_m" in recalled_cal:
            tokens.extend(["--size-mm", f"{float(recalled_cal['diameter_m']) * 1e3:.4f}"])

    try:
        parsed, unknown = parser.parse_known_args(tokens)
        if unknown:
            print(f"[CALIBRATION] Notice: Unrecognized custom flags ignored: {' '.join(unknown)}")
    except Exception as e:
        print(f"[CALIBRATION] Argument parsing warning ({e}). Using defaults...")
        parsed, _ = parser.parse_known_args([])

    return {
        "input_path": parsed.input or output_dir,
        "size_mm": parsed.size_mm,
        "size_m": parsed.size,
        "ref_arg": parsed.ref,
        "output_path": parsed.output,
        "copy_to": parsed.copy_to,
    }


def prompt_calibration_action(
    cal_files: list[Path],
    recalled_cal: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    """Prompt user to choose what to do with existing calibration file(s)."""
    print("\n" + "=" * 54)
    print("           CALIBRATION FILE(S) DETECTED           ")
    print("=" * 54)
    print("Existing calibration file(s) found:")
    for f in cal_files:
        print(f"  - {f}")
    print("-" * 54)
    print("What would you like to do?")
    print("  [1] Delete calibration (default: data becomes stale after cropping)")
    print("  [2] Do nothing (keep existing calibration)")
    print("  [3] Redo calibration (re-calibrate on cropped output)")
    print("=" * 54)

    while True:
        try:
            ans = input("Select option [1/2/3, default: 1 (delete)]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDefaulting to [1] Delete calibration.")
            return "delete", {}

        if ans in ("", "1", "delete", "remove", "d"):
            return "delete", {}
        elif ans in ("2", "keep", "do nothing", "do-nothing", "nothing", "k"):
            return "keep", {}
        elif ans in ("3", "redo", "recalibrate", "re-calibrate", "r"):
            default_call = ""
            if recalled_cal:
                print(format_recalled_calibration(recalled_cal))
                if (
                    "diameter_real" in recalled_cal
                    and str(recalled_cal.get("diameter_unit", "")).lower() == "mm"
                ):
                    default_call = f"--size-mm {recalled_cal['diameter_real']}"
                elif "diameter_m" in recalled_cal:
                    default_call = f"--size-mm {float(recalled_cal['diameter_m']) * 1e3:.2f}"

            print("\nEnter how you want to call the calibration for cropped output.")
            print(
                "You can specify CLI flags, e.g.: --size-mm 20 --copy-to dataset/masks --output custom.txt"
            )
            if default_call:
                print(f"[Press Enter for default: '{default_call}']")
            else:
                print("[Press Enter for default interactive calibration]")

            try:
                custom_call = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                custom_call = default_call

            if not custom_call and default_call:
                custom_call = default_call

            target_out = output_dir or Path(".")
            redo_kwargs = parse_custom_calibration_call(custom_call, target_out, recalled_cal)
            return "redo", redo_kwargs
        else:
            print("Invalid selection. Please enter 1, 2, or 3.")


def check_and_handle_calibration(
    input_dir: Path,
    output_dir: Path,
    action: str | None = None,
    redo_args: str | None = None,
) -> tuple[str, list[Path], dict[str, Any]]:
    """
    Check if calibration file(s) exist in input_dir or output_dir.
    If found:
      - Prompt or apply action ('delete', 'keep', 'redo').
      - Recalls previous calibration parameters if available.
      - Standard/default is 'delete' since cropping makes prior calibration stale.
      - If 'delete', unlinks found calibration files immediately.
    Returns (action, list_of_found_files, redo_kwargs).
    """
    cals_in = find_calibration_files(input_dir)
    cals_out = find_calibration_files(output_dir) if output_dir.exists() else []
    all_cals = sorted(set(cals_in + cals_out))

    if not all_cals:
        return "none", [], {}

    # Attempt to parse recalled calibration from found files
    recalled_cal: dict[str, Any] | None = None
    from image_scripts.calibrate_pixel_size import find_and_load_calibration, parse_calibration_file

    for f in all_cals:
        try:
            recalled_cal = parse_calibration_file(f)
            recalled_cal["_file_path"] = str(f)
            break
        except Exception:
            pass
    if recalled_cal is None:
        recalled_cal = find_and_load_calibration(input_dir)

    redo_kwargs: dict[str, Any] = {}

    if action is None:
        if sys.stdin.isatty():
            action, redo_kwargs = prompt_calibration_action(
                all_cals, recalled_cal=recalled_cal, output_dir=output_dir
            )
        else:
            print(
                f"[CALIBRATION DETECTED] Found {len(all_cals)} calibration file(s). "
                "Defaulting to 'delete' (stale after cropping) in non-interactive mode."
            )
            action = "delete"

    action = action.lower().strip()
    if action in ("", "1", "delete", "remove", "d"):
        normalized = "delete"
    elif action in ("2", "keep", "do nothing", "do-nothing", "nothing", "k"):
        normalized = "keep"
    elif action in ("3", "redo", "recalibrate", "re-calibrate", "r"):
        normalized = "redo"
        if not redo_kwargs:
            redo_kwargs = parse_custom_calibration_call(redo_args or "", output_dir, recalled_cal)
    else:
        normalized = "delete"

    if normalized == "delete":
        for f in all_cals:
            try:
                f.unlink()
                print(f"[CALIBRATION] Deleted stale calibration: {f}")
            except Exception as e:
                print(f"[CALIBRATION] Error deleting {f}: {e}", file=sys.stderr)
    elif normalized == "redo":
        # Clear outdated calibration in output_dir so new calibration will be clean
        for f in cals_out:
            try:
                f.unlink()
                print(f"[CALIBRATION] Cleared outdated output calibration: {f.name}")
            except Exception:
                pass

    return normalized, all_cals, redo_kwargs


def run_crop_pipeline(
    input_dir: Path,
    output_dir: Path,
    roi_arg: str | None = None,
    size_arg: str | None = None,
    ref_arg: str | None = None,
    on_calibration: str | None = None,
    redo_args: str | None = None,
) -> None:
    """Execute complete crop pipeline with calibration checks and optional recalibration."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    files = list_images(input_dir)
    print(f"Found {len(files)} images in {input_dir}")

    # Check for calibration files and handle according to user preference
    cal_action, found_cals, redo_kwargs = check_and_handle_calibration(
        input_dir, output_dir, action=on_calibration, redo_args=redo_args
    )

    if roi_arg:
        roi = parse_roi_string_or_file(roi_arg)
    else:
        if ref_arg is None:
            ref = files[0]
        else:
            try:
                idx = int(ref_arg)
                ref = files[idx]
            except ValueError:
                ref_path = Path(ref_arg)
                if ref_path.is_file():
                    ref = ref_path
                else:
                    matches = [f for f in files if f.name == ref_arg]
                    if not matches:
                        raise FileNotFoundError(f"Reference image '{ref_arg}' not found") from None
                    ref = matches[0]

        fixed_size = parse_size_string_or_file(size_arg) if size_arg else None
        if fixed_size is not None:
            print(
                f"Selecting position for fixed-size ROI ({fixed_size[0]} x {fixed_size[1]} px) on: {ref.name}"
            )
        else:
            print(f"Selecting ROI on: {ref.name}")
        roi = select_roi_interactive(ref, fixed_size=fixed_size)

    print(f"Using ROI (x, y, w, h) = {roi}")
    ref_name = (
        ref.name
        if "ref" in locals() and hasattr(ref, "name")
        else (files[0].name if files else None)
    )
    crop_and_save(files, roi, output_dir, ref_image_name=ref_name, input_dir=input_dir)

    if cal_action == "redo":
        print("\n[CALIBRATION] Launching calibration on newly cropped images...")
        from image_scripts.calibrate_pixel_size import run_calibration

        target_input = redo_kwargs.get("input_path") or output_dir
        target_ref = redo_kwargs.get("ref_arg") or ref_arg
        target_output = redo_kwargs.get("output_path")
        target_copy = redo_kwargs.get("copy_to")
        target_size_mm = redo_kwargs.get("size_mm")
        target_size_m = redo_kwargs.get("size_m")

        run_calibration(
            input_path=target_input,
            size_mm=target_size_mm,
            size_m=target_size_m,
            ref_arg=target_ref,
            output_path=target_output,
            copy_to=target_copy,
        )


def main():
    parser = argparse.ArgumentParser(description="ROI batch cropper for image sequences")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=photo_path,
        help="Input directory with images",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=result_path,
        help="Output directory for cropped images",
    )
    parser.add_argument(
        "--roi",
        help="ROI as 'x,y,w,h' OR path to a metadata file (e.g. 0_crop_roi.txt) or folder containing one",
    )
    parser.add_argument(
        "--size",
        "-s",
        help="Lock crop box to fixed (w,h) size (e.g. '1051,472' or path to previous crop file/folder). "
        "Allows clicking/dragging to choose location without altering image resolution.",
    )
    parser.add_argument(
        "--ref",
        help="Reference image filename or index (0-based) to select ROI on (default: first image)",
        default=None,
    )
    parser.add_argument(
        "--on-calibration",
        choices=["delete", "keep", "redo", "do-nothing"],
        default=None,
        help="Action if calibration files exist: 'delete' (default: stale data), 'keep' (do nothing), or 'redo'",
    )
    parser.add_argument(
        "--redo-args",
        type=str,
        default=None,
        help="CLI arguments for recalibration when --on-calibration redo is selected (e.g. '--size-mm 20 --copy-to dataset/masks')",
    )
    args = parser.parse_args()

    try:
        run_crop_pipeline(
            input_dir=args.input,
            output_dir=args.output,
            roi_arg=args.roi,
            size_arg=args.size,
            ref_arg=args.ref,
            on_calibration=args.on_calibration,
            redo_args=args.redo_args,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
