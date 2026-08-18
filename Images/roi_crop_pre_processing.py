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
import sys
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
photo_path = Path(
    r"/Users/gabrielribeiro/Library/CloudStorage/GoogleDrive-gribeiro.gr933@gmail.com/O meu "
    r"disco/Estagio/CTF2_python/Images/images_1/raw_1"
)
result_path = Path(r"Images/images_2/")


def list_images(folder: Path) -> list[Path]:
    files = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS
    )
    if not files:
        raise FileNotFoundError(f"No images found in {folder}")
    return files


def select_roi_interactive(image_path: Path) -> tuple[int, int, int, int]:
    """
    Show the image with matplotlib. User zooms/pans freely, then clicks two
    points defining opposite corners of the ROI. Returns (x, y, w, h) in
    original image pixel coordinates.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise OSError(f"Could not read {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    clicks: list[tuple[float, float]] = []
    current_patch: list[Optional[Rectangle]] = [None]
    markers = []

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_rgb)
    ax.set_title(
        "Zoom/pan with the toolbar, then CLICK two opposite corners.\n"
        "Right-click to undo last point. Close window when done."
    )

    def redraw_rect():
        if current_patch[0] is not None:
            current_patch[0].remove()
            current_patch[0] = None

        if len(clicks) == 2:
            p1, p2 = clicks[0], clicks[1]
            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            rect = Rectangle(
                (x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none"
            )
            ax.add_patch(rect)
            current_patch[0] = rect
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

        if event.button == 1:  # left click: add point
            if len(clicks) >= 2:
                clicks.clear()
                clear_markers()
            marker = ax.plot(event.xdata, event.ydata, "r+", markersize=12, mew=2)[0]
            markers.append(marker)
            clicks.append((float(event.xdata), float(event.ydata)))
            redraw_rect()
        elif event.button == 3:  # right click: undo
            if clicks:
                clicks.pop()
                if markers:
                    markers.pop().remove()
                redraw_rect()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()  # blocks until the user closes the window

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


def crop_and_save(
    files: list[Path], roi: tuple[int, int, int, int], out_dir: Path
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

    print(f"\nDone. Saved: {n_ok}, Failed: {n_fail} -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="ROI batch cropper for image sequences"
    )
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
    parser.add_argument("--roi", help="ROI as x,y,w,h (skip interactive selection)")
    parser.add_argument(
        "--ref",
        help="Reference image filename or index (0-based) to select ROI on (default: first image)",
        default=None,
    )
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    roi_arg = args.roi
    ref_arg = args.ref

    try:
        files = list_images(input_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"Found {len(files)} images in {input_dir}")

    if roi_arg:
        parts = [p.strip() for p in roi_arg.split(",")]
        if len(parts) != 4:
            print(
                "Error: --roi expects 4 comma-separated values: x,y,w,h",
                file=sys.stderr,
            )
            sys.exit(2)
        try:
            x, y, w, h = (int(v) for v in parts)
        except ValueError:
            print("Error: ROI values must be integers", file=sys.stderr)
            sys.exit(2)
        roi = (x, y, w, h)
    else:
        if ref_arg is None:
            ref = files[0]
        else:
            # try index
            try:
                idx = int(ref_arg)
                ref = files[idx]
            except ValueError:
                # treat as filename
                ref_path = Path(ref_arg)
                if ref_path.is_file():
                    ref = ref_path
                else:
                    matches = [f for f in files if f.name == ref_arg]
                    if not matches:
                        print(
                            f"Error: reference image {ref_arg} not found",
                            file=sys.stderr,
                        )
                        sys.exit(2)
                    ref = matches[0]
        print(f"Selecting ROI on: {ref.name}")
        roi = select_roi_interactive(ref)

    print(f"Using ROI (x, y, w, h) = {roi}")
    crop_and_save(files, roi, output_dir)


if __name__ == "__main__":
    main()
