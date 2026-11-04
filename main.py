#!/usr/bin/env python3
"""
CTF2 Two-Phase Flow Boiling Pipeline Orchestrator.

Unified Command Line Interface for:
  - Image Pre-processing & ROI Cropping
  - Glare & Margin Preserving Background Subtraction
  - Seed Mask Generation & Annotation Conversion
  - Deep Learning Segmentation (U-Net Training, Evaluation & Batch Inference)
  - 3D Void Fraction Reconstruction & Rouhani-Axelsson Drift-Flux Validation
  - Bubble Kinematics & Hydrodynamic Breakup Tracking (2000 FPS)
  - Single-Camera Depth from Defocus (DFD) Sizing & 3D Spatial Localization
  - Publication-Grade 3D Flow Visualization & Interactive HTML Rendering
"""

import argparse
import sys
from pathlib import Path


def _resolve_auto_calibration(args, path: Path):
    """Auto-detect pixel_size and diameter from calibration file if not explicitly passed."""
    from image_scripts.calibrate_pixel_size import find_and_load_calibration

    cal = find_and_load_calibration(path)
    if getattr(args, "pixel_size", None) is None:
        if cal and "pixel_size_m" in cal:
            args.pixel_size = float(cal["pixel_size_m"])
            source = cal.get("_file_path", "calibration file")
            print(
                f"[CALIBRATION] Auto-loaded optical scale from {source}:\n"
                f"              pixel_size = {args.pixel_size:.6e} m/px ({args.pixel_size * 1e3:.4f} mm/px)"
            )
        else:
            args.pixel_size = 6.1e-5
            print(
                f"[INFO] No calibration file found near '{path}'. Using default pixel_size: {args.pixel_size:.6e} m/px"
            )

    if hasattr(args, "diameter") and getattr(args, "diameter", None) is None:
        if cal and "diameter_m" in cal:
            args.diameter = float(cal["diameter_m"])
            source = cal.get("_file_path", "calibration file")
            print(
                f"[CALIBRATION] Auto-loaded channel diameter from {source}:\n"
                f"              diameter = {args.diameter:.4f} m ({args.diameter * 1e3:.1f} mm)"
            )
        else:
            args.diameter = 0.020


def main():
    parser = argparse.ArgumentParser(
        description="CTF2 Flow Boiling Analysis & 3D Reconstruction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available pipeline stages")

    # Subcommand: void-fraction
    p_vf = subparsers.add_parser("void-fraction", help="Compute 3D gas volumes and void fraction")
    p_vf.add_argument(
        "--masks", "-m", type=Path, required=True, help="Folder containing binary masks"
    )
    p_vf.add_argument(
        "--output-csv",
        "-o",
        type=Path,
        default=Path("results/void_fraction.csv"),
        help="Output CSV path",
    )
    p_vf.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Pixel size in m/px (SI, default: auto-detected or 6.1e-5 = 0.061 mm/px)",
    )
    p_vf.add_argument(
        "--diameter",
        "-d",
        type=float,
        default=None,
        help="Channel inner diameter in meters (SI, default: auto-detected or 0.020 m = 20 mm)",
    )
    p_vf.add_argument("--vapor-quality", "-x", type=float, default=None, help="Vapor quality x [-]")
    p_vf.add_argument(
        "--mass-flux", "-G", type=float, default=None, help="Mass flux G [kg/(m^2 s)]"
    )
    p_vf.add_argument("--render", action="store_true", help="Render sample 3D visualization")
    p_vf.add_argument("--render-output", type=Path, default=None, help="Path for rendered 3D image")
    p_vf.add_argument("--pipe-center-y", type=float, default=None, help="Pipe centerline y [px]")
    p_vf.add_argument(
        "--regime-thresh-px",
        type=float,
        default=None,
        help="Regime area threshold [px]",
    )
    p_vf.add_argument(
        "--plot-timeline",
        action="store_true",
        help="Automatically generate publication-quality void fraction timeline plot and interactive dashboard",
    )
    p_vf.add_argument(
        "--fps",
        type=float,
        default=2000.0,
        help="Camera framerate in FPS for timeline mapping (default: 2000.0)",
    )

    # Subcommand: track
    p_tr = subparsers.add_parser(
        "track", help="Track bubble kinematics and breakup events at 2000 FPS"
    )
    p_tr.add_argument(
        "--masks", "-m", type=Path, required=True, help="Folder containing binary masks"
    )
    p_tr.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("results/kinematics"),
        help="Output directory for CSV logs",
    )
    p_tr.add_argument("--fps", type=float, default=2000.0, help="Framerate in FPS (default: 2000)")
    p_tr.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Pixel size in m/px (SI, default: auto-detected or 6.1e-5 = 0.061 mm/px)",
    )
    p_tr.add_argument(
        "--v-max",
        type=float,
        default=6.0,
        help="Physical velocity cutoff in m/s (default: 6.0 m/s; dynamically converted to pixels once upfront based on calibration)",
    )
    p_tr.add_argument(
        "--max-disp",
        type=float,
        default=None,
        help="Explicit max search displacement in pixels (overrides --v-max if passed)",
    )
    p_tr.add_argument(
        "--images",
        "-i",
        type=Path,
        default=None,
        help="Optional folder with raw images for visual background overlay",
    )
    p_tr.add_argument(
        "--render-animation",
        action="store_true",
        help="Render slow-motion MP4 animation, PNG figures, and interactive HTML player",
    )
    p_tr.add_argument(
        "--playback-fps",
        type=float,
        default=5.0,
        help="Playback framerate for generated MP4 animation (default: 5.0 FPS)",
    )
    p_tr.add_argument(
        "--vector-step",
        type=float,
        default=12.0,
        help="Contour sampling interval in pixels for boundary vector arrows (default: 12.0)",
    )

    # Subcommand: animate-flow
    p_anim = subparsers.add_parser(
        "animate-flow",
        aliases=["visualize-tracking"],
        help="Generate high-speed bubble velocity vector field animation (MP4, PNGs, interactive HTML)",
    )
    p_anim.add_argument(
        "--masks", "-m", type=Path, required=True, help="Folder containing binary masks"
    )
    p_anim.add_argument(
        "--images",
        "-i",
        type=Path,
        default=None,
        help="Optional folder containing raw high-speed flow frames",
    )
    p_anim.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("results/flow_animation"),
        help="Output folder for animation video, PNG figures, HTML player, and CSV",
    )
    p_anim.add_argument(
        "--fps", type=float, default=2000.0, help="Camera frame rate in 1/s (Hz, default: 2000)"
    )
    p_anim.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Pixel size in m/px (SI, default: auto-detected or 6.1e-5 = 0.061 mm/px)",
    )
    p_anim.add_argument(
        "--max-disp",
        type=float,
        default=50.0,
        help="Max centroid search displacement in pixels (default: 50 px)",
    )
    p_anim.add_argument(
        "--vector-step",
        type=float,
        default=12.0,
        help="Contour sampling interval in pixels for vector arrows (default: 12.0)",
    )
    p_anim.add_argument(
        "--playback-fps",
        type=float,
        default=5.0,
        help="Video playback framerate for smooth slow-motion observation (default: 5.0 FPS)",
    )
    p_anim.add_argument(
        "--max-duration",
        type=float,
        default=20.0,
        help="Maximum duration of the rendered animation in seconds (default: 20.0 s, e.g. 10 to 30s)",
    )
    p_anim.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame step between compared pairs (default: 1 for smooth consecutive frame tracking)",
    )
    p_anim.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Starting frame index in the sequence (default: 0)",
    )
    p_anim.add_argument(
        "--max-html-frames",
        type=int,
        default=150,
        help="Maximum number of frames embedded in the interactive HTML player (default: 150)",
    )
    p_anim.add_argument(
        "--save-frames",
        action="store_true",
        help="Save individual annotated PNG frames to disk (default: False for large datasets)",
    )
    p_anim.add_argument(
        "--trail-length",
        type=int,
        default=5,
        help="Number of past frames for centroid trajectory history trails (default: 5)",
    )
    p_anim.add_argument(
        "--colormap",
        type=str,
        choices=["turbo", "viridis"],
        default="turbo",
        help="Perceptual colormap for velocity magnitude encoding (default: turbo)",
    )
    p_anim.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Fixed max speed in m/s for colormap normalization (default: 95th percentile)",
    )

    # Subcommand: animate-3d
    p_3d_anim = subparsers.add_parser(
        "animate-3d",
        help="Generate 3D animated WebGL flow reconstruction across time in SI units",
    )
    p_3d_anim.add_argument(
        "--masks", "-m", type=Path, required=True, help="Folder containing binary masks"
    )
    p_3d_anim.add_argument(
        "--output-html",
        "-o",
        type=Path,
        default=Path("results/w45_3d_flow_animation/interactive_3d_flow_viewer.html"),
        help="Output standalone HTML interactive 3D WebGL player",
    )
    p_3d_anim.add_argument(
        "--fps", type=float, default=2000.0, help="Camera frame rate in Hz (default: 2000)"
    )
    p_3d_anim.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Pixel scale in m/px (SI). If omitted, parsed from 0_*_calibration.txt",
    )
    p_3d_anim.add_argument(
        "--diameter",
        type=float,
        default=0.020,
        help="Conduit internal diameter in meters (default: 0.020 m)",
    )
    p_3d_anim.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Starting frame index (default: 0)",
    )
    p_3d_anim.add_argument(
        "--max-frames",
        type=int,
        default=120,
        help="Maximum frames to reconstruct in 3D (default: 120)",
    )
    p_3d_anim.add_argument(
        "--stride",
        "-s",
        type=int,
        default=1,
        help="Frame evaluation stride (default: 1)",
    )
    p_3d_anim.add_argument(
        "--slug-step",
        type=int,
        default=6,
        help="Sampling step for 3D slug mesh grid (default: 6 px)",
    )
    p_3d_anim.add_argument(
        "--title",
        type=str,
        default="3D Multi-Phase Bubble Flow Animation",
        help="Header title for the 3D HTML viewer",
    )
    p_3d_anim.add_argument(
        "--calib-file",
        "-c",
        type=Path,
        default=None,
        help="Path to calibration file (*calibration.txt) to automatically resolve pixel size, pipe diameter, and vertical channel centerline",
    )
    p_3d_anim.add_argument(
        "--y-center",
        type=float,
        default=None,
        help="Explicit channel centerline vertical coordinate in pixels (defaults to mask center or calibrated midpoint)",
    )

    # Subcommand: dfd
    p_dfd = subparsers.add_parser(
        "dfd", help="Perform single-camera Depth from Defocus (DFD) analysis"
    )
    p_dfd.add_argument(
        "--images",
        "-i",
        type=Path,
        required=True,
        help="Folder containing raw flow images",
    )
    p_dfd.add_argument(
        "--masks", "-m", type=Path, required=True, help="Folder containing binary masks"
    )
    p_dfd.add_argument(
        "--output-csv",
        "-o",
        type=Path,
        default=Path("results/dfd_results.csv"),
        help="Output CSV path",
    )
    p_dfd.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Pixel size in m/px (SI, default: auto-detected or 6.1e-5 = 0.061 mm/px)",
    )
    p_dfd.add_argument(
        "--beta-eff",
        type=float,
        default=2000.0,
        help="Effective DFD calibration in px/m",
    )
    p_dfd.add_argument(
        "--near-wall-threshold",
        type=float,
        default=0.05,
        help="Dimensionless blur threshold (sigma_tilde) for near-wall/near-focus regularization (default: 0.05)",
    )
    p_dfd.add_argument(
        "--max-sigma-threshold",
        type=float,
        default=0.35,
        help="Upper dimensionless blur threshold (sigma_tilde) for DFD validity (default: 0.35)",
    )
    p_dfd.add_argument(
        "--wall-thickness",
        type=float,
        default=0.003,
        help="Borosilicate glass wall thickness in meters (default: 0.003 m = 3.0 mm)",
    )
    p_dfd.add_argument(
        "--diameter",
        "-d",
        type=float,
        default=0.020,
        help="Channel inner diameter in meters (default: 0.020 m = 20.0 mm)",
    )
    p_dfd.add_argument(
        "--pipe-center-y",
        type=float,
        default=None,
        help="Pipe centerline y [px]",
    )
    p_dfd.add_argument(
        "--render-diagnostic",
        action="store_true",
        help="Render publication-grade 2-panel DFD diagnostic visualizations (x-y front view and y-z depth profile)",
    )
    p_dfd.add_argument(
        "--diagnostic-dir",
        type=Path,
        default=Path("results/dfd_diagnostics"),
        help="Output directory for diagnostic visualizations (default: results/dfd_diagnostics)",
    )
    p_dfd.add_argument(
        "--background",
        "-bg",
        type=Path,
        default=None,
        help="Optional background image for normalization",
    )

    # Subcommand: render
    p_rend = subparsers.add_parser("render", help="Render 3D flow structure and interactive HTML")
    p_rend.add_argument("--mask", "-m", type=Path, required=True, help="Single binary mask path")
    p_rend.add_argument(
        "--output-png",
        type=Path,
        default=Path("results/3d_render.png"),
        help="Output PNG path",
    )
    p_rend.add_argument(
        "--output-html",
        type=Path,
        default=Path("results/3d_render.html"),
        help="Output interactive HTML path",
    )
    p_rend.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Pixel size in m/px (default: auto-detected or 6.1e-5 = 0.061 mm/px)",
    )
    p_rend.add_argument(
        "--diameter",
        type=float,
        default=None,
        help="Pipe diameter in meters (default: auto-detected or 0.020 m = 20 mm)",
    )
    p_rend.add_argument(
        "--smoothing-lambda",
        "--lambda",
        dest="smoothing_lambda",
        type=float,
        default=0.15,
        help="Laplacian boundary smoothing parameter lambda in pixels (default: 0.15)",
    )

    # Subcommand: plot-vf
    p_pvf = subparsers.add_parser(
        "plot-vf",
        aliases=["plot-void-fraction", "plot-timeline"],
        help="Generate publication-grade void fraction timeline graph and interactive dashboard",
    )
    p_pvf.add_argument(
        "--input",
        "-i",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to void fraction CSV file(s) or directory",
    )
    p_pvf.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("results/void_fraction_timeline.png"),
        help="Output path for publication figure (PNG, SVG, or PDF)",
    )
    p_pvf.add_argument(
        "--html",
        "--interactive-html",
        dest="interactive_html",
        type=Path,
        default=Path("results/void_fraction_timeline.html"),
        help="Output path for standalone interactive HTML viewer",
    )
    p_pvf.add_argument(
        "--no-html",
        action="store_true",
        help="Disable interactive HTML dashboard generation",
    )
    p_pvf.add_argument(
        "--fps",
        type=float,
        default=2000.0,
        help="High-speed camera framerate in 1/s (Hz, default: 2000.0 FPS, dt = 0.5 ms)",
    )
    p_pvf.add_argument(
        "--time-unit",
        choices=["ms", "s"],
        default="ms",
        help="Physical time unit for timeline axis (default: ms)",
    )
    p_pvf.add_argument(
        "--absolute-time",
        action="store_true",
        help="Use absolute camera time instead of zero-relative start time",
    )
    p_pvf.add_argument(
        "--rolling-window",
        "-w",
        type=int,
        default=5,
        help="Moving average smoothing window in frames (default: 5)",
    )
    p_pvf.add_argument(
        "--vapor-quality",
        "-x",
        type=float,
        default=None,
        help="Thermodynamic vapor quality x [-] for Rouhani-Axelsson benchmark",
    )
    p_pvf.add_argument(
        "--mass-flux",
        "-G",
        type=float,
        default=None,
        help="Total mass flux G [kg/(m^2 s)] for Rouhani-Axelsson benchmark",
    )
    p_pvf.add_argument(
        "--multi-panel",
        action="store_true",
        help="Generate 3-panel layout: Void fraction, Gas volume, and Distribution histogram",
    )
    p_pvf.add_argument(
        "--alpha-col",
        type=str,
        default=None,
        help="Explicit column name for void fraction values",
    )
    p_pvf.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Custom labels for series when comparing multiple CSVs",
    )
    p_pvf.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom figure title",
    )
    p_pvf.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution in DPI for raster figure output (default: 300)",
    )

    # Subcommand: calibrate
    p_cal = subparsers.add_parser(
        "calibrate", help="Interactively calibrate optical pixel size (m/px) using 2 points"
    )
    p_cal.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input image folder or single image file",
    )
    p_cal.add_argument(
        "--size-mm",
        "-smm",
        type=float,
        default=None,
        help="Known physical dimension in millimeters (e.g., -smm 20)",
    )
    p_cal.add_argument(
        "--size",
        "-s",
        type=float,
        default=None,
        help="Known physical dimension in meters (e.g., -s 0.020)",
    )
    p_cal.add_argument(
        "--ref",
        type=str,
        default=None,
        help="Reference image index (0-based) or filename in folder (default: first image)",
    )
    p_cal.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional custom output path for calibration report text file",
    )
    p_cal.add_argument(
        "--copy-to",
        "-c",
        "--export-dir",
        "-e",
        type=Path,
        nargs="*",
        default=None,
        help="Additional folder(s) to export calibration report to (e.g., -c dataset/masks)",
    )

    # Subcommand: gen-masks
    p_gm = subparsers.add_parser(
        "gen-masks", help="Generate initial binary masks using Otsu thresholding"
    )
    p_gm.add_argument("--input", "-i", type=Path, required=True, help="Input raw images folder")
    p_gm.add_argument("--output", "-o", type=Path, required=True, help="Output folder for masks")
    p_gm.add_argument("--min-area", type=int, default=1, help="Minimum bubble area in px")
    p_gm.add_argument("--max-area", type=int, default=50000, help="Maximum bubble area in px")

    # Subcommand: bg-sub
    p_bg = subparsers.add_parser(
        "bg-sub", help="Perform background subtraction preserving defocus margins"
    )
    p_bg.add_argument("--input", "-i", type=Path, required=True, help="Input images folder")
    p_bg.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output folder for cleaned images",
    )
    p_bg.add_argument(
        "--bg-source",
        "--bg",
        type=Path,
        default=None,
        help="Optional path to background image file or folder",
    )
    p_bg.add_argument(
        "--threshold",
        "-t",
        type=int,
        default=15,
        help="Intensity difference threshold (0-255, default: 15; try 35-50 for shifted/noisy frames)",
    )
    p_bg.add_argument(
        "--margin",
        "-m",
        type=int,
        default=10,
        help="Padding margin in pixels around bubble to preserve blur (default: 10)",
    )
    p_bg.add_argument(
        "--blur",
        type=int,
        default=0,
        help="Gaussian pre-blur kernel size (e.g. 3, 5) to tolerate slight focus/grain differences (default: 0)",
    )
    p_bg.add_argument(
        "--sample-range",
        "--sample-direction",
        type=str,
        default=None,
        help="Consecutive frame range for background estimation as 'start,count' (e.g. '0,80' for first 80 frames) "
        "or 'start:end' (e.g. '0:80')",
    )
    p_bg.add_argument(
        "--normalize-lighting",
        action="store_true",
        help="Scale global brightness to match background before differencing",
    )

    # Subcommand: crop
    p_crop = subparsers.add_parser(
        "crop", help="Interactive or batch ROI cropping for image sequences"
    )
    p_crop.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input image folder with images to crop",
    )
    p_crop.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output folder for cropped images",
    )
    p_crop.add_argument(
        "--roi",
        type=str,
        default=None,
        help="ROI coordinates as 'x,y,w,h' OR path to a crop metadata file (e.g. 0_crop_roi.txt) or folder",
    )
    p_crop.add_argument(
        "--size",
        "-s",
        type=str,
        default=None,
        help="Lock crop box to fixed (w,h) size (e.g. '1051,472' or path to previous crop file/folder). "
        "Allows clicking/dragging to choose location without altering image resolution.",
    )
    p_crop.add_argument(
        "--ref",
        type=str,
        default=None,
        help="Reference image filename or index (0-based) to select ROI on",
    )
    p_crop.add_argument(
        "--on-calibration",
        choices=["delete", "keep", "redo", "do-nothing"],
        default=None,
        help="Action if calibration files are found: 'delete' (default: stale data), 'keep' (do nothing), or 'redo'",
    )
    p_crop.add_argument(
        "--redo-args",
        type=str,
        default=None,
        help="CLI arguments for recalibration when --on-calibration redo is selected",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    import importlib

    if args.command == "calibrate":
        from image_scripts.calibrate_pixel_size import run_calibration

        run_calibration(
            input_path=args.input,
            size_mm=args.size_mm,
            size_m=args.size,
            ref_arg=args.ref,
            output_path=args.output,
            copy_to=args.copy_to,
        )

    elif args.command == "void-fraction":
        _resolve_auto_calibration(args, args.masks)
        try:
            mod = importlib.import_module("model_training.4_void_fraction_reconstruct")
        except ModuleNotFoundError:
            mod = importlib.import_module("model_training.4_2d_void_fraction_reconstruct")
        mod.run_reconstruction(
            args.masks,
            args.output_csv,
            args.pixel_size,
            args.diameter,
            args.vapor_quality,
            args.mass_flux,
            render_sample=args.render,
            render_output=args.render_output,
            pipe_center_y_px=args.pipe_center_y,
            regime_thresh_px=args.regime_thresh_px,
            plot_timeline=getattr(args, "plot_timeline", False),
            fps=getattr(args, "fps", 2000.0),
        )

    elif args.command in ("plot-vf", "plot-void-fraction", "plot-timeline"):
        mod = importlib.import_module("model_training.plot_void_fraction_timeline")
        html_out = None if args.no_html else args.interactive_html
        mod.run_timeline_pipeline(
            input_paths=args.input,
            output_plot=args.output,
            output_html=html_out,
            fps=args.fps,
            time_unit=args.time_unit,
            relative_time=not args.absolute_time,
            rolling_window=args.rolling_window,
            vapor_quality=args.vapor_quality,
            mass_flux=args.mass_flux,
            multi_panel=args.multi_panel,
            alpha_col=args.alpha_col,
            labels=args.labels,
            title=args.title,
            dpi=args.dpi,
        )

    elif args.command == "track":
        _resolve_auto_calibration(args, args.masks)
        mod = importlib.import_module("model_training.5_track_bubbles_kinematics")
        mod.run_tracking(
            args.masks,
            args.output_dir,
            args.fps,
            args.pixel_size,
            max_disp_px=args.max_disp,
            v_max=args.v_max,
            image_dir=args.images,
            render_animation=args.render_animation,
            playback_fps=args.playback_fps,
            vector_step_px=args.vector_step,
        )

    elif args.command in ("animate-flow", "visualize-tracking"):
        _resolve_auto_calibration(args, args.masks)
        mod = importlib.import_module("model_training.visualize_flow_animation")
        mod.render_flow_animation(
            image_dir=args.images,
            mask_dir=args.masks,
            output_dir=args.output_dir,
            fps=args.fps,
            pixel_size_m=args.pixel_size,
            max_disp_px=args.max_disp,
            vector_step_px=args.vector_step,
            playback_fps=args.playback_fps,
            max_duration_s=args.max_duration,
            stride=args.stride,
            start_frame=args.start_frame,
            trail_length=args.trail_length,
            colormap_name=args.colormap,
            v_max_override=args.vmax,
            max_html_frames=args.max_html_frames,
            save_individual_frames=args.save_frames,
        )

    elif args.command == "animate-3d":
        if args.calib_file is None:
            _resolve_auto_calibration(args, args.masks)
        mod = importlib.import_module("model_training.render_3d_flow_animation")
        mod.process_sequence_3d(
            mask_dir=args.masks,
            output_html_path=args.output_html,
            pixel_size_m=args.pixel_size,
            pipe_dia_m=args.diameter,
            fps=args.fps,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            stride=args.stride,
            slug_mesh_step=args.slug_step,
            title=args.title,
            calib_file=args.calib_file,
            y_center_px=args.y_center,
        )

    elif args.command == "dfd":
        _resolve_auto_calibration(args, args.images)
        mod = importlib.import_module("model_training.6_depth_from_defocus")
        mod.run_dfd_pipeline(
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

    elif args.command == "render":
        _resolve_auto_calibration(args, args.mask)
        import cv2

        mod = importlib.import_module("model_training.render_3d_flow")
        mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not load mask at: {args.mask}")
        recon = mod.reconstruct_3d_bubbles_si(
            mask,
            pixel_size_m=args.pixel_size,
            pipe_dia_m=args.diameter,
            smoothing_lambda_px=args.smoothing_lambda,
        )
        mod.render_3d_flow_visualization(recon, args.output_png)
        mod.render_interactive_html(recon, args.output_html)
        print(f"[DONE] Rendered 3D flow to {args.output_png} and {args.output_html}")

    elif args.command == "gen-masks":
        mod = importlib.import_module("image_scripts.2_generate_masks")
        mod.generate_masks(args.input, args.output, min_area=args.min_area, max_area=args.max_area)

    elif args.command == "bg-sub":
        mod = importlib.import_module("image_scripts.1_background_subtraction")
        mod.run_batch_background_removal(
            args.input,
            args.output,
            bg_source=args.bg_source,
            sample_range=args.sample_range,
            diff_threshold=args.threshold,
            margin_px=args.margin,
            blur_kernel_size=args.blur,
            normalize_lighting=args.normalize_lighting,
        )

    elif args.command == "crop":
        mod = importlib.import_module("image_scripts.0_roi_crop")
        mod.run_crop_pipeline(
            input_dir=args.input,
            output_dir=args.output,
            roi_arg=args.roi,
            size_arg=args.size,
            ref_arg=args.ref,
            on_calibration=args.on_calibration,
            redo_args=args.redo_args,
        )


if __name__ == "__main__":
    main()
