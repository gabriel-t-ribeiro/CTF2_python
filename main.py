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
        default=6.7e-5,
        help="Pixel size in m/px (SI, default: 6.7e-5)",
    )
    p_vf.add_argument(
        "--diameter",
        "-d",
        type=float,
        default=0.020,
        help="Channel inner diameter in meters (SI, default: 0.020 m = 20 mm)",
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
        default=6.7e-5,
        help="Pixel size in m/px (SI, default: 6.7e-5)",
    )
    p_tr.add_argument(
        "--max-disp",
        type=float,
        default=50.0,
        help="Max search displacement in pixels (default: 50 px)",
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
        default=6.7e-5,
        help="Pixel size in m/px (SI, default: 6.7e-5)",
    )
    p_dfd.add_argument(
        "--beta-eff",
        type=float,
        default=2000.0,
        help="Effective DFD calibration in px/m",
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
    p_rend.add_argument("--pixel-size", type=float, default=6.7e-5, help="Pixel size in m/px")
    p_rend.add_argument("--diameter", type=float, default=0.020, help="Pipe diameter in meters")

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

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    import importlib

    if args.command == "void-fraction":
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
        )

    elif args.command == "track":
        mod = importlib.import_module("model_training.5_track_bubbles_kinematics")
        mod.run_tracking(
            args.masks,
            args.output_dir,
            args.fps,
            args.pixel_size,
            args.max_disp,
        )

    elif args.command == "dfd":
        mod = importlib.import_module("model_training.6_depth_from_defocus")
        mod.run_dfd_pipeline(
            args.images,
            args.masks,
            args.output_csv,
            pixel_size=args.pixel_size,
            beta_eff=args.beta_eff,
            background_path=args.background,
        )

    elif args.command == "render":
        import cv2

        mod = importlib.import_module("model_training.render_3d_flow")
        mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not load mask at: {args.mask}")
        recon = mod.reconstruct_3d_bubbles_si(
            mask, pixel_size_m=args.pixel_size, pipe_dia_m=args.diameter
        )
        mod.render_3d_flow_visualization(recon, args.output_png)
        mod.render_interactive_html(recon, args.output_html)
        print(f"[DONE] Rendered 3D flow to {args.output_png} and {args.output_html}")

    elif args.command == "gen-masks":
        mod = importlib.import_module("image_scripts.2_generate_masks")
        mod.generate_masks(args.input, args.output, min_area=args.min_area, max_area=args.max_area)

    elif args.command == "bg-sub":
        mod = importlib.import_module("image_scripts.1_background_subtraction")
        mod.run_batch_background_removal(args.input, args.output)


if __name__ == "__main__":
    main()
