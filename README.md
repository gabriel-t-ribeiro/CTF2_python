# CTF2 Python: Two-Phase Flow Boiling Image Analysis & 3D Reconstruction Pipeline

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Units: SI](https://img.shields.io/badge/units-standard%20SI-success.svg)](#8-engineering-standards--si-unit-conventions)

An automated computer-vision and quantitative physics pipeline for high-speed imaging of two-phase subcooled flow
boiling in conduits. Designed for mechanical and chemical engineers, the pipeline translates high-framerate shadowgraphy
footage (typically 2000 fps) into volumetric void fraction ($\alpha$), bubble velocity fields ($\vec{v}$), hydrodynamic
breakup rates, and 3D spatial bubble distributions $(x, y, z)$ using single-camera Depth from Defocus (DFD).

---

## 1. Physical Context & Engineering Motivation

Accurate void fraction ($\alpha$) and interfacial area concentration are foundational parameters in thermal-hydraulic
modeling of nuclear reactor cooling channels, compact heat exchangers, and boiling refrigeration loops. In subcooled
flow boiling, vapor bubble dynamics govern wall heat transfer, departure from nucleate boiling (DNB), and flow regime
transitions (bubbly $\to$ slug $\to$ annular).

This repository addresses key experimental optical challenges in high-speed flow boiling:

1. **Optical Refraction & Internal Glare**: Bubbles act as divergent spherical lenses, producing dark shadow boundaries
   around bright internal glare cores. The pipeline prevents threshold-induced core erosion.
2. **Defocus Blur**: Bubbles displaced from the focal plane exhibit blurred interfaces. Rather than discarding
   out-of-focus bubbles, this pipeline uses Depth from Defocus (DFD) to quantify bubble depth ($z$) and in-focus
   diameter ($d_0$).
3. **Dual-Regime 3D Volume Reconstruction**: Small spherical/ellipsoidal bubbles are reconstructed as triaxial
   ellipsoids bounded by channel walls; large vapor slugs are reconstructed using distance-transform-based Laplacian
   extrusion constrained by the channel chord depth (Eduardo Reis, 2025).
4. **Hydrodynamic Breakup Decoupling**: High-framerate tracking at 2000 FPS decouples stable steady translation from
   turbulent shear breakup via optimal bipartite assignment and projected area conservation.
5. **Analytical Validation**: Experimental void fraction is directly validated against the Rouhani & Axelsson (1970)
   drift-flux theoretical correlation for boiling water.

---

## 2. Pipeline Architecture

```text
Raw High-Speed Shadowgraphy (2000 FPS)
  │
  ├──> Stage 0: Optical Preprocessing (image_scripts/)
  │      ├─ 0_roi_crop.py               --> Fluid conduit isolation (headless/GUI)
  │      ├─ 1_background_subtraction.py  --> Temporal median removal protecting glare & blur
  │      ├─ 2_generate_masks.py          --> Otsu dynamic thresholding & component filtering
  │      └─ 3_create_preannotations.py   --> Label Studio polygon serialization (Base64)
  │
  ├──> Stage 1: Machine Learning Segmentation (model_training/)
  │      ├─ 1_convert_annotations.py     --> Label Studio export to training pairs
  │      ├─ 2_train_u_net.py             --> ResNet-34 U-Net (Dice + BCE loss, MPS/CUDA)
  │      ├─ evaluate_unet_metrics.py     --> IoU, Dice, MCC, Precision, Recall evaluation
  │      └─ 3_batch_inference.py         --> Sequence inference producing binary masks
  │
  ├──> Stage 2: Quantitative Physics & 3D Reconstruction (model_training/)
  │      ├─ 4_void_fraction_reconstruct.py --> Dual-regime volume & Rouhani-Axelsson validation
  │      ├─ 5_track_bubbles_kinematics.py  --> 2000 FPS velocity & breakup tracking
  │      ├─ 6_depth_from_defocus.py        --> Optical depth & in-focus sizing (Rao et al.)
  │      └─ render_3d_flow.py              --> Publication 3D figures & Three.js WebGL HTML
  │
  └──> Unified CLI Orchestrator: main.py
```

---

## 3. Environment Setup & Prerequisites

The codebase is built for Python 3.14+ and leverages PyTorch with hardware acceleration (`mps` on macOS Apple Silicon,
`cuda` on Linux/NVIDIA, or `cpu`).

### 3.1 Activate the Virtual Environment

```bash
# Clone the repository and enter the workspace
cd CTF2_python

# Activate the local virtual environment
source .venv/bin/activate

# Verify Python version (>= 3.14)
python --version
```

Expected terminal output:

```text
Python 3.14.0
```

### 3.2 Fresh Installation (if configuring a new machine)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install opencv-python numpy pandas scipy matplotlib pillow albumentations segmentation-models-pytorch torch ruff pre-commit
```

---

## 4. Verification & Automated Test Suite

Run the comprehensive unit test suite to verify physical units, dimensional consistency, mathematical model bounds, and
optical inversion routines:

```bash
python tests/test_3d_render_and_si.py
```

Expected terminal output:

```text
Running comprehensive CTF2 verification tests...
  [PASS] test_si_unit_scales_and_dimensions
  [PASS] test_nan_robustness_and_ellipse_axis_sorting
  [PASS] test_dual_regime_large_slug_confinement
  [PASS] test_rouhani_axelsson_drift_flux_water_si
  [PASS] test_3d_render_generation
Tracking bubble kinematics at 2000.0 FPS across 2 frames in SI units...
  Temporal resolution (dt)  : 5.000000e-04 s (0.500 ms)
  Pixel size (s_px)         : 8.600000e-05 m/px
  Max search displacement   : 4.300000e-03 m (50.0 px)

====================================================
     KINEMATICS & BREAKUP ANALYSIS SUMMARY (SI)
====================================================
Total Trajectory Links (Stable Pairs): 2
Mean Velocity (Stable Pairs)        : 2.5800 m/s
Max Velocity (Stable Pairs)         : 2.5800 m/s
Mean Bubble Displacement            : 1.290000e-03 m
Total Breakup Events Detected       : 0
====================================================
[DONE] Saved velocity logs (SI) to: results/bubble_velocities_si.csv
[DONE] Saved breakup logs (SI) to : results/bubble_breakups_si.csv
  [PASS] test_kinematic_tracking_optimal_bipartite_matching
  [PASS] test_depth_from_defocus_inversion

ALL 7 TESTS PASSED SUCCESSFULLY!
```

---

## 5. Quickstart: Unified CLI Orchestration (`main.py`)

The root `main.py` entrypoint provides a single CLI dispatcher for all core pipeline stages.

### 5.1 Subcommand Reference

```bash
python main.py --help
```

```text
usage: main.py [-h] {void-fraction,track,dfd,render,calibrate,gen-masks,bg-sub} ...

CTF2 Flow Boiling Analysis & 3D Reconstruction Pipeline

positional arguments:
  {void-fraction,track,dfd,render,calibrate,gen-masks,bg-sub}
                        Available pipeline stages
    void-fraction       Compute 3D gas volumes and void fraction
    track               Track bubble kinematics and breakup events at 2000 FPS
    dfd                 Perform single-camera Depth from Defocus (DFD) analysis
    render              Render 3D flow structure and interactive HTML
    calibrate           Interactively calibrate optical pixel size (m/px) using 2 points
    gen-masks           Generate initial binary masks using Otsu thresholding
    bg-sub              Perform background subtraction preserving defocus margins
```

---

### 5.2 Execute Void Fraction Reconstruction & Benchmark

Calculates frame-by-frame 3D gas volumes, experimental void fraction $\alpha_{\text{exp}}$, and compares against the
Rouhani-Axelsson model:

```bash
python main.py void-fraction \
  --masks dataset/masks \
  --output-csv results/void_fraction.csv \
  --pixel-size 8.6e-5 \
  --diameter 0.020 \
  --vapor-quality 0.05 \
  --mass-flux 500.0
```

Expected terminal output:

```text
Executing 3D void fraction reconstruction in standard SI units:
  Masks count         : 11
  Pixel size (SI)     : 8.600000e-05 m/px (0.0860 mm/px)
  Channel diameter (SI): 2.000000e-02 m (20.00 mm)
  Channel radius (SI) : 1.000000e-02 m (116.3 px)

====================================================
      VOID FRACTION RECONSTRUCTION SUMMARY (SI)
====================================================
Frames Analyzed     : 11
Mean Alpha (alpha_exp): 0.3027 (30.266%)
Min Alpha           : 0.2057
Max Alpha           : 0.3398
Standard Dev        : 0.0402
Mean Gas Volume     : 7.285777e-06 m³
Mean Pipe Volume    : 2.407277e-05 m³
----------------------------------------------------
     ROUHANI-AXELSSON THEORETICAL BENCHMARK (SI)
----------------------------------------------------
Vapor Quality (x)   : 0.0500 [-]
Mass Flux (G)       : 500.0 kg/(m² s)
Theoretical Void (ε): 0.8839 (88.39%)
Relative Difference : 65.76 %
====================================================
Results saved to    : results/void_fraction.csv
```

---

### 5.3 Bubble Kinematics & Breakup Tracking

Tracks bubble translation velocities and logs hydrodynamic fragmentation events:

```bash
python main.py track \
  --masks dataset/masks \
  --output-dir results/kinematics \
  --fps 2000 \
  --pixel-size 8.6e-5 \
  --max-disp 50
```

Expected terminal output:

```text
Tracking bubble kinematics at 2000.0 FPS across 11 frames in SI units...
  Temporal resolution (dt)  : 5.000000e-04 s (0.500 ms)
  Pixel size (s_px)         : 8.600000e-05 m/px
  Max search displacement   : 4.300000e-03 m (50.0 px)

====================================================
     KINEMATICS & BREAKUP ANALYSIS SUMMARY (SI)
====================================================
Total Trajectory Links (All)        : 125
Mean Velocity (All)                 : 2.9193 m/s
Mean Bubble Displacement            : 1.459633e-03 m
Total Breakup Events Detected       : 20
====================================================
[DONE] Saved velocity logs (SI) to: results/kinematics/bubble_velocities_si.csv
[DONE] Saved breakup logs (SI) to : results/kinematics/bubble_breakups_si.csv
```

---

### 5.4 Single-Camera Depth from Defocus (DFD)

Evaluates optical boundary gradient softening to extract 3D coordinates $(x, y, z)$ and in-focus diameters $d_0$:

```bash
python main.py dfd \
  --images dataset/images \
  --masks dataset/masks \
  --output-csv results/dfd_results.csv \
  --pixel-size 8.6e-5 \
  --beta-eff 2000.0
```

Expected terminal output:

```text
Executing Depth from Defocus (DFD) analysis across 11 frames:
  Pixel size (SI)     : 8.600000e-05 m/px
  Beta calibration    : 2000.00 px/m

====================================================
      DEPTH FROM DEFOCUS (DFD) SUMMARY (SI)
====================================================
Total Bubbles Sized : 173
Mean Diameter (d0)  : 4.767814e-03 m (4.768 mm)
Mean Depth (|z|)    : 8.760282e-03 m (8.760 mm)
Max Depth (|z|)     : 1.268535e-01 m (126.853 mm)
Mean Blur (sigma)   : 17.52 px
====================================================
[DONE] Saved DFD results to: results/dfd_results.csv
```

---

### 5.5 3D Flow Visualization & WebGL HTML

Renders high-resolution publication figures and standalone interactive 3D WebGL HTML:

```bash
python main.py render \
  --mask dataset/masks/frame_0016.png \
  --output-png results/3d_render.png \
  --output-html results/3d_render.html \
  --pixel-size 8.6e-5 \
  --diameter 0.020
```

Expected terminal output:

```text
[DONE] Rendered 3D flow to results/3d_render.png and results/3d_render.html
```

---

### 5.6 Interactive Optical Scale Calibration

Foolproof 2-point interactive calibration for optical pixel scale ($s_{\text{px}}$ in $\text{m/px}$) and inner tube
diameter:

```bash
# In millimeters (recommended):
python main.py calibrate --input dataset/images -smm 20

# Export calibration to both images and masks folders simultaneously:
python main.py calibrate --input dataset/images -smm 20 -c dataset/masks

# Or in meters:
python main.py calibrate --input dataset/images -s 0.020
```

1. **Toolbar Controls**: Use the standard Matplotlib magnifier/hand tools to zoom and pan with pixel accuracy.
2. **Point Selection**: Left-click to place Point 1 and Point 2 across the tube diameter.
3. **Live Angle & Scale Preview**: The window title immediately displays Euclidean distance ($L_{\text{px}}$),
   calculated pixel scale ($\text{mm/px}$, $\text{m/px}$), optical scale ($\text{px/mm}$), line angle ($\theta^\circ$),
   and displacements ($\Delta x$, $\Delta y$).
4. **Drag & Arrow Nudging**:
    - **Direct Drag**: Click and drag either point with the mouse to adjust its position.
    - **Arrow Keys**: Press `Up` / `Down` / `Left` / `Right` to nudge the active point by 1 pixel.
    - **Subpixel Fine Tuning**: Hold `Shift + Arrow keys` to nudge by **0.2 px** for pinpoint wall edge placement.
    - **Switch Active Point**: Press `[Tab]` (or `[1]` / `[2]`) to switch which point moves. The active point is
      highlighted with a yellow circle.
5. **Alignment Snapping**: Press `[V]` to snap Point 2 to a **perfectly vertical** line ($x_2 = x_1$, $\Delta x = 0$),
   `[H]` to snap to a **perfectly horizontal** line ($y_2 = y_1$, $\Delta y = 0$), or `[U]` to undo snap.
6. **Undo / Adjust**: Right-click to undo and re-pick if necessary.
7. **Confirmation**: Press `[ENTER]` or close the window.
8. **Report Generation**: Automatically saves `0_<folder_name>_calibration.txt` (e.g. `0_images_calibration.txt`) inside
   the target directory. The `0_` prefix ensures the file always stays sorted right at the very top of the directory
   tree in PyCharm and file explorers.
9. **Dual-Folder Export (`-c` / `--copy-to`)**: Automatically exports `0_<dest_folder>_calibration.txt` (e.g.
   `0_masks_calibration.txt`) to secondary folders such as `dataset/masks`. In interactive mode, prompts with detected
   sibling folders.
10. **Pipeline Auto-Detection**: Subsequent pipeline stages (`void-fraction`, `track`, `dfd`, `render`) automatically
    detect and load `pixel_size_m` and `diameter_m` from this file if `--pixel-size` is omitted!

---

### 5.7 Conduit Isolation & ROI Cropping with Calibration Collision Detection

Isolate the channel fluid domain interactively or with explicit ROI bounds. If existing calibration file(s) are detected
on the images being cropped (or in the output destination), the tool prompts with 3 safe actions:

```bash
# Interactive ROI crop via unified CLI:
python main.py crop --input dataset/raw_images --output dataset/cropped

# Or with explicit bounds (skip GUI):
python main.py crop --input dataset/raw_images --output dataset/cropped --roi 20,45,800,240

# Non-interactive calibration handling via --on-calibration {delete,keep,redo}:
python main.py crop --input dataset/raw_images --output dataset/cropped --roi 20,45,800,240 --on-calibration redo
```

**Interactive Calibration Options when Collision is Detected**:

- **`[1] Delete calibration (default)`**: Deletes the outdated calibration file(s) since cropping renders previous
  physical scales and coordinates stale.
- **`[2] Do nothing`**: Keeps the existing calibration file intact without alteration.
- **`[3] Redo calibration`**: Recalls and displays the previous calibration parameters (diameter, pixel scale, measured
  distance) and prompts you for how you want to call the calibration on the cropped images (e.g.
  `--size-mm 20 --copy-to dataset/masks`). Pressing `[ENTER]` reuses the recalled diameter automatically!

---

## 6. End-to-End Modular Workflow

For in-depth customization and script-level tuning, each pipeline stage can also be executed modularly.

```bash
# 1. Optical Preprocessing (Cropping & Glare-Preserving Background Subtraction)
python image_scripts/0_roi_crop.py --input images/1st_run/images_1 --output results/cropped --roi 10,10,600,200
python image_scripts/1_background_subtraction.py --input results/cropped --output results/cleaned --sample-size 50 --threshold 15 --margin 10
python image_scripts/2_generate_masks.py --input results/cleaned --output results/masks --min-area 5 --max-area 50000
python image_scripts/3_create_preannotations.py --images-dir results/cleaned --masks-dir results/masks --output-json results/preannotations.json

# 2. Convert Verified Label Studio JSON to Training Pairs
python model_training/1_convert_annotations.py --json model_training/project_export.json --dataset dataset

# 3. Train ResNet-34 U-Net Segmentation Model
python model_training/2_train_u_net.py --dataset dataset --output-model models/run_1_unet.pth --epochs 20 --batch-size 4 --lr 1e-3

# 4. Evaluate Validation Performance
python model_training/evaluate_unet_metrics.py --model models/run_1_unet.pth --dataset dataset --output-csv results/validation_metrics.csv

# 5. Batch Inference Across Experimental Sequences
python model_training/3_batch_inference.py --input dataset/images --output results/inferred_masks --model models/run_1_unet.pth

# 6. Physical Analysis & 3D Reconstruction
python model_training/4_2d_void_fraction_reconstruct.py --masks results/inferred_masks --output-csv results/void_fraction.csv --pixel-size 8.6e-5 --diameter 0.020 --vapor-quality 0.05 --mass-flux 500.0 --render
python model_training/5_track_bubbles_kinematics.py --masks results/inferred_masks --output-dir results/kinematics --fps 2000 --pixel-size 8.6e-5 --max-disp 50
python model_training/6_depth_from_defocus.py --images dataset/images --masks results/inferred_masks --output-csv results/dfd.csv --pixel-size 8.6e-5 --beta-eff 2000.0
python model_training/render_3d_flow.py --mask results/inferred_masks/frame_0016_mask.png --output results/render.png --interactive-html results/render.html --pixel-size 8.6e-5 --diameter 0.020
```

Refer to the sub-package documentation for advanced flags and implementation details:

- [Image Preparation Guide](image_scripts/README.md)
- [Model Training & Physics Guide](model_training/README.md)

---

## 7. Repository Layout

```text
.
├── main.py                         # Unified CLI pipeline orchestrator
├── pyproject.toml                  # Project metadata, Ruff & linter rules
├── .pre-commit-config.yaml         # Git pre-commit hook specifications
│
├── image_scripts/                  # Stage 0: Optical image preparation
│   ├── 0_roi_crop.py               # Conduit ROI cropping (headless/GUI)
│   ├── 1_background_subtraction.py # Temporal median background subtraction
│   ├── 2_generate_masks.py         # Dynamic Otsu binary mask generator
│   ├── 3_create_preannotations.py  # Label Studio polygon pre-annotation exporter
│   ├── test_single_mask.py         # Single-frame diagnostic mask visualizer
│   └── README.md                   # Image processing documentation
│
├── model_training/                 # Stages 1 & 2: Deep learning & physics
│   ├── 1_convert_annotations.py    # Converts Label Studio JSON to image/mask pairs
│   ├── 2_train_u_net.py            # Trains ResNet-34 U-Net segmentation network
│   ├── 3_batch_inference.py        # Batch neural inference across image folders
│   ├── 4_void_fraction_reconstruct.py # 3D volume reconstruction & Rouhani-Axelsson model
│   ├── 5_track_bubbles_kinematics.py  # 2000 FPS bubble velocity & breakup tracking
│   ├── 6_depth_from_defocus.py     # Single-camera DFD depth & sizing analysis
│   ├── render_3d_flow.py           # 3D Matplotlib visualizer & Three.js WebGL generator
│   ├── evaluate_unet_metrics.py    # Validation metrics reporting (Dice, IoU, MCC)
│   ├── project_export.json         # Sample verified Label Studio ground truth
│   └── README.md                   # Model training documentation
│
├── dataset/                        # Sample paired training dataset
│   ├── images/                     # Grayscale source frames
│   └── masks/                      # Binary ground-truth masks
│
├── models/                         # Trained model weights & checkpoints
│   └── run_1_unet.pth              # Pretrained U-Net weights
│
├── tests/                          # Automated verification suite
│   └── test_3d_render_and_si.py    # 7 unit tests (physics, SI units, models)
│
└── results/                        # Generated analysis CSVs and visualizations (untracked)
```

---

## 8. Engineering Standards & SI Unit Conventions

All physical quantities in the codebase are calculated and saved in standard SI units:

| Physical Quantity | Symbol               | SI Unit                               | Code Flag / Variable           | Typical Range                       | Notes                                                            |
|:------------------|:---------------------|:--------------------------------------|:-------------------------------|:------------------------------------|:-----------------------------------------------------------------|
| Optical Scale     | $s_{\text{px}}$      | $\text{m/px}$                         | `--pixel-size`, `pixel_size_m` | $5\times 10^{-5} - 1\times 10^{-4}$ | E.g. $8.6\times 10^{-5}\text{ m/px} = 0.086\text{ mm/px}$.       |
| Conduit Diameter  | $D$                  | $\text{m}$                            | `--diameter`, `diameter_m`     | $0.010 - 0.050$                     | Channel inner diameter (e.g. $0.020\text{ m} = 20\text{ mm}$).   |
| Gas Volume        | $V_{\text{gas}}$     | $\text{m}^3$                          | `gas_volume_m3`                | $10^{-7} - 10^{-4}$                 | Integrated 3D gas volume.                                        |
| Pipe Volume       | $V_{\text{pipe}}$    | $\text{m}^3$                          | `pipe_volume_m3`               | $10^{-6} - 10^{-3}$                 | Visualized section volume $\pi R^2 L$.                           |
| Void Fraction     | $\alpha$             | dimensionless                         | `void_fraction_alpha`          | $0.0 - 1.0$                         | Volumetric gas fraction $\sum V_{\text{gas}} / V_{\text{pipe}}$. |
| Vapor Quality     | $x$                  | dimensionless                         | `--vapor-quality`, `x`         | $0.0 - 1.0$                         | Mass fraction of vapor in two-phase mixture.                     |
| Mass Flux         | $G$                  | $\text{kg}/(\text{m}^2\cdot\text{s})$ | `--mass-flux`, `mass_flux_G`   | $100 - 2000$                        | Total mass flow rate per unit cross-section.                     |
| Bubble Velocity   | $v$                  | $\text{m/s}$                          | `velocity_m_s`                 | $0.5 - 15.0$                        | Inter-frame translation $\Delta r / \Delta t$.                   |
| Framing Rate      | $f$                  | $\text{Hz}$ ($1/\text{s}$)            | `--fps`, `fps`                 | $1000 - 10000$                      | Frame interval $\Delta t = 1/f$.                                 |
| Defocus Parameter | $\beta_{\text{eff}}$ | $\text{px/m}$                         | `--beta-eff`, `beta_eff`       | $1000 - 5000$                       | Optical blur growth rate $\sigma /                               |\Delta z|$. |
| Surface Tension   | $\sigma$             | $\text{N/m}$                          | `sigma`                        | $0.0589$                            | Water/steam at saturation (100°C).                               |
| Liquid Density    | $\rho_l$             | $\text{kg/m}^3$                       | `rho_l`                        | $958.4$                             | Saturated liquid water at 100°C.                                 |
| Vapor Density     | $\rho_v$             | $\text{kg/m}^3$                       | `rho_v`                        | $0.597$                             | Saturated steam at 100°C.                                        |

> **Automated Conversion Guardrail**: In both `4_void_fraction_reconstruct.py` and `render_3d_flow.py`, if a user
> provides convenience values in millimeters (e.g., `--diameter 20.0` or `--pixel-size 0.086`), the parser automatically
> detects and converts inputs $> 0.001$ or $> 0.05$ into standard SI meters ($10^{-3}$ factor).

---

## 9. Code Quality & Pre-Commit Standards

The repository enforces formatting and linting via **Ruff**:

```bash
# Check for lint issues and automatically fix safe patterns
ruff check . --fix

# Format code to project standards
ruff format .

# Run pre-commit hooks across all files
pre-commit run --all-files
```

To install automated git pre-commit hooks:

```bash
pre-commit install
```
