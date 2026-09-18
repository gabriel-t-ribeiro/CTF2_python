# CTF2 Python: Two-Phase Flow Boiling Image Analysis & 3D Reconstruction Pipeline

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Units: SI](https://img.shields.io/badge/units-standard%20SI-success.svg)](#9-engineering-standards--si-unit-conventions)

An automated computer-vision and quantitative physics pipeline for high-speed imaging of two-phase subcooled flow
boiling in conduits. Designed for mechanical and chemical engineers, the pipeline translates high-framerate shadowgraphy
footage (typically 2000 fps) into volumetric void fraction ($\alpha$), temporal void fraction evolution ($\alpha(t)$),
bubble velocity fields ($\vec{v}$), hydrodynamic breakup rates, and 3D spatial bubble distributions $(x, y, z)$ using
single-camera Depth from Defocus (DFD).

---

## Visual Showcase

| 3D Multi-Regime Flow Reconstruction | Void Fraction Timeline Evolution $\alpha(t)$ |
| :---: | :---: |
| ![3D Flow Visualization](docs/assets/3d_bubble_render.png) | ![Void Fraction Timeline](docs/assets/void_fraction_timeline.png) |

| High-Speed Velocity Field Animation (2000 FPS) | Multi-Method Benchmark Comparison |
| :---: | :---: |
| ![High-Speed Bubble Velocity Animation](docs/assets/bubble_velocity_animation.gif) | ![Void Fraction Comparative Overlay](docs/assets/void_fraction_comparison.png) |

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
6. **Temporal Evolution & Interactive Dashboards**: Translates frame sequences into physical time axes ($t$ [ms]),
   computing rolling averages, statistical dispersion envelopes ($\pm 1\sigma$), and standalone interactive WebGL/Chart.js
   dashboards.

---

## 2. Pipeline Architecture

```text
Raw High-Speed Shadowgraphy (2000 FPS)
  │
  ├──> Stage 0: Optical Preprocessing (image_scripts/)
  │      ├─ calibrate_pixel_size.py     --> Optical magnification calibration (m/px)
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
  │      ├─ plot_void_fraction_timeline.py --> Publication timeline figures & interactive HTML
  │      ├─ 5_track_bubbles_kinematics.py  --> 2000 FPS velocity & breakup tracking
  │      ├─ visualize_flow_animation.py    --> Slow-motion velocity vector animations (MP4/GIF)
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

### 3.2 Fresh Installation (if configuring a new machine)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install opencv-python numpy pandas scipy matplotlib pillow albumentations segmentation-models-pytorch torch ruff pre-commit
```

---

## 4. Verification & Automated Test Suite

Run the comprehensive unit test suite (40 unit tests) covering optical calibration, ROI cropping, 3D volumetric math,
Rouhani-Axelsson drift-flux, kinematic tracking, velocity vector calculation, and void fraction timeline generation:

```bash
export MPLCONFIGDIR=/tmp/mpl
python -m unittest discover tests
```

Expected terminal output:
```text
Ran 40 tests in 1.795s

OK
```

---

## 5. Quickstart: Unified CLI Orchestration (`main.py`)

The root `main.py` entrypoint provides a single unified CLI dispatcher for all pipeline stages.

### 5.1 Subcommand Reference

```bash
python main.py --help
```

| Subcommand | Aliases | Description |
|:---|:---|:---|
| `calibrate` | — | Interactively calibrate optical pixel size ($\text{m/px}$) and conduit diameter using 2 points |
| `crop` | — | Interactive or headless ROI conduit cropping with stale calibration collision detection |
| `bg-sub` | — | Temporal median background subtraction preserving interface blur margins |
| `gen-masks` | — | Generate binary segmentation masks using dynamic Otsu thresholding |
| `void-fraction` | — | Reconstruct 3D gas volumes, compute void fraction CSV, and benchmark with Rouhani-Axelsson |
| `plot-vf` | `plot-void-fraction`, `plot-timeline` | Generate publication timeline figure (PNG/SVG) and interactive HTML dashboard |
| `track` | — | Track bubble kinematics and hydrodynamic breakup events at 2000 FPS |
| `animate-flow` | `visualize-tracking` | Render slow-motion velocity vector field animation (MP4, animated GIF, HTML player) |
| `dfd` | — | Single-camera Depth from Defocus (DFD) 3D coordinate and in-focus sizing analysis |
| `render` | — | Render publication-grade 3D flow visualization and standalone Three.js WebGL HTML |

---

### 5.2 Master End-to-End Workflow Recipes (`Bash 1 > Bash 2 > Bash 3`)

Below are complete runnable command sequences taking raw camera footage through each analytical workflow. Standard placeholder directories (`path/to/...`) can be replaced with your dataset paths (e.g. `dataset/images`, `dataset/masks`).

#### Recipe 1: Raw Footage $\to$ 3D Void Fraction $\to$ Publication Timeline & HTML Dashboard

```bash
# Bash 1: Crop fluid conduit from raw camera frames
python main.py crop -i path/to/raw_images -o results/cropped_frames --roi 10,10,891,328
# (Alt: python image_scripts/0_roi_crop.py -i path/to/raw_images -o results/cropped_frames --roi 10,10,891,328)

# Bash 2: Generate binary segmentation masks (Otsu dynamic thresholding)
python main.py gen-masks -i results/cropped_frames -o results/segmented_masks --min-area 5
# (Alt: python image_scripts/2_generate_masks.py -i results/cropped_frames -o results/segmented_masks --min-area 5)

# Bash 3: Reconstruct 3D volumes and calculate void fraction CSV in SI units
python main.py void-fraction \
  --masks results/segmented_masks \
  -o results/void_fraction.csv \
  --pixel-size 6.1e-5 \
  --diameter 0.020 \
  -x 0.05 -G 400.0 \
  --render
# (Alt: python model_training/4_void_fraction_reconstruct.py --masks results/segmented_masks -o results/void_fraction.csv --pixel-size 6.1e-5 --diameter 0.020 -x 0.05 -G 400.0 --render)

# Bash 4: Generate publication-grade timeline plot (PNG) and interactive HTML dashboard
python main.py plot-vf \
  -i results/void_fraction.csv \
  -o results/void_fraction_timeline.png \
  --html results/void_fraction_timeline.html \
  --fps 2000 \
  --multi-panel \
  -x 0.05 -G 400.0
# (Alt: python model_training/plot_void_fraction_timeline.py -i results/void_fraction.csv -o results/void_fraction_timeline.png --html results/void_fraction_timeline.html --fps 2000 --multi-panel -x 0.05 -G 400.0)
```

> [!TIP]
> **One-Step Shortcut**: Combine Bash 3 and Bash 4 by adding `--plot-timeline` to the `void-fraction` command:
> ```bash
> python main.py void-fraction --masks results/segmented_masks -o results/void_fraction.csv --plot-timeline -x 0.05 -G 400.0
> ```

---

#### Recipe 2: High-Speed Bubble Velocity Field Animation & Breakup Tracking (2000 FPS)

```bash
# Bash 1: Compute bubble velocities and detect hydrodynamic breakup events
python main.py track \
  --masks results/segmented_masks \
  -o results/kinematics \
  --fps 2000 \
  --pixel-size 6.1e-5 \
  --max-disp 50
# (Alt: python model_training/5_track_bubbles_kinematics.py --masks results/segmented_masks -o results/kinematics --fps 2000 --pixel-size 6.1e-5 --max-disp 50)

# Bash 2: Render slow-motion velocity vector field animation (MP4, animated GIF, HTML player)
python main.py animate-flow \
  --masks results/segmented_masks \
  -i results/cropped_frames \
  -o results/flow_animation \
  --fps 2000 \
  --playback-fps 5.0
# (Alt: python model_training/visualize_flow_animation.py --mask-dir results/segmented_masks --image-dir results/cropped_frames --output-dir results/flow_animation --fps 2000 --playback-fps 5.0)
```

---

#### Recipe 3: Multi-Method Void Fraction Comparison (Python Dual-Regime vs. MATLAB Voxel)

```bash
# Bash 1: Run Python 3D void fraction reconstruction
python main.py void-fraction --masks dataset/masks -o results/void_fraction_python.csv

# Bash 2: Plot comparative overlay on a shared physical time axis
python main.py plot-vf \
  -i results/void_fraction_python.csv matlab_image_processing_final/matlab_void_fraction_masks.csv \
  --labels "Python Dual-Regime (SI)" "MATLAB Voxel Method" \
  -o results/void_fraction_comparison.png \
  --html results/void_fraction_comparison.html
```

---

#### Recipe 4: Single-Camera Depth from Defocus (DFD) 3D Spatial Localization

```bash
# Bash 1: Extract 3D out-of-plane depth (z) and in-focus diameter (d0)
python main.py dfd \
  --images results/cropped_frames \
  --masks results/segmented_masks \
  -o results/dfd_results.csv \
  --pixel-size 6.1e-5 \
  --beta-eff 2000.0
# (Alt: python model_training/6_depth_from_defocus.py --images results/cropped_frames --masks results/segmented_masks -o results/dfd_results.csv --pixel-size 6.1e-5 --beta-eff 2000.0)
```

---

### 5.3 Detailed Stage Invocations

#### A. 3D Void Fraction Reconstruction (`void-fraction`)

```bash
python main.py void-fraction \
  --masks dataset/masks \
  --output-csv results/void_fraction.csv \
  --pixel-size 6.1e-5 \
  --diameter 0.020 \
  --vapor-quality 0.05 \
  --mass-flux 400.0 \
  --render
```

#### B. Void Fraction Timeline Graph & Interactive Dashboard (`plot-vf`)

```bash
python main.py plot-vf \
  -i results/void_fraction.csv \
  -o results/void_fraction_timeline.png \
  --html results/void_fraction_timeline.html \
  --fps 2000 \
  --multi-panel \
  -x 0.05 -G 400.0
```

#### C. High-Speed Kinematics Tracking (`track`)

```bash
python main.py track \
  --masks dataset/masks \
  --output-dir results/kinematics \
  --fps 2000 \
  --pixel-size 6.1e-5 \
  --max-disp 50
```

#### D. Interactive Optical Scale Calibration (`calibrate`)

```bash
# In millimeters (recommended):
python main.py calibrate --input dataset/images -smm 20

# Export calibration to both images and masks folders simultaneously:
python main.py calibrate --input dataset/images -smm 20 -c dataset/masks
```

#### E. Conduit Isolation & ROI Cropping (`crop`)

```bash
# Interactive ROI crop:
python main.py crop --input path/to/raw_images --output results/cropped

# Headless batch crop (skip GUI):
python main.py crop --input path/to/raw_images --output results/cropped --roi 10,10,891,328
```

---

## 6. Human-in-the-Loop Orchestration (`pipeline.py`)

A master orchestrator (`pipeline.py`) is provided for human-in-the-loop (HITL) workflows, enabling rapid iteration of
U-Net models by running inference, exporting to Label Studio for human verification, and retraining.

### 6.1 Prepare Dataset (Inference + Label Studio Export)

```bash
python pipeline.py prepare \
  --images-dir dataset/images \
  --model models/run_1_unet.pth \
  --output-json results/hitl_preannotations.json
```

### 6.2 Train on Human-Corrected Annotations

```bash
python pipeline.py train \
  --export-json results/human_corrected.json \
  --dataset-dir dataset/updated_dataset \
  --model-out models/run_2_unet.pth \
  --epochs 40 \
  --batch-size 8 \
  --lr 0.001
```

---

## 7. Sub-Module Documentation

For in-depth customization, mathematical derivations, and script-level tuning, refer to the dedicated module guides:

- [Image Preparation & Optical Calibration Guide](image_scripts/README.md)
- [Model Training, Physics & Timeline Guide](model_training/README.md)

---

## 8. Repository Layout

```text
.
├── main.py                         # Unified CLI pipeline orchestrator
├── pipeline.py                     # Human-in-the-loop (HITL) retraining orchestrator
├── pyproject.toml                  # Project metadata, Ruff & linter rules
├── .pre-commit-config.yaml         # Git pre-commit hook specifications
│
├── docs/                           # Documentation and media assets
│   └── assets/                     # High-resolution showcase images & GIFs
│       ├── 3d_bubble_render.png
│       ├── void_fraction_timeline.png
│       ├── void_fraction_comparison.png
│       ├── bubble_velocity_animation.gif
│       ├── sample_raw_frame.png
│       └── sample_mask.png
│
├── image_scripts/                  # Stage 0: Optical image preparation
│   ├── calibrate_pixel_size.py     # 2-point interactive optical scale calibration
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
│   ├── plot_void_fraction_timeline.py # Publication timeline figures & interactive HTML
│   ├── 5_track_bubbles_kinematics.py  # 2000 FPS bubble velocity & breakup tracking
│   ├── visualize_flow_animation.py # 2000 FPS velocity vector animation generator
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
├── tests/                          # Automated verification suite (40 unit tests)
│   ├── test_3d_render_and_si.py    # Physics, SI units, 3D watertight mesh tests
│   ├── test_calibration.py         # Optical calibration and cropping tests
│   ├── test_flow_animation.py      # Bubble velocity vectors and animation tests
│   └── test_plot_void_fraction_timeline.py # Timeline plotting, HTML, and Rouhani-Axelsson tests
│
└── results/                        # Generated analysis CSVs, figures, and HTML viewers
```

---

## 9. Engineering Standards & SI Unit Conventions

All physical quantities in the codebase are calculated and saved in standard SI units:

| Physical Quantity | Symbol               | SI Unit                               | Code Flag / Variable           | Typical Range                       | Notes                                                            |
|:------------------|:---------------------|:--------------------------------------|:-------------------------------|:------------------------------------|:-----------------------------------------------------------------|
| Optical Scale     | $s_{\text{px}}$      | $\text{m/px}$                         | `--pixel-size`, `pixel_size_m` | $5\times 10^{-5} - 1\times 10^{-4}$ | E.g. $6.1\times 10^{-5}\text{ m/px} = 0.061\text{ mm/px}$.       |
| Conduit Diameter  | $D$                  | $\text{m}$                            | `--diameter`, `diameter_m`     | $0.010 - 0.050$                     | Channel inner diameter (e.g. $0.020\text{ m} = 20\text{ mm}$).   |
| Gas Volume        | $V_{\text{gas}}$     | $\text{m}^3$                          | `gas_volume_m3`                | $10^{-7} - 10^{-4}$                 | Integrated 3D gas volume.                                        |
| Pipe Volume       | $V_{\text{pipe}}$    | $\text{m}^3$                          | `pipe_volume_m3`               | $10^{-6} - 10^{-3}$                 | Visualized section volume $\pi R^2 L$.                           |
| Void Fraction     | $\alpha$             | dimensionless                         | `void_fraction_alpha`          | $0.0 - 1.0$                         | Volumetric gas fraction $\sum V_{\text{gas}} / V_{\text{pipe}}$. |
| Vapor Quality     | $x$                  | dimensionless                         | `--vapor-quality`, `x`         | $0.0 - 1.0$                         | Mass fraction of vapor in two-phase mixture.                     |
| Mass Flux         | $G$                  | $\text{kg}/(\text{m}^2\cdot\text{s})$ | `--mass-flux`, `mass_flux_G`   | $100 - 2000$                        | Total mass flow rate per unit cross-section.                     |
| Bubble Velocity   | $v$                  | $\text{m/s}$                          | `velocity_m_s`                 | $0.5 - 15.0$                        | Inter-frame translation $\Delta r / \Delta t$.                   |
| Framing Rate      | $f$                  | $\text{Hz}$ ($1/\text{s}$)            | `--fps`, `fps`                 | $1000 - 10000$                      | Frame interval $\Delta t = 1/f$.                                 |
| Defocus Parameter | $\beta_{\text{eff}}$ | $\text{px/m}$                         | `--beta-eff`, `beta_eff`       | $1000 - 5000$                       | Optical blur growth rate $\sigma / |\Delta z|$.                  |
| Surface Tension   | $\sigma$             | $\text{N/m}$                          | `sigma`                        | $0.0589$                            | Water/steam at saturation (100°C).                               |
| Liquid Density    | $\rho_l$             | $\text{kg/m}^3$                       | `rho_l`                        | $958.4$                             | Saturated liquid water at 100°C.                                 |
| Vapor Density     | $\rho_v$             | $\text{kg/m}^3$                       | `rho_v`                        | $0.597$                             | Saturated steam at 100°C.                                        |

> **Automated Conversion Guardrail**: In both `4_void_fraction_reconstruct.py` and `render_3d_flow.py`, if a user
> provides convenience values in millimeters (e.g., `--diameter 20.0` or `--pixel-size 0.061`), the parser automatically
> detects and converts inputs $> 0.001$ or $> 0.05$ into standard SI meters ($10^{-3}$ factor).

---

## 10. Code Quality & Pre-Commit Standards

The repository enforces formatting and linting via **Ruff**:

```bash
# Check for lint issues and automatically fix safe patterns
ruff check . --fix

# Format code to project standards
ruff format .

# Run pre-commit hooks across all files
pre-commit run --all-files
```
