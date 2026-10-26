# CTF2 Python: High-Speed Flow Boiling Vision & 3D Reconstruction

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Units: Standard SI](https://img.shields.io/badge/units-standard%20SI-success.svg)](#6-engineering-standards--si-units)
[![Tests: 40 Passed](https://img.shields.io/badge/tests-40%20passed-brightgreen.svg)](#3-quickstart--immediate-testing)

An end-to-end computer vision and quantitative physical analysis suite for high-speed two-phase flow boiling videos (typically acquired at 2000–3200 FPS).

---

## ⚡ Why This Pipeline? Key Advances Over Last Year's Code

Compared to the previous year's MATLAB-based framework (CTF 1), this Python/PyTorch pipeline delivers major breakthroughs in **speed**, **segmentation fidelity**, and **physical accuracy**:

| Benchmark / Capability | Previous Year (MATLAB CTF 1) | Current Pipeline (Python / PyTorch CTF 2) | Quantified Advantage |
| :--- | :--- | :--- | :--- |
| **Model Training Speed** | 45.4 – 49.6 minutes (CPU) | **1.17 minutes (70.3 s on Apple Silicon GPU/MPS)** | **~40× faster model convergence** |
| **Video Inference Rate** | ~20 – 40 FPS (single-frame loops) | **43.2 – 46.4 FPS (batched GPU inference)** | **Processes 2,000+ frames in <50 s** |
| **Segmentation Precision** | 97.15% Precision (3,373 px noise) | **97.87% Precision (2,031 px noise)** | **40% reduction in false-positive noise** |
| **Bubble Boundary Detection** | 95.02% Recall (6,025 px missed) | **97.98% Recall (1,923 px missed)** | **3.1× fewer missed bubble interfaces** |
| **3D Reconstruction Speed** | 57.2 ms/frame (17.5 FPS) | **13.0 ms/frame (77.1 FPS)** | **4.4× faster volumetric reconstruction** |
| **Large Slug Extrusion** | 453.2 ms per slug frame | **17.5 ms per slug frame** | **26× speedup on complex vapor slugs** |
| **Kinematic Tracking Reliability** | Spurious speeds up to 24.5 m/s (phantom jumps) | **Physically gated ($\le 6.0\text{ m/s}$ SI)** | **Zero teleportation artifacts at 2000 FPS** |
| **3D Out-of-Plane Localization** | Centerline planar assumption ($z = 0$) | **Single-camera Depth from Defocus (DFD)** | **Full $(x, y, z)$ spatial bubble mapping** |
| **Workflow Portability** | Proprietary MATLAB license required | **100% Open-Source Python (PyTorch + OpenCV)** | **Immediately runnable on macOS / Linux / Windows** |

---

## Visual Showcase

| 3D Flow Reconstruction | Temporal Void Fraction Timeline |
| :---: | :---: |
| ![3D Flow Visualization](0_docs/assets/3d_bubble_render.png) | ![Void Fraction Timeline](0_docs/assets/void_fraction_timeline.png) |
| *Watertight 3D bubble volumes inside circular tube* | *Physical time evolution with rolling mean & confidence bounds* |

| High-Speed Bubble Velocity Tracking (2000 FPS) | Multi-Method Comparison (Python vs. MATLAB) |
| :---: | :---: |
| ![High-Speed Bubble Velocity Animation](0_docs/assets/bubble_velocity_animation.gif) | ![Void Fraction Comparative Overlay](0_docs/assets/void_fraction_comparison.png) |
| *Frame-by-frame velocity vectors with particle trails* | *Direct overlay benchmarking against prior baseline methods* |

---

## 1. What This Code Does (Plain English)

1. **Optical Preprocessing (`image_scripts/`)**:
   - Calibrates camera scale ($m/\text{px}$) from known tube diameter.
   - Crops the active tube flow channel with interactive GUI or fixed-size locking.
   - Cleans stationary backgrounds using smart median filtering while preserving bubble internal glare cores and defocus blur boundaries.
   - Extracts binary masks using dynamic thresholding and exports vector pre-annotations to Label Studio.

2. **Deep Learning Segmentation (`model_training/`)**:
   - Converts Label Studio annotations into training dataset pairs.
   - Trains a ResNet-34 U-Net segmentation model with mixed Dice + Binary Cross-Entropy loss.
   - Runs fast batch inference over thousands of unlabelled camera frames.

3. **Physics, 3D Reconstruction & Tracking (`model_training/`)**:
   - **Dual-Regime 3D Reconstruction**: Small dispersed bubbles are reconstructed as triaxial ellipsoids bounded by the circular tube chord; large vapor slugs are extruded with smooth Laplacian distance transforms.
   - **Kinematics & Breakup Tracking**: Matches bubble centroids across consecutive frames at 2000 FPS to calculate real physical velocity vectors ($m/s$) and detect bubble breakup events.
   - **Single-Camera Depth from Defocus (DFD)**: Determines out-of-plane depth ($z$) and in-focus diameter from optical edge blur gradients.
   - **Timeline Dashboards**: Generates publication-ready static figures and interactive HTML charts (with pan/zoom and tooltips) showing void fraction evolution over time.

---

## 2. Pipeline Overview

```text
Raw High-Speed Footage (2000 FPS)
  │
  ├──> Stage 0: Optical Preprocessing (image_scripts/)
  │      ├─ calibrate_pixel_size.py     --> Measure optical scale (m/px)
  │      ├─ 0_roi_crop.py               --> Crop tube conduit (GUI or fixed --size)
  │      ├─ 1_background_subtraction.py  --> Remove background while protecting glare & blur
  │      ├─ 2_generate_masks.py          --> Dynamic Otsu thresholding & filtering
  │      └─ 3_create_preannotations.py   --> Export vector polygons to Label Studio
  │
  ├──> Stage 1: Machine Learning Segmentation (model_training/)
  │      ├─ 1_convert_annotations.py     --> Label Studio export to training pairs
  │      ├─ 2_train_u_net.py             --> Train ResNet-34 U-Net (CUDA / Apple MPS / CPU)
  │      ├─ evaluate_unet_metrics.py     --> Compute validation metrics (IoU, Dice, MCC)
  │      └─ 3_batch_inference.py         --> Sequence batch inference producing binary masks
  │
  ├──> Stage 2: Quantitative Physics & 3D Reconstruction (model_training/)
  │      ├─ 4_void_fraction_reconstruct.py --> 3D dual-regime volume & void fraction CSV
  │      ├─ plot_void_fraction_timeline.py --> Timeline figures (PNG/PDF) & interactive HTML
  │      ├─ 5_track_bubbles_kinematics.py  --> 2000 FPS velocity & breakup tracking
  │      ├─ visualize_flow_animation.py    --> Slow-motion velocity vector animations (MP4/GIF)
  │      ├─ 6_depth_from_defocus.py        --> Optical depth & in-focus sizing (DFD)
  │      └─ render_3d_flow.py              --> Publication 3D flow renders & WebGL HTML
  │
  └──> Unified CLI Orchestrator: main.py
```

---

## 3. Quickstart & Immediate Testing

The repository includes curated ground-truth test datasets (`dataset/`, `dataset_2/`, `dataset_2_no_background/`, and `test_dataset/`) so that you can clone and test any stage immediately without downloading large external video archives.

### 3.1 Setup Environment
```bash
# Clone and enter the repository
cd CTF2_python

# Activate local virtual environment
source .venv/bin/activate

# Verify Python version
python --version
```

### 3.2 Automated Unit Test Suite (40 Tests)
Run the full automated test suite covering calibration, ROI cropping, 3D void fraction math, kinematics, and flow animation:
```bash
export MPLCONFIGDIR=/tmp/mpl
python -m unittest discover tests
```

### 3.3 Instant Smoke Test on Bundled Dataset
Test 3D void fraction reconstruction and automatic timeline generation on the bundled ground truth frames in seconds:
```bash
# Computes 3D volumes across the included dataset and auto-generates a timeline plot
python main.py void-fraction \
  --masks dataset/masks \
  -o results/test_void_fraction.csv \
  --pixel-size 6.1e-5 \
  --diameter 0.020 \
  --plot-timeline
```

### 3.4 Code Quality & Linter Checks
```bash
ruff check .
```

---

## 4. Unified Command Line Interface (`main.py`)

The root `main.py` provides a single entrypoint for every stage of the pipeline:

| Command | What It Does | Common Flags |
|:---|:---|:---|
| `main.py calibrate` | Interactive optical scale calibration ($m/\text{px}$) | `-i`, `--size-mm 20.0` |
| `main.py crop` | Crop active conduit region (GUI or headless) | `-i`, `-o`, `--roi 10,10,891,328`, `--size 891,328` |
| `main.py bg-sub` | Glare- and blur-preserving background subtraction | `-i`, `-o`, `--sample-direction "0,80"`, `--threshold 25` |
| `main.py gen-masks` | Fast binary mask extraction from cleaned frames | `-i`, `-o`, `--min-area 5`, `--max-area 50000` |
| `main.py void-fraction` | 3D volume reconstruction & void fraction CSV | `--masks`, `-o`, `--pixel-size 6.1e-5`, `--diameter 0.020`, `--plot-timeline` |
| `main.py plot-vf` | Publication timeline figures and interactive HTML | `-i`, `-o`, `--html`, `--fps 2000`, `--multi-panel` |
| `main.py track` | Bubble velocity vectors and breakup events | `--masks`, `-o`, `--fps 2000`, `--pixel-size 6.1e-5`, `--v-max 6.0` |
| `main.py animate-flow` | Slow-motion velocity vector animation (MP4 / GIF) | `--masks`, `-i`, `-o`, `--fps 2000`, `--playback-fps 5.0` |
| `main.py dfd` | Single-camera Depth from Defocus ($z$-position & sizing) | `--images`, `--masks`, `-o`, `--beta-eff 2000.0` |
| `main.py render` | 3D flow field visualization & Three.js WebGL HTML | `--mask`, `--output-png`, `--output-html` |

---

## 5. Ready-to-Run Recipes

### Recipe 1: Raw Frames $\to$ 3D Void Fraction $\to$ Timeline & HTML Dashboard

```bash
# 1. Crop tube flow channel
python main.py crop -i path/to/raw_images -o results/cropped_frames --roi 10,10,891,328

# 2. Subtract background using consecutive frame range (avoids ghost bubbles)
python main.py bg-sub -i results/cropped_frames -o results/bg_removed --sample-direction "0,80" --threshold 25

# 3. Generate binary masks
python main.py gen-masks -i results/bg_removed -o results/segmented_masks --min-area 5

# 4. Reconstruct 3D volumes and generate timeline plot automatically
python main.py void-fraction \
  --masks results/segmented_masks \
  -o results/void_fraction.csv \
  --pixel-size 6.1e-5 \
  --diameter 0.020 \
  -x 0.05 -G 400.0 \
  --plot-timeline
```

### Recipe 2: Bubble Kinematics & Velocity Animation (2000 FPS)

```bash
# 1. Track kinematics and breakup events
python main.py track \
  --masks results/segmented_masks \
  -o results/kinematics \
  --fps 2000 \
  --pixel-size 6.1e-5 \
  --v-max 6.0

# 2. Render slow-motion velocity vector animation (MP4 and animated GIF)
python main.py animate-flow \
  --masks results/segmented_masks \
  -i results/cropped_frames \
  -o results/flow_animation \
  --fps 2000 \
  --playback-fps 5.0
```

### Recipe 3: Multi-Run / Multi-Method Benchmark Comparison

```bash
python main.py plot-vf \
  -i results/void_fraction_python.csv matlab_image_processing_final/matlab_void_fraction_masks.csv \
  --labels "Python Dual-Regime (SI)" "MATLAB Voxel Method" \
  -o results/void_fraction_comparison.png \
  --html results/void_fraction_comparison.html
```

---

## 6. Engineering Standards & SI Units

All calculations and pipeline outputs strictly adhere to standard SI units:

| Physical Quantity | SI Unit | Code Argument / Variable | Typical Experimental Value |
|:---|:---|:---|:---|
| Optical Scale | $\text{m/px}$ | `--pixel-size`, `pixel_size_m` | $6.1\times 10^{-5}\text{ m/px}$ ($0.061\text{ mm/px}$) |
| Conduit Diameter | $\text{m}$ | `--diameter`, `diameter_m` | $0.020\text{ m}$ ($20\text{ mm}$) |
| Gas Volume | $\text{m}^3$ | `gas_volume_m3` | $10^{-7} - 10^{-4}\text{ m}^3$ |
| Void Fraction ($\alpha$) | dimensionless | `void_fraction_alpha` | $0.0 - 1.0$ |
| Bubble Velocity | $\text{m/s}$ | `velocity_m_s` | $0.5 - 6.0\text{ m/s}$ |
| Acquisition Rate | $\text{Hz}$ ($1/\text{s}$) | `--fps`, `fps` | $2000 - 3200\text{ FPS}$ |
| Optical Defocus Constant | $\text{px/m}$ | `--beta-eff`, `beta_eff` | $1000 - 5000\text{ px/m}$ |

> **Unit Safety Guardrail**: Inputs mistakenly passed in millimeters (such as `--diameter 20.0`) are automatically detected and converted to meters ($0.020\text{ m}$).

---

## 7. Deep Learning Architecture

The segmentation model utilizes a U-Net with a pretrained ResNet-34 encoder and feature skip connections:

![ResNet-34 U-Net Architecture](0_docs/assets/resnet_unet_architecture.png)

*Detailed benchmark reports comparing performance on background-removed vs raw images are available in [`models/model_comparison_report.md`](models/model_comparison_report.md).*

---

## 8. Detailed Sub-Module Guides

For complete flag references, GUI keybindings, and deeper workflow examples:
- [Optical Preprocessing Guide](image_scripts/README.md) (`image_scripts/`)
- [Model Training, 3D Physics & Tracking Guide](model_training/README.md) (`model_training/`)
