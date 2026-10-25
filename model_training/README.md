# Deep Learning Segmentation, 3D Physics & Tracking (`model_training`)

This directory houses the neural network segmentation pipeline, 3D volume reconstruction, high-speed kinematic bubble tracking (2000+ FPS), single-camera Depth from Defocus (DFD), and publication-ready timeline plotting.

---

## Visual Showcase

| 3D Dual-Regime Bubble Flow | Physical Void Fraction Timeline |
| :---: | :---: |
| ![3D Flow Visualization](../0_docs/assets/3d_bubble_render.png) | ![Void Fraction Timeline](../0_docs/assets/void_fraction_timeline.png) |
| *Watertight 3D reconstruction bounded by cylindrical tube walls* | *Physical time evolution with rolling mean & confidence bands* |

| High-Speed Velocity Field Animation (2000 FPS) | U-Net Segmentation Architecture |
| :---: | :---: |
| ![High-Speed Bubble Velocity Animation](../0_docs/assets/bubble_velocity_animation.gif) | ![ResNet-34 U-Net](../0_docs/assets/resnet_unet_architecture.png) |
| *Frame-by-frame velocity vectors with particle trails* | *Pretrained ResNet-34 encoder with matched skip connections* |

---

## 1. What This Code Does (Plain English)

1. **Neural Network Bubble Segmentation**:
   - Trains a ResNet-34 U-Net network using a balanced Dice + Binary Cross-Entropy loss. This handles small bubbles occupying as little as 5% of the frame without losing detection precision.
   - Runs fast batch inference (over 46 FPS on Apple Silicon / CUDA GPUs) across thousands of raw video frames.

2. **Dual-Regime 3D Bubble Volume Reconstruction**:
   - A single 2D camera view cannot directly show depth. This pipeline uses physics-based geometric bounds:
     - **Small bubbles**: Modeled as 3D ellipsoids, with their depth strictly capped by the local physical width of the circular tube.
     - **Large vapor slugs**: Extruded across the tube cross-section with smooth distance transforms so they don't artificially breach the pipe walls.
   - Outputs total gas volume ($m^3$) and physical volumetric void fraction ($\alpha = V_{\text{gas}} / V_{\text{pipe}}$).

3. **High-Speed Velocity Tracking & Breakup Detection (2000 FPS)**:
   - At high camera framerates, bubbles move only small distances between frames. An optimal matching algorithm tracks each bubble's trajectory, computing its speed ($m/s$).
   - Detects when a parent bubble splits into two or more daughter bubbles by checking position and area conservation.

4. **Single-Camera Depth from Defocus (DFD)**:
   - Evaluates the optical blur sharpness of each bubble's outer edge.
   - Infers both the true in-focus diameter and the out-of-plane physical depth ($z$) inside the fluid column.

5. **Timeline Figures & Interactive Dashboards**:
   - Converts frame indices to physical milliseconds based on camera framerate.
   - Generates publication-ready figures (PNG, PDF) and standalone interactive HTML dashboards (with pan, zoom, and inspection tables).

---

## 2. Directory Layout

```text
model_training/
├── 1_convert_annotations.py       # Step 1: Convert Label Studio JSON into training image/mask pairs
├── 2_train_u_net.py               # Step 2: Train ResNet-34 U-Net segmentation model
├── evaluate_unet_metrics.py       # Step 3: Compute validation metrics (IoU, Dice, MCC, Accuracy)
├── 3_batch_inference.py           # Step 4: Run neural network inference across frame sequences
├── 4_void_fraction_reconstruct.py # Step 5: Reconstruct 3D bubble volumes and void fraction CSV
├── 5_track_bubbles_kinematics.py  # Step 6: 2000 FPS velocity tracking and breakup detection
├── 6_depth_from_defocus.py        # Step 7: Single-camera DFD depth and bubble sizing analysis
├── render_3d_flow.py              # Step 8: Publication 3D flow renders & WebGL HTML
├── plot_void_fraction_timeline.py # Step 9: Publication timeline plots & interactive HTML
├── visualize_flow_animation.py    # Tool: Generate slow-motion MP4/GIF animations
├── project_export.json            # Sample verified Label Studio ground-truth export
└── README.md
```

---

## 3. End-to-End Execution Recipes

### Recipe 1: Masks $\to$ 3D Void Fraction $\to$ Timeline Plot & Interactive HTML

```bash
# 1. Reconstruct 3D volumes, void fraction CSV, and render sample frame
python main.py void-fraction \
  --masks dataset/masks \
  -o results/void_fraction.csv \
  --pixel-size 6.1e-5 \
  --diameter 0.020 \
  -x 0.05 -G 400.0 \
  --render \
  --plot-timeline
```

Or run the timeline generator separately with custom multi-panel layouts:

```bash
python main.py plot-vf \
  -i results/void_fraction.csv \
  -o results/void_fraction_timeline.png \
  --html results/void_fraction_timeline.html \
  --fps 2000 \
  --multi-panel \
  -x 0.05 -G 400.0
```

---

### Recipe 2: Bubble Kinematics & Velocity Vector Field Animation (2000 FPS)

```bash
# 1. Compute velocity vectors and detect breakup events
python main.py track \
  --masks dataset/masks \
  -o results/kinematics \
  --fps 2000 \
  --pixel-size 6.1e-5 \
  --v-max 6.0

# 2. Render slow-motion velocity vector animation (MP4 and animated GIF)
python main.py animate-flow \
  --masks dataset/masks \
  -i dataset/images \
  -o results/flow_animation \
  --fps 2000 \
  --playback-fps 5.0
```

---

### Recipe 3: Multi-Method Comparison (Python Dual-Regime vs. MATLAB)

```bash
python main.py plot-vf \
  -i results/void_fraction_python.csv matlab_image_processing_final/matlab_void_fraction_masks.csv \
  --labels "Python Dual-Regime (SI)" "MATLAB Voxel Method" \
  -o results/void_fraction_comparison.png \
  --html results/void_fraction_comparison.html
```

---

## 4. Step-by-Step Script Reference

### Step 1: Convert Annotations to Training Pairs (`1_convert_annotations.py`)
Converts Label Studio JSON export files into paired PNG training images and masks:

```bash
python model_training/1_convert_annotations.py \
  --json model_training/project_export.json \
  --dataset dataset
```

---

### Step 2: Train U-Net Model (`2_train_u_net.py`)
Trains a ResNet-34 U-Net with automatic device selection (`cuda` on Nvidia GPUs, `mps` on Apple Silicon, or `cpu`):

```bash
python model_training/2_train_u_net.py \
  --dataset dataset \
  --output-model models/run_1_unet.pth \
  --epochs 40 \
  --batch-size 8 \
  --lr 1e-3
```

---

### Step 3: Evaluate Model Performance (`evaluate_unet_metrics.py`)
Evaluates segmentation accuracy on the validation split:

```bash
python model_training/evaluate_unet_metrics.py \
  --model models/run_1_unet.pth \
  --dataset dataset \
  --output-csv results/model_validation_metrics.csv
```

---

### Step 4: Batch Inference Across Image Sequences (`3_batch_inference.py`)
Runs neural network segmentation across an entire experimental image sequence:

```bash
python model_training/3_batch_inference.py \
  --input path/to/cropped_images \
  --output results/segmented_masks \
  --model models/run_1_unet.pth
```

---

### Step 5: 3D Void Fraction Reconstruction (`4_void_fraction_reconstruct.py`)
Computes volumetric gas volume ($m^3$) and void fraction ($\alpha$) frame-by-frame:

```bash
python main.py void-fraction \
  --masks results/segmented_masks \
  -o results/void_fraction.csv \
  --pixel-size 6.1e-5 \
  --diameter 0.020 \
  --render
```

| Flag | Purpose | Default |
|:---|:---|:---|
| `--masks`, `-m` | Folder with input binary masks | *Required* |
| `--output-csv`, `-o` | Path for void fraction CSV output | `results/void_fraction.csv` |
| `--pixel-size` | Camera magnification scale ($m/\text{px}$) | Auto-loaded or `6.1e-5` |
| `--diameter`, `-d` | Pipe inner diameter in meters ($0.020\text{ m} = 20\text{ mm}$) | Auto-loaded or `0.020` |
| `--vapor-quality`, `-x` | Thermodynamic vapor quality for drift-flux comparison | Optional |
| `--mass-flux`, `-G` | Mass flux ($\text{kg}/(\text{m}^2\cdot\text{s})$) for drift-flux comparison | Optional |
| `--render` | Renders a sample 3D flow visualization image | `False` |
| `--plot-timeline` | Automatically generates timeline figures upon completion | `False` |

---

### Step 6: Bubble Kinematics & Breakup Tracking (`5_track_bubbles_kinematics.py`)
Tracks bubbles frame-by-frame at 2000 FPS and identifies hydrodynamic breakup events:

```bash
python main.py track \
  --masks results/segmented_masks \
  -o results/kinematics \
  --fps 2000 \
  --pixel-size 6.1e-5 \
  --v-max 6.0
```

Outputs generated:
- `bubble_velocities_si.csv`: Bubble velocities, displacements, and coordinates.
- `bubble_breakups_si.csv`: Log of parent-to-daughter breakup events with area conservation validation.

To render the animated velocity vector visualization (MP4 and GIF):
```bash
python main.py animate-flow \
  --masks results/segmented_masks \
  -i results/cropped_frames \
  -o results/flow_animation \
  --fps 2000 \
  --playback-fps 5.0
```

---

### Step 7: Single-Camera Depth from Defocus (`6_depth_from_defocus.py`)
Measures out-of-plane physical depth ($z$) and in-focus diameter ($d_0$) using boundary edge blur:

```bash
python main.py dfd \
  --images results/cropped_frames \
  --masks results/segmented_masks \
  -o results/dfd_results.csv \
  --pixel-size 6.1e-5 \
  --beta-eff 2000.0
```

---

### Step 8: Publication 3D Flow Visualization (`render_3d_flow.py`)
Renders full 3D flow structures inside a cylindrical tube to high-resolution PNG and interactive WebGL HTML:

```bash
python main.py render \
  --mask results/segmented_masks/frame_0016.png \
  --output-png results/3d_bubble_render.png \
  --output-html results/3d_bubble_render.html \
  --pixel-size 6.1e-5 \
  --diameter 0.020
```

---

### Step 9: Void Fraction Timeline Dashboards (`plot_void_fraction_timeline.py`)
Produces publication figures (PNG, SVG, PDF) and zero-dependency interactive Chart.js HTML dashboards:

```bash
python main.py plot-vf \
  -i results/void_fraction.csv \
  -o results/void_fraction_timeline.png \
  --html results/void_fraction_timeline.html \
  --fps 2000 \
  --multi-panel
```

Key features:
- Dual axes: bottom axis in physical milliseconds, top axis in frame numbers.
- Rolling average curve with configurable smoothing window.
- Statistical confidence envelope ($\pm 1\sigma$ band).
- Multi-panel mode displaying $\alpha(t)$, gas volume ($mm^3$), and void distribution histograms.
- Comparative multi-run overlays on a common physical time axis.
