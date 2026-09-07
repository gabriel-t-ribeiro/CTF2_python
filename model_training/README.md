# Deep Learning Model Training & Quantitative Flow Analysis Pipeline

This directory contains the machine learning, 3D volumetric reconstruction, kinematic tracking, and single-camera Depth from Defocus (DFD) analysis pipeline for high-speed two-phase flow boiling in conduits.

The tools convert experimental annotations into training data, train a ResNet-34 U-Net segmentation network, execute batch inference across experimental sequences, and calculate physical quantities (volumetric void fraction $\alpha$, bubble velocities $v$, hydrodynamic breakup events, and 3D spatial coordinates $(x, y, z)$) strictly in standard SI units ($\text{m}$, $\text{m}^3$, $\text{m/s}$, $\text{kg}/(\text{m}^2\cdot\text{s})$).

---

## 1. Physical & Mathematical Foundations

### 1.1 U-Net Segmentation & Class Imbalance
In bubbly and slug flow regimes, vapor bubbles often occupy less than 5–30% of total image area, causing class imbalance. The training architecture utilizes:
- **ResNet-34 Backbone**: Pretrained feature extraction transferring high-contrast boundary filters.
- **Combined Dice + BCE Loss**:
  $$\mathcal{L} = \mathcal{L}_{\text{BCE}} + \left( 1 - \frac{2 |P \cap G| + \varepsilon}{|P| + |G| + \varepsilon} \right)$$
  Where $P$ is predicted probability and $G$ is binary ground truth. Dice loss prevents gradient collapse when vapor area fraction is small.

### 1.2 Dual-Regime 3D Gas Volume Reconstruction
High-speed cameras capture 2D planar projections of 3D flow. Reconstructing 3D volumes requires physically consistent assumptions based on bubble size and conduit geometry:

```text
               Planar 2D Binary Mask
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
Small Bubbles (A < A_thresh)   Large Slugs (A >= A_thresh)
  3D Triaxial Ellipsoid           Laplacian Chord Extrusion
  a = d_major / 2                 z_chord(y) = sqrt(R^2 - delta_y^2)
  b = d_minor / 2                 D'(p) = z_chord(y) * exp(-1/(lambda*d(p)))
  c = min(b, z_chord(y))          Surface integration: 2 * sum(z * dA)
         │                             │
         └──────────────┬──────────────┘
                        ▼
           Total Gas Volume V_gas [m³]
           Conduit Volume V_pipe = pi * R² * L [m³]
           Void Fraction alpha = V_gas / V_pipe [-]
```

1. **Small Bubbles ($A < A_{\text{thresh}} = \pi R_{\text{px}}^2$)**: Modeled as triaxial ellipsoids with semi-axes $(a, b, c)$. The out-of-plane semi-axis $c$ is constrained by the local circular tube chord depth $z_{\text{chord}}(y) = \sqrt{\max(0, R^2 - \Delta y^2)}$:
   $$V_i = \frac{4}{3}\pi a b c, \quad c = \min(b, z_{\text{chord}}(y))$$
2. **Large Slugs ($A \ge A_{\text{thresh}}$)**: Elongated vapor plugs undergo wall confinement and boundary layer flattening. Volume is computed by extruding the local cylindrical chord profile modulated by Euclidean distance transform ($d(p)$) with exponential Laplacian boundary smoothing (Eduardo Reis, 2025, Eq 3.2):
   $$D'(p) = z_{\text{chord}}(y) \cdot \exp\left(-\frac{1}{\lambda \cdot d(p) + \varepsilon}\right), \quad V_i = 2 \iint D'(p) \, dA$$
3. **Volumetric Void Fraction ($\alpha$)**:
   $$\alpha = \frac{\sum_i V_i}{V_{\text{pipe}}} = \frac{\sum_i V_i}{\pi R^2 L}$$
   Where $R$ is channel inner radius (m) and $L = W_{\text{px}} \cdot s_{\text{px}}$ is the visualized channel section length (m).

### 1.3 Rouhani & Axelsson (1970) Drift-Flux Benchmark
The experimental void fraction is compared against the analytical Rouhani-Axelsson correlation for boiling water in tubes:
$$\varepsilon = \frac{\frac{x}{\rho_v}}{C_0 \left( \frac{x}{\rho_v} + \frac{1-x}{\rho_l} \right) + \frac{1.18(1-x)[g \sigma (\rho_l - \rho_v)]^{0.25}}{G \sqrt{\rho_l}}}$$
Where:
- $C_0 = 1.0 + 0.12(1 - x)$ is the radial distribution parameter
- $x$ is vapor quality (dimensionless, $0.0 \le x \le 1.0$)
- $G$ is mass flux ($\text{kg}/(\text{m}^2\cdot\text{s})$)
- $\rho_l = 958.4\text{ kg/m}^3$, $\rho_v = 0.597\text{ kg/m}^3$ (water/steam at 100°C)
- $\sigma = 0.0589\text{ N/m}$, $g = 9.81\text{ m/s}^2$

### 1.4 Bubble Kinematics & Hydrodynamic Breakup (2000 FPS)
With camera temporal resolution $\Delta t = 1/\text{FPS} = 0.500\text{ ms}$, bubble centroids are tracked across consecutive frames:
- **Optimal Bipartite Matching**: Solves linear sum assignment on the inter-frame distance cost matrix, preventing multi-bubble collision artifacts.
- **Hydrodynamic Breakup Detection**: When a parent bubble at frame $t$ splits into $\ge 2$ children at frame $t+1$, candidate children are matched within a displacement window. The split is verified via projected area conservation:
  $$\frac{|\sum A_{\text{children}} - A_{\text{parent}}|}{A_{\text{parent}}} < 0.35$$
- **Velocity Calculation**:
  $$\vec{v} = \frac{\Delta \vec{r}_{\text{px}} \cdot s_{\text{px}}}{\Delta t} \quad [\text{m/s}]$$

### 1.5 Single-Camera Depth from Defocus (DFD)
Using the calibration-free optical formulation of Rao et al. (2024), out-of-plane position $|z|$ and in-focus diameter $d_0$ are obtained from boundary gradient blur:
1. Spatial image gradients $\nabla I$ are evaluated along the bubble interface contour.
2. Dimensionless boundary gradient $\tilde{G} = r_t |\frac{\partial g_t}{\partial r_t}|$ is matched to a precomputed Gaussian convolution lookup table.
3. Inversion yields dimensionless blur $\tilde{\sigma}$ and dimensionless radius $\tilde{\rho}$, yielding in-focus diameter $d_0 = (2 r_t) / (2 \tilde{\rho})$ and optical blur $\sigma_{\text{px}} = \tilde{\sigma} d_{0,\text{px}}$.
4. Out-of-plane physical depth $|z| = \sigma_{\text{px}} / \beta_{\text{eff}}$, where $\beta_{\text{eff}}$ ($\text{px/m}$) is the optical depth calibration constant.

---

## 2. Directory Layout & Artifacts

```text
model_training/
├── 1_convert_annotations.py       # Converts Label Studio JSON to paired images/masks
├── 2_train_u_net.py               # Trains ResNet-34 U-Net segmentation network
├── 3_batch_inference.py           # Runs neural inference on raw sequence folders
├── 4_void_fraction_reconstruct.py # 3D volume reconstruction & Rouhani-Axelsson benchmark
├── 5_track_bubbles_kinematics.py  # 2000 FPS velocity tracking & breakup detection
├── 6_depth_from_defocus.py        # Single-camera DFD depth & sizing analysis
├── render_3d_flow.py              # 3D Matplotlib visualizer & Three.js WebGL generator
├── evaluate_unet_metrics.py       # Validation metrics (IoU, Dice, MCC, Accuracy)
├── project_export.json            # Sample verified Label Studio ground-truth export
└── README.md
```

---

## 3. Step-by-Step Execution Workflows

All commands are executed from the repository root with the project virtual environment active:

```bash
source .venv/bin/activate
```

---

### Step 1: Convert Annotations to Training Pairs (`1_convert_annotations.py`)

Converts exported Label Studio JSON vector polygons into rasterized binary masks (PNG) and extracts base64 raw frames:

```bash
python model_training/1_convert_annotations.py \
  --json model_training/project_export.json \
  --dataset dataset
```

Expected terminal output:
```text
[DONE] Extracted 11 images to: dataset/images
[DONE] Extracted 11 masks to: dataset/masks
```

#### Parameter Reference

| Parameter | Type | Default | Units | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--json`, `-j` | `Path` | *Required* | — | Path to Label Studio exported JSON file. |
| `--dataset`, `-d` | `Path` | `dataset` | — | Target dataset root directory (creates `images/` and `masks/`). |

---

### Step 2: Train the U-Net Segmentation Model (`2_train_u_net.py`)

Trains a ResNet-34 U-Net with automatic device selection (`cuda` on Linux GPUs, `mps` on Apple Silicon, or `cpu`):

```bash
python model_training/2_train_u_net.py \
  --dataset dataset \
  --output-model models/run_1_unet.pth \
  --epochs 20 \
  --batch-size 4 \
  --lr 1e-3
```

Expected terminal output:
```text
Training on device: mps
Epoch [01/20] - Train Loss: 0.8238 | Val Loss: 0.7911
  --> Saved best model checkpoint to: models/run_1_unet.pth
...
Epoch [20/20] - Train Loss: 0.2315 | Val Loss: 0.2840
  --> Saved best model checkpoint to: models/run_1_unet.pth
```

#### Parameter Reference

| Parameter | Type | Default | Units | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--dataset`, `-d` | `Path` | `dataset` | — | Root directory containing `images/` and `masks/`. |
| `--output-model`, `-o` | `Path` | `best_bubble_unet.pth` | — | Path to save the best model weights checkpoint (`.pth`). |
| `--epochs`, `-e` | `int` | `40` | cycles | Total training epochs across the dataset. |
| `--batch-size`, `-b` | `int` | `8` | frames | Mini-batch size for DataLoader. |
| `--lr` | `float` | `0.001` | — | Initial learning rate for AdamW optimizer. |
| `--num-workers` | `int` | `0` | threads | DataLoader subprocess workers (0 recommended on macOS). |

---

### Step 3: Evaluate Model Performance (`evaluate_unet_metrics.py`)

Computes pixel-level classification metrics on the validation split:

```bash
python model_training/evaluate_unet_metrics.py \
  --model models/run_1_unet.pth \
  --dataset dataset \
  --output-csv results/model_validation_metrics.csv
```

Expected terminal output:
```text
Running evaluation on device: mps
Evaluating on 3 validation frame(s)...

=============================================
       U-NET SEGMENTATION PERFORMANCE
=============================================
Total Pixels Analyzed : 876,744
True Positives  (TP)  : 307,880
True Negatives  (TN)  : 431,610
False Positives (FP)  : 67,412
False Negatives (FN)  : 69,842
---------------------------------------------
Accuracy              : 0.8435
Precision             : 0.8204
Recall (Sensitivity)  : 0.8151
Specificity           : 0.8649
F1-Score (Dice)       : 0.8177
Jaccard Index (IoU)   : 0.6917
MCC                   : 0.6806
=============================================
[SAVED] Results exported to: results/model_validation_metrics.csv
```

#### Parameter Reference

| Parameter | Type | Default | Units | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--model`, `-m` | `Path` | `models/run_1_unet.pth` | — | Path to trained PyTorch weights file. |
| `--dataset`, `-d` | `Path` | `dataset` | — | Dataset root folder with `images/` and `masks/`. |
| `--output-csv`, `-o` | `Path` | `model_validation_metrics.csv` | — | Target path for validation metric summary table. |

---

### Step 4: Batch Inference Across Sequences (`3_batch_inference.py`)

Processes a full folder of unannotated flow boiling frames to produce high-resolution binary masks:

```bash
python model_training/3_batch_inference.py \
  --input dataset/images \
  --output results/segmented_masks \
  --model models/run_1_unet.pth
```

Expected terminal output:
```text
Segmenting 11 frames from 'dataset/images' -> 'results/segmented_masks'...
  Processed [11/11]
```

#### Parameter Reference

| Parameter | Type | Default | Units | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--input`, `-i` | `Path` | *Required* | — | Directory containing raw cropped flow frames. |
| `--output`, `-o` | `Path` | *Required* | — | Directory to write predicted binary mask PNGs. |
| `--model`, `-m` | `Path` | `best_bubble_unet.pth` | — | Trained model checkpoint weights. |

---

### Step 5: 3D Void Fraction Reconstruction (`4_void_fraction_reconstruct.py`)

Reconstructs frame-by-frame 3D gas volumes and void fraction $\alpha$, with optional Rouhani-Axelsson benchmark comparison and 3D visual rendering:

```bash
python model_training/4_void_fraction_reconstruct.py \
  --masks results/segmented_masks \
  --output-csv results/void_fraction_results.csv \
  --pixel-size 8.6e-5 \
  --diameter 0.020 \
  --vapor-quality 0.05 \
  --mass-flux 500.0 \
  --render \
  --render-output results/3d_flow_sample.png
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
Mean Gas Volume     : 7.285778e-06 m³
Mean Pipe Volume    : 2.407277e-05 m³
----------------------------------------------------
     ROUHANI-AXELSSON THEORETICAL BENCHMARK (SI)
----------------------------------------------------
Vapor Quality (x)   : 0.0500 [-]
Mass Flux (G)       : 500.0 kg/(m² s)
Theoretical Void (ε): 0.8839 (88.39%)
Relative Difference : 65.76 %
====================================================
Results saved to    : results/void_fraction_results.csv
Rendered 3D artifact: results/3d_flow_sample.png
```

#### Parameter Reference

| Parameter | Type | Default | Units | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--masks`, `-m` | `Path` | *Required* | — | Directory containing binary mask PNGs. |
| `--output-csv`, `-o` | `Path` | `results/void_fraction_results.csv` | — | Output CSV path for frame-by-frame void fraction logs. |
| `--pixel-size` | `float` | `8.6e-5` | $\text{m/px}$ | Optical scale in meters per pixel ($8.6\times 10^{-5}\text{ m/px} = 0.086\text{ mm/px}$). |
| `--diameter`, `-d` | `float` | `0.020` | $\text{m}$ | Channel inner diameter in meters ($0.020\text{ m} = 20\text{ mm}$). |
| `--vapor-quality`, `-x` | `float` | `None` | dimensionless | Thermodynamic vapor quality $x$ ($0.0 \le x \le 1.0$). |
| `--mass-flux`, `-G` | `float` | `None` | $\text{kg}/(\text{m}^2\cdot\text{s})$ | Total flow mass flux $G$. |
| `--render` | `flag` | `False` | — | Renders a sample 3D flow visualization artifact. |
| `--render-output` | `Path` | `None` | — | Path for rendered 3D image artifact. |
| `--pipe-center-y` | `float` | `None` | `px` | Vertical centerline coordinate (defaults to $H/2$). |
| `--regime-thresh-px`| `float` | `None` | `px` | Area boundary between ellipsoid and slug regimes (defaults to $\pi R_{\text{px}}^2$). |

---

### Step 6: Bubble Kinematics & Breakup Tracking (`5_track_bubbles_kinematics.py`)

Tracks bubble centroids at 2000 FPS and identifies hydrodynamic breakup events:

```bash
python model_training/5_track_bubbles_kinematics.py \
  --masks results/segmented_masks \
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

#### Generated Outputs:
1. `bubble_velocities_si.csv`: Columns `[frame_t, frame_t1, bubble_id, velocity_m_s, displacement_m, dx_m, dy_m, area_m2, is_stable_count_pair]`
2. `bubble_breakups_si.csv`: Columns `[frame_t, frame_t1, parent_id, parent_area_m2, total_child_area_m2, area_error_rel, num_children, parent_dx_m, parent_dy_m]`

#### Parameter Reference

| Parameter | Type | Default | Units | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--masks`, `-m` | `Path` | *Required* | — | Directory containing binary mask PNGs. |
| `--output-dir`, `-o` | `Path` | `results/kinematics` | — | Destination directory for kinematic CSV logs. |
| `--fps` | `float` | `2000.0` | $\text{Hz}$ | High-speed camera acquisition framerate ($1/\text{s}$). |
| `--pixel-size` | `float` | `8.6e-5` | $\text{m/px}$ | Optical calibration factor ($8.6\times 10^{-5}\text{ m/px}$). |
| `--max-disp` | `float` | `50.0` | `px` | Maximum inter-frame search displacement radius. |

---

### Step 7: Single-Camera Depth from Defocus (`6_depth_from_defocus.py`)

Extracts 3D spatial position $(x, y, z)$ and in-focus diameter $d_0$ using boundary gradient blur analysis:

```bash
python model_training/6_depth_from_defocus.py \
  --images dataset/images \
  --masks results/segmented_masks \
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

#### Generated Outputs:
`dfd_results.csv`: Contains `[bubble_id, centroid_x_m, centroid_y_m, depth_z_m, diameter_d0_m, blur_sigma_px, sigma_tilde, g_tilde, area_px, frame]`

#### Parameter Reference

| Parameter | Type | Default | Units | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--images`, `-i` | `Path` | *Required* | — | Directory containing raw flow frames. |
| `--masks`, `-m` | `Path` | *Required* | — | Matching binary mask directory. |
| `--output-csv`, `-o`| `Path` | `results/dfd_results.csv` | — | Output CSV path for 3D coordinates and diameters. |
| `--pixel-size` | `float` | `8.6e-5` | $\text{m/px}$ | Optical calibration factor ($8.6\times 10^{-5}\text{ m/px}$). |
| `--beta-eff` | `float` | `2000.0` | $\text{px/m}$ | Effective optical defocus calibration slope ($\sigma / |\Delta z|$). |
| `--background`, `-bg`| `Path`| `None` | — | Optional static background image for intensity normalization. |

---

### Step 8: Publication-Grade 3D Flow Rendering (`render_3d_flow.py`)

Renders 3D flow structure in a cylindrical conduit to high-resolution PNG and interactive WebGL HTML:

```bash
python model_training/render_3d_flow.py \
  --mask results/segmented_masks/frame_0016_mask.png \
  --output results/3d_bubble_render.png \
  --interactive-html results/3d_bubble_render.html \
  --pixel-size 8.6e-5 \
  --diameter 0.020
```

Expected terminal output:
```text
Reconstructing 3D multi-phase flow geometry in SI units...
  Pixel size (SI) : 8.600000e-05 m/px (0.0860 mm/px)
  Diameter (SI)   : 2.000000e-02 m (20.00 mm)
  Reconstructed Bubbles: 88
  Gas Volume (SI)      : 8.036177e-06 m³
  Pipe Volume (SI)     : 2.407277e-05 m³
  Void Fraction (alpha): 0.3338 (33.383%)
[DONE] Rendered 3D visualization saved to: results/3d_bubble_render.png
[DONE] Interactive 3D HTML saved to: results/3d_bubble_render.html
```

#### Parameter Reference

| Parameter | Type | Default | Units | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--mask`, `-m` | `Path` | *Required* | — | Path to 2D binary segmentation mask. |
| `--output`, `-o` | `Path` | `results/3d_bubble_render.png` | — | Destination PNG path for 3D rendered figure. |
| `--interactive-html`| `Path` | `None` | — | Destination path for standalone Three.js WebGL HTML. |
| `--pixel-size` | `float` | `8.6e-5` | $\text{m/px}$ | Optical scale in meters per pixel. |
| `--diameter` | `float` | `0.020` | $\text{m}$ | Conduit inner diameter in meters. |
| `--pixel-size-mm` | `float` | `0.086` | $\text{mm/px}$ | Optical scale convenience flag in millimeters per pixel. |
| `--diameter-mm` | `float` | `20.0` | $\text{mm}$ | Conduit diameter convenience flag in millimeters. |
