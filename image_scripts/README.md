# Image Preparation and Pre-Annotation Pipeline

This module provides the image preprocessing and semi-automated annotation pipeline for high-speed two-phase flow boiling video sequences. It processes raw camera frames into clean, background-corrected images, generates initial binary bubble masks, and formats polygon predictions for vector labeling in Label Studio.

---

## 1. Physical Intuition & Optical Methodology

In high-speed flow boiling experiments, two-phase hydrodynamic structures (nucleate bubbles, slug plugs, and vapor films) are captured via backlit shadowgraphy at high framerates (typically 2000 Hz). Several optical and experimental phenomena complicate raw image segmentation:

1. **Optical Phase Inversion & Glare**: Bubbles act as divergent spherical lenses. Backlighting refracts strongly at the gas-liquid interface, producing dark outer boundary rings with bright internal glare spots in the bubble core. Direct thresholding naively cuts holes into the bubble interiors.
2. **Conduit Boundaries & Optical Vignetting**: Rigid channel walls, external mounting flanges, and non-uniform illumination fields introduce spatial noise that must be isolated through Region of Interest (ROI) cropping.
3. **Static Background Structures**: Stationary boiling nucleation sites, wall deposits, scratched sight glasses, and persistent reflections corrupt bubble shape analysis. A temporal median filter across moving flow frames reconstructs the static illumination field without ghosting.
4. **Defocus Blur Halos**: Bubbles displaced along the optical depth axis ($z$) exhibit boundary gradient softening (defocus blur). Background subtraction must expand bubble envelopes by a safety dilation margin ($\Delta r_{\text{margin}}$) to protect blur gradients required for single-camera Depth from Defocus (DFD) analysis.

```text
Raw High-Speed Sequence (2000 FPS)
  │
  ├──> Step 0: ROI Cropping (0_roi_crop.py)
  │      └─ Isolates active fluid conduit, removing external mounting edges
  │
  ├──> Step 1: Glare- & Blur-Preserving Background Removal (1_background_subtraction.py)
  │      └─ Temporal median subtraction with hole-filling & margin dilation
  │
  ├──> Step 2: Dynamic Binary Mask Generation (2_generate_masks.py)
  │      └─ Inverted Otsu thresholding + contour closing + component area filtering
  │
  ├──> Step 3: Label Studio Pre-Annotation Export (3_create_preannotations.py)
  │      └─ Sub-pixel polygon vectorization with base64 frame embedding
  │
  └──> Tool: Single-Frame Mask Visualizer (test_single_mask.py)
         └─ Rapid parameter tuning with side-by-side diagnostic overlays
```

---

## 2. Prerequisites & Environment Setup

All scripts execute within the project virtual environment with Python 3.14+.

```bash
# Activate virtual environment from the repository root
source .venv/bin/activate

# Verify critical imaging dependencies
python -c "import cv2, numpy, scipy, matplotlib; print('OpenCV:', cv2.__version__)"
```

Expected terminal output:
```text
OpenCV: 4.11.0
```

---

## 3. Directory Layout

The preprocessing scripts operate on local, untracked image directories:

```text
images/
├── 1st_run/
│   ├── images_1/              # Raw cropped high-speed frames
│   │   ├── frame_0001.png
│   │   └── ...
│   ├── generated_masks/       # Intermediate binary masks (255 = gas, 0 = liquid)
│   │   ├── frame_0001_mask.png
│   │   └── ...
│   └── preannotations.json    # Vectorized tasks for Label Studio import
image_scripts/
├── 0_roi_crop.py               # Step 0: Interactive or headless ROI cropper
├── 1_background_subtraction.py # Step 1: Temporal median background subtraction
├── 2_generate_masks.py         # Step 2: Otsu dynamic thresholding & mask creation
├── 3_create_preannotations.py  # Step 3: Base64 JSON packaging for Label Studio
├── test_single_mask.py         # Diagnostic visualizer for single frame parameter tuning
└── README.md
```

---

## 4. Execution Workflows

### Step 0: Region of Interest (ROI) Cropping (`0_roi_crop.py`)

Crops the active fluid conduit from raw sensor frames to eliminate mounting flanges and non-uniform boundary illumination.

#### Option A: Headless / Scripted Execution (`--roi x,y,w,h`)

Use known bounding box pixel coordinates to batch-crop frames without launching a GUI:

```bash
python image_scripts/0_roi_crop.py \
  --input images/1st_run/images_1 \
  --output results/cropped_frames \
  --roi 10,10,600,200
```

Expected terminal output:
```text
Found 14 images in images/1st_run/images_1
Using ROI (x, y, w, h) = (10, 10, 600, 200)
  Processed 14/14

Done. Saved: 14, Failed: 0 -> results/cropped_frames
```

#### Option B: Interactive Graphical Selection

When processing a new experimental sequence with unknown pixel bounds, omit `--roi` to launch an interactive Matplotlib selection window:

```bash
python image_scripts/0_roi_crop.py \
  --input images/1st_run/images_1 \
  --output results/cropped_frames \
  --ref 0
```

*Selection Controls*:
- Use the Matplotlib magnifying glass toolbar icon to zoom and pan.
- **Left-Click**: Define the first corner, then left-click the opposite diagonal corner of the ROI rectangle.
- **Right-Click**: Undo the last point selection.
- **Close Window**: Automatically begins batch cropping across all sequence frames.

#### Parameter Reference

| Parameter | Type | Default | Units | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--input`, `-i` | `Path` | *images/1st_run/...* | — | Directory containing raw uncropped camera frames. |
| `--output`, `-o` | `Path` | *images/1st_run/...* | — | Output directory for cropped image files. |
| `--roi` | `str` | `None` | `px` | Bounding box coordinates formatted as `x,y,w,h`. |
| `--ref` | `str` | `None` | — | Reference image index (e.g. `0`) or filename for GUI selection. |

---

### Step 1: Glare- & Blur-Preserving Background Subtraction (`1_background_subtraction.py`)

Isolates transient vapor bubbles from stationary conduit walls while protecting both internal glare spots and outer defocus blur halos.

```bash
python image_scripts/1_background_subtraction.py \
  --input results/cropped_frames \
  --output results/bg_removed \
  --sample-size 50 \
  --threshold 15 \
  --margin 10 \
  --bg-value 255
```

Expected terminal output:
```text
No external background provided. Computing temporal median from sequence (14 frames)...
Estimating static background from 14 frames...
[SAVED] Background reference saved to: results/bg_removed/static_background_reference.png
Cleaning background across 14 frames -> 'results/bg_removed'...
  Processed [14/14]

[DONE] Background removal complete. Cleaned frames saved to: results/bg_removed
```

#### Physical Operation & Algorithmic Stages:
1. **Temporal Median Filtering**: Samples $N$ evenly spaced frames across the sequence. Because moving vapor bubbles occupy any given spatial coordinate $(x, y)$ only momentarily, the temporal median per pixel extracts the stationary conduit structure.
2. **Absolute Difference & Thresholding**: Computes $|I_{\text{raw}}(x, y) - I_{\text{bg}}(x, y)| > T_{\text{diff}}$ to detect bubble boundary edges.
3. **Internal Hole Filling**: Applies morphological closing and fills enclosed contours, ensuring that internal refraction glare spots are retained as vapor phase.
4. **Safety Margin Dilation**: Dilates the detected bubble envelope by $r_{\text{margin}}$ pixels. Pixels inside this dilated zone retain original raw pixel values, preserving out-of-plane defocus blur for DFD analysis. Pixels outside are set to uniform background intensity (`--bg-value 255`).

#### Parameter Reference

| Parameter | Type | Default | Units | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--input`, `-i` | `Path` | *Required* | — | Input directory containing cropped flow frames. |
| `--output`, `-o` | `Path` | *Required* | — | Output directory for cleaned frames. |
| `--bg-source`, `--bg` | `Path` | `None` | — | Optional path to an empty-channel static background image or folder. |
| `--sample-size` | `int` | `50` | frames | Number of frames sampled across sequence to compute median background. |
| `--threshold`, `-t` | `int` | `15` | count (0–255) | Intensity difference threshold for detecting bubble boundary interfaces. |
| `--margin`, `-m` | `int` | `10` | `px` | Radial dilation safety padding around bubbles to preserve defocus blur. |
| `--bg-value` | `int` | `255` | count (0–255) | Fill intensity for removed static background (255 = clean white, 0 = black). |

---

### Step 2: Binary Mask Generation (`2_generate_masks.py`)

Extracts binary bubble segmentation masks from background-subtracted frames using Otsu dynamic thresholding and component area filtering.

```bash
python image_scripts/2_generate_masks.py \
  --input results/bg_removed \
  --output results/generated_masks \
  --min-area 5 \
  --max-area 50000
```

Expected terminal output:
```text
Checking for images in: results/bg_removed
Found 15 image(s) to process.
[SUCCESS] Saved mask: Test_run_21_2_51200kHz_div_lens_4_2000fps_C001H001S0001001178_mask.png
[SUCCESS] Saved mask: Test_run_21_2_51200kHz_div_lens_4_2000fps_C001H001S0001001179_mask.png
...
[DONE] Successfully generated 15/15 masks in results/generated_masks
```

#### Parameter Reference

| Parameter | Type | Default | Units | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--input`, `-i` | `Path` | `image_scripts/images_1` | — | Directory containing background-cleaned images. |
| `--output`, `-o` | `Path` | `image_scripts/generated_masks` | — | Directory where generated binary PNG masks are saved. |
| `--min-area` | `int` | `1` | `px` | Minimum connected-component area to reject sensor speckle noise. |
| `--max-area` | `int` | `50000` | `px` | Maximum connected-component area to reject full-frame lighting artifacts. |

---

### Step 3: Vector Pre-Annotation Serialization (`3_create_preannotations.py`)

Converts binary raster masks into vector polygon tasks encoded in JSON format, directly embedding base64 image data for importing into Label Studio.

```bash
python image_scripts/3_create_preannotations.py \
  --images-dir results/bg_removed \
  --masks-dir results/generated_masks \
  --output-json results/preannotations.json
```

Expected terminal output:
```text
[DONE] Successfully created results/preannotations.json with 15 tasks!
```

#### Parameter Reference

| Parameter | Type | Default | Units | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--images-dir` | `Path` | `image_scripts/images_1` | — | Directory containing raw or background-cleaned frames. |
| `--masks-dir` | `Path` | `image_scripts/generated_masks` | — | Directory containing matching binary mask PNGs. |
| `--output-json` | `Path` | `image_scripts/preannotations.json` | — | Target JSON file formatted for Label Studio task import. |

---

### Single-Frame Diagnostic Visualizer (`test_single_mask.py`)

Provides visual feedback on segmentation thresholds on a single frame before running batch operations across thousands of frames.

#### Headless Mode (Automated / CI)

```bash
python image_scripts/test_single_mask.py \
  --image images/1st_run/images_1/Test_run_21_2_51200kHz_div_lens_4_2000fps_C001H001S0001001178.png \
  --output results/test_vis.png \
  --save-mask results/test_mask.png \
  --min-area 5 \
  --max-area 50000 \
  --no-show
```

Expected terminal output:
```text
Testing bubble segmentation on: images/1st_run/images_1/Test_run_21_2_51200kHz_div_lens_4_2000fps_C001H001S0001001178.png
[DONE] Detected 253 bubble component(s).
[SUCCESS] Saved binary mask to results/test_mask.png
[SUCCESS] Saved visualization to results/test_vis.png
```

#### Interactive Mode (GUI)

Omit `--no-show` to open a 3-panel Matplotlib window displaying:
1. Original Grayscale Image
2. Generated Binary Mask
3. Detected Bubble Boundary Contour Overlays

#### Parameter Reference

| Parameter | Type | Default | Units | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--image`, `-i` | `Path` | `None` | — | Path to source frame. If omitted, uses first image in `images_1/`. |
| `--min-area` | `int` | `1` | `px` | Minimum connected-component area threshold. |
| `--max-area` | `int` | `50000` | `px` | Maximum connected-component area threshold. |
| `--output`, `-o` | `Path` | `None` | — | Optional output path to save the 3-panel diagnostic figure. |
| `--save-mask` | `Path` | `None` | — | Optional path to export the standalone binary mask PNG. |
| `--no-show` | `flag` | `False` | — | Disables the interactive display window (required for headless environments). |

---

## 5. Label Studio Annotation Workflow

Import the generated JSON pre-annotations into Label Studio to rapidly establish verified ground-truth masks for neural network training:

### A. Labeling Interface Configuration

In Label Studio, navigate to **Project Settings** → **Labeling Interface** → **Code**, and paste the XML template below:

```xml
<View>
    <Image name="image" value="$image" zoom="true" zoomBy="1.5" zoomControl="true" rotateControl="true"/>
    <Header value="Two-Phase Flow Boiling Bubble Annotation Interface:"/>
    <BrushLabels name="brush_labels" toName="image">
        <Label value="Bubble" background="#0000FF"/>
    </BrushLabels>
    <PolygonLabels name="poly_labels" toName="image">
        <Label value="Bubble" background="#FF0000"/>
    </PolygonLabels>
</View>
```

### B. Verification & Export

1. Open the project dashboard and click **Import**.
2. Select and upload `results/preannotations.json`.
3. Open a task: pre-detected bubbles display as red vector polygons overlaid on the raw flow image.
4. Correct annotations:
   - Select and press `Backspace` / `Delete` to remove false positives (e.g. wall shadows).
   - Drag vertices to conform to bubble interface contours.
   - Use the Brush Tool (`1`) or Polygon Tool (`2`) to add undetected bubbles.
5. Click **Submit** to commit the ground-truth annotation.
6. Export the completed dataset as JSON (e.g., `model_training/project_export.json`) for downstream model training.
