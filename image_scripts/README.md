# Instructions for Use

### Prerequisites

- Python 3.10+
- Required packages:

```bash
pip install opencv-python numpy pillow matplotlib
```

- Label Studio running locally or in Docker.

---

### Directory Structure

```plaintext
Images/                          # Uncommitted to the repo
├── images_1/                    # Input raw high-speed flow frames
│   ├── raw_1/                   # Subfolder for uncut images
│   └── test_image.png
├── generated_masks/             # Intermediate binary PNG masks
│   └── test_image_mask.png
└── preannotations.json          # Final Label Studio import file

image_scripts/
├── 0_roi_crop.py                # Step 0: Interactive or batch ROI cropper
├── 1_background_subtraction.py  # Step 1: Static background removal & bubble isolation
├── 2_generate_masks.py          # Step 2: Binary mask generation (Otsu + contours)
├── 3_create_preannotations.py   # Step 3: Polygon JSON serialization for Label Studio
├── test_single_mask.py          # Single-frame test & visualizer script (Matplotlib)
└── README.md
```

---

## Workflow & Script Execution Examples

### Step 0. Region of Interest (ROI) Cropping (`0_roi_crop.py`)

Crops the active flow boiling area from raw high-speed camera frames to remove boundaries, edges, or irrelevant
background before processing.

#### Option A: Interactive Selection (GUI)

Opens a Matplotlib viewer where you can zoom/pan (using the toolbar) and left-click two opposite corners of the ROI
rectangle (right-click to undo). Closing the window starts the batch crop.

```bash
python 0_roi_crop.py \
  --input "/path/to/raw_images" \
  --output "/path/to/cropped_images"
```

Specifying a specific reference image for selection:

```bash
# Using a frame index (0-based)
python 0_roi_crop.py -i ./Images/raw_frames -o ./Images/cropped --ref 0

# Using a specific filename
python 0_roi_crop.py -i ./Images/raw_frames -o ./Images/cropped --ref frame_00120.png
```

#### Option B: Non-Interactive / Headless Crop (`--roi x,y,w,h`)

Skip the GUI and directly apply known bounding box coordinates:

```bash
python 0_roi_crop.py \
  --input "./Images/raw_frames" \
  --output "./Images/cropped" \
  --roi 120,45,600,400
```

---

### Step 1. Background Subtraction (`1_background_subtraction.py`)

Removes static structures (pipe walls, static bubbles, sensor artifacts) across time while strictly isolating bubbles
and preserving defocus blur margins.

#### Example 1: Automatic Temporal Median Background (Default)

Computes the static background across the sequence (samples 200 frames across time):

```bash
python 1_background_subtraction.py \
  --input "./Images/cropped" \
  --output "./Images/bg_removed"
```

#### Example 2: Lingering Vapor Bubbles (Percentile Method)

If large vapor bubbles linger across frames, use percentile aggregation (e.g. 75th-85th percentile):

```bash
python 1_background_subtraction.py \
  --input "./Images/cropped" \
  --output "./Images/bg_removed" \
  --stat-method percentile \
  --percentile 80.0 \
  --threshold 25 \
  --min-area 30 \
  --margin 10 \
  --save-masks
```

#### Example 3: Using a Dedicated Background Reference Image or Folder

If you have a separate reference image or empty flow sequence:

```bash
python 1_background_subtraction.py \
  --input "./Images/cropped" \
  --output "./Images/bg_removed" \
  --bg-source "/path/to/static_background.png"
```

**Key Arguments:**

- `--input` / `-i` *(required)*: Input directory with frames.
- `--output` / `-o` *(required)*: Output directory for cleaned frames.
- `--bg-source` / `--bg`: Optional path to dedicated background image file or folder.
- `--stat-method`: Background estimation method (`median`, `percentile`, `max`, `min`).
- `--threshold` / `-t`: Intensity difference threshold for bubble detection (default: `20`).
- `--min-area`: Minimum pixel area to reject sensor noise specks (default: `30`).
- `--margin` / `-m`: Dilation margin in pixels around bubbles to preserve defocus blur (default: `10`).
- `--bg-value`: Background fill intensity (default: `255` = white, `0` = black).
- `--save-masks`: Flag to save envelope masks under `output_dir/bubble_envelopes/`.

---

### Step 2. Generate Binary Masks (`2_generate_masks.py`)

Generates binary segmentation masks for a batch of images using inverted Otsu dynamic thresholding, contour hole
filling, and morphological operations.

Place your input images in the folder specified in the script (`image_scripts/images_1/` by default) and run:

```bash
python 2_generate_masks.py
```

*Outputs binary masks to `image_scripts/generated_masks/` (e.g., `frame_001_mask.png`).*

> **Tip:** Pass `--min-area` and `--max-area` CLI options or tune `segment_bubbles()` if needed to filter tiny noise
> specks or large bulk waves.

---

### Step 3. Generate Label Studio JSON (`3_create_preannotations.py`)

Converts the binary masks and images into polygon vector annotations encoded as Label Studio task predictions with
embedded base64 images.

Ensure `images_1/` and `generated_masks/` are populated, then execute:

```bash
python 3_create_preannotations.py
```

*Produces `image_scripts/preannotations.json` ready for Label Studio import.*

---

### Tool: Single-Image Test & Visualizer (`test_single_mask.py`)

Use this script to visually inspect and tune segmentation parameters (`--min-area`, `--max-area`) on a single frame
before launching batch jobs:

```bash
# View interactive side-by-side plot (Original, Binary Mask, Contour Overlay)
python test_single_mask.py --image path/to/frame.png

# Save visualization figure and binary mask
python test_single_mask.py --image path/to/frame.png --output result_plot.png --save-mask mask.png
```

---

## 4. Label Studio Configuration & Workflow

### A. Interface XML Setup

Inside your Label Studio project settings (**Settings** → **Labeling Interface** → **Code**), paste the following XML
template:

```xml
<View>
    <Image name="image" value="$image" zoom="true" zoomBy="1.5" zoomControl="true" rotateControl="true"/>
    <Header value="Select a tool and annotate bubbles:"/>
    <BrushLabels name="brush_labels" toName="image">
        <Label value="Bubble" background="#0000FF"/>
    </BrushLabels>
    <PolygonLabels name="poly_labels" toName="image">
        <Label value="Bubble" background="#FF0000"/>
    </PolygonLabels>
</View>
```

> **Note:** Running Label Studio in a Docker container ensures environment independence and avoids Python dependency
> conflicts.

### B. Import and Annotation

1. Navigate to the Label Studio project dashboard and click **Import**.
2. Upload `preannotations.json`.
3. Open any task: pre-detected bubbles will appear as red vector polygons overlaid on the raw frame.
4. Correct pre-annotations:
   - Delete false positives (`Backspace` / `Delete`).
   - Adjust polygon vertices.
   - Refine tricky regions with brush (`1`) or polygon (`2`) tools.
5. Click **Submit** to finalize the ground truth annotations.
