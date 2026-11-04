# Image Preparation & Optical Preprocessing (`image_scripts`)

This folder contains the preprocessing tools for high-speed two-phase flow boiling videos. It takes raw camera frames with strong reflections, dark borders, and glare, and turns them into clean images and binary bubble masks ready for labeling or neural network inference.

---

## Visual Summary

| Raw High-Speed Camera Frame | Extracted Binary Bubble Mask |
| :---: | :---: |
| ![Raw High-Speed Frame](../0_docs/assets/sample_raw_frame.png) | ![Extracted Binary Mask](../0_docs/assets/sample_mask.png) |
| *High-speed frame with glare and boundary vignetting* | *Clean bubble masks with preserved interfaces* |

---

## 1. What These Scripts Do (Plain English)

1. **Optical Phase Inversion & Glare**: Bubbles act like divergent lenses under backlighting. They have dark outer edges with bright internal glare spots. If you apply a standard brightness threshold, it cuts holes in the bubble centers. These scripts use morphological contour closing and hole-filling so the bubble interior stays solid.
2. **Tube Conduit Isolation**: Camera sensors capture mounting flanges and dark exterior walls. Cropping isolates the active fluid conduit.
3. **Smart Background Subtraction**: Stationary tube scratches and sight glass reflections corrupt bubble analysis. A temporal median filter removes static walls while preserving bubble shapes. A safety margin dilation around bubble edges keeps the optical blur needed for Depth from Defocus (DFD).
4. **Vector Pre-Annotations**: Generates polygon tasks directly packaged with embedded base64 images so you can inspect and refine them in Label Studio with zero setup friction.

---

## 2. Directory Structure

```text
image_scripts/
├── calibrate_pixel_size.py     # Step -1: Measure optical scale (m/px) from known geometry
├── 0_roi_crop.py               # Step 0: Crop tube channel (GUI or fixed-size locking)
├── 1_background_subtraction.py # Step 1: Remove static walls while protecting bubble glare & blur
├── 2_generate_masks.py         # Step 2: Extract binary masks using dynamic Otsu thresholding
├── 3_create_preannotations.py  # Step 3: Package vector polygons into Label Studio JSON
├── test_single_mask.py         # Tool: Interactive preview on a single test frame
└── README.md
```

---

## 3. End-to-End Execution Sequence

Run through these five simple steps to turn raw camera frames into cleaned images, binary masks, and Label Studio annotation tasks:

```bash
# 1. Calibrate optical pixel size from known conduit diameter (e.g. 20 mm inner diameter)
python main.py calibrate -i path/to/raw_images --size-mm 20.0

# 2. Crop active fluid conduit (removes mounting flanges and non-uniform borders)
python main.py crop -i path/to/raw_images -o results/cropped_frames --roi 10,10,891,328

# 3. Clean stationary background (preserves glare and defocus blur)
python main.py bg-sub -i results/cropped_frames -o results/bg_removed --sample-direction "0,80" --threshold 25

# 4. Generate binary bubble masks with dynamic thresholding
python main.py gen-masks -i results/bg_removed -o results/generated_masks --min-area 5 --max-area 50000

# 5. Package masks into Label Studio pre-annotations JSON
python image_scripts/3_create_preannotations.py \
  --images-dir results/bg_removed \
  --masks-dir results/generated_masks \
  --output-json results/preannotations.json
```

---

## 4. Step-by-Step Tool Reference

### Step -1: Optical Scale Calibration (`calibrate_pixel_size.py`)
Calculates the optical scale in meters per pixel ($m/\text{px}$) by letting you click two points across a known physical dimension (e.g., tube inner diameter).

```bash
# Via Unified CLI:
python main.py calibrate -i path/to/raw_images --size-mm 20.0

# Direct script:
python image_scripts/calibrate_pixel_size.py -i path/to/raw_images -smm 20.0
```

> **Downstream Integration & Centerline Calibration**:
> Calibration produces a machine-readable report named `0_<folder>_calibration.txt`. In addition to `pixel_size_m` and `diameter_m`, it records the vertical coordinates of the tube boundaries (`point1_y`, `point2_y`) and native image dimensions. Downstream tools (`main.py void-fraction`, `main.py animate-3d`) use these points to compute the exact physical channel centerline ($y_{\text{center}} = (y_{\text{top}} + y_{\text{bot}})/2$). This ensures that even if binary masks are later padded or resized, 3D bubble geometries remain strictly centered and enclosed inside the conduit.

---

### Step 0: Conduit ROI Cropping (`0_roi_crop.py`)
Isolates the active flow section from the rest of the sensor view.

#### Headless Mode (Known Coordinates)
```bash
python main.py crop \
  -i path/to/raw_images \
  -o results/cropped_frames \
  --roi 10,10,891,328
```

#### Fixed-Size Locking (`--size`)
Locks the crop box dimensions to match a previous run or standard size, allowing you to drag the box without altering width or height:
```bash
python main.py crop \
  -i path/to/raw_images \
  -o results/cropped_frames \
  --size 891,328
```

#### Interactive GUI Selection
Omit `--roi` to launch an interactive selection window with full keyboard nudging:
- **Click & drag**: Adjust corners.
- **Arrow keys**: Nudge active point by 1 pixel.
- **Shift + Arrow keys**: Fine 1-pixel nudging.
- **Ctrl / Alt + Arrow keys**: Fast 10-pixel jumps.
- **Tab**: Cycle between selection corners.
- **Enter / Space**: Confirm selection and start batch crop.

```bash
python main.py crop -i path/to/raw_images -o results/cropped_frames --ref 0
```

---

### Step 1: Background Subtraction (`1_background_subtraction.py`)
Removes static channel walls, scratches, and persistent lighting patterns while protecting bubble boundaries and internal glare.

```bash
python main.py bg-sub \
  -i results/cropped_frames \
  -o results/bg_removed \
  --sample-direction "0,80" \
  --threshold 25 \
  --margin 10
```

| Flag | Purpose | Recommended Value |
|:---|:---|:---|
| `--sample-direction` | Specific consecutive frame range (e.g. `"0,80"`) to sample empty channel background without ghost bubbles | Window with low bubble density |
| `--threshold`, `-t` | Pixel difference threshold for detecting bubbles | `15` to `25` |
| `--margin`, `-m` | Safety padding pixels around bubbles to keep edge blur for DFD | `10` |
| `--bg-value` | Background fill color (255 = clean white, 0 = black) | `255` |

---

### Step 2: Binary Mask Extraction (`2_generate_masks.py`)
Extracts clean black-and-white bubble masks using dynamic Otsu thresholding and component area filtering.

```bash
python main.py gen-masks \
  -i results/bg_removed \
  -o results/generated_masks \
  --min-area 5 \
  --max-area 50000
```

| Flag | Purpose | Recommended Value |
|:---|:---|:---|
| `--min-area` | Discards tiny camera sensor noise speckles | `5` px |
| `--max-area` | Discards massive full-frame lighting artifacts | `50000` px |

---

### Step 3: Vector Pre-Annotations (`3_create_preannotations.py`)
Embeds masks and frames into a single base64 JSON file for fast importing into Label Studio:

```bash
python image_scripts/3_create_preannotations.py \
  --images-dir results/bg_removed \
  --masks-dir results/generated_masks \
  --output-json results/preannotations.json
```

---

### Single-Frame Diagnostic Visualizer (`test_single_mask.py`)
Test and tune segmentation thresholds on a single frame before running batch operations:

```bash
python image_scripts/test_single_mask.py \
  --image results/bg_removed/frame_0016.png \
  --output results/test_diagnostic.png \
  --min-area 5 \
  --max-area 50000 \
  --no-show
```

---

## 5. Label Studio Annotation Setup

1. In Label Studio, create a new project and go to **Settings** $\rightarrow$ **Labeling Interface** $\rightarrow$ **Code**.
2. Paste this configuration:
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
3. Click **Import** and upload `results/preannotations.json`.
4. The bubbles appear pre-outlined as red polygons. Review and adjust any boundaries, then export the verified dataset as JSON (e.g. `model_training/project_export.json`).
