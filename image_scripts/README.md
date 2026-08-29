# Instructions for Use

### Prerequisites

- Python 3.10+

- Required packages:

```bash
pip install opencv-python numpy pillow
```

- Label Studio running locally or in Docker.

### Project Structure

```plaintext
Images/                          # uncommitted to the repo
├── images_1/                    # Input raw high-speed flow frames
│   ├── raw_1/                   # Subfolder for uncut images
│   └── test_image.png
├── generated_masks/             # Intermediate binary PNG masks
│   └── test_image_mask.png
└── preannotations.json          # Final Label Studio import file

image_scripts/
├── roi_crop_pre_processing.py   # Optional ROI crop and pre-processing script
├── 1_generate_masks.py          # OpenCV mask creation script
└── 2_create_preannotations.py   # Polygon JSON serialization script

```

## Workflow Steps

### 0. Crop the image

🚨 **Choose the directory of photos you want to work with** in every script.

Use `roi_crop_pre_processing.py` to crop the region of interest (ROI) from your raw/uncut images.
**Note:** Raw is not the file format, but simply the original `.png` or `.jpg` images.

This removes unnecessary background and focuses on the area where bubbles are expected to appear.

### 1. Generate Binary Masks

Place your unannotated raw images inside the `image_scripts/images_1/` directory and execute:

```bash
python 1_generate_masks.py
```

_Adjust `min_area` (lower bound noise filter) and `max_area` (upper bound bulk wave cutoff) inside the script as needed
for your flow conditions._

### 2. Generate Label Studio JSON

Generate the pre-annotations task file:

```bash
python 2_create_preannotations.py
```

This produces `preannotations.json` in the `image_scripts` folder.

### 3. Configure Label Studio Interface

Inside your Label Studio project settings (**Settings** → **Labeling Interface** → **Code**), use the following XML
template:

**Important:** You should run your `Label Studio` instance inside a Docker container since it might need a specific
version of Python, and you might not be using it.

In this way `Label Studio` will run in a virtual environment, and you will not have to worry about dependencies. It will
also become project agnostic, and you can use it elsewhere if you need.

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

### 4. Import and Annotate

1. Navigate to the project dashboard and click `Import`.

2. Upload `preannotations.json`.

3. Open any task: pre-detected bubbles will appear as red vector polygons over the original raw frame.

4. Correct the pre-annotations by deleting false positives (`Backspace`), adjusting vertices, or painting fine regions
   using the brush (`1`) and polygon (`2`) tools.

5. Click `Submit` to finalize the ground truth masks.
