# CTF2 Python

This repository implements a computer-vision and physics-analysis pipeline for high-speed two-phase flow imaging. The project combines image preprocessing, semi-automatic bubble annotation, U-Net segmentation, and post-processing to estimate void fraction and bubble kinematics from flow-boiling videos.

The workflow is designed around training a deep learning model on labeled bubble masks, running inference on full image sequences, then reconstructing volumetric gas fraction and tracking bubble motion over time.

## Project overview

- Segment bubbles from grayscale flow images using a U-Net model
- Generate rough segmentation masks and pre-annotations for Label Studio
- Manually refine a small subset of annotations to build a training dataset
- Train and validate a segmentation model with binary mask labels
- Run batch inference across the full frame set
- Reconstruct volumetric void fraction from binary masks
- Track bubble displacement and detect breakup events from consecutive frames

## High-level pipeline

```text
Raw high-speed frames
  -> ROI cropping / preprocessing
  -> OpenCV mask seed + Label Studio annotations
  -> Train U-Net on binary segmentation masks
  -> Batch inference to generate full-mask outputs
  -> Void fraction reconstruction
  -> Bubble tracking and breakup analysis
```

## Repository structure

```text
.
├── image_scripts/                     # Image preparation and annotation support
│   ├── 1_generate_masks.py            # Seed binary masks from frame images
│   ├── 2_create_preannotations.py    # Convert masks into Label Studio import JSON
│   ├── pre_annotations_for_label_studio.py
│   ├── roi_crop_pre_processing.py    # Optional ROI crop helper
│   └── README.md                     # Data-prep workflow instructions
│
├── model_training/                    # Deep learning training and analysis scripts
│   ├── 1_convert_annotations.py      # Convert annotation outputs to training masks
│   ├── 2_train_u_net.py              # Train the U-Net model
│   ├── 3_batch_inference.py          # Run segmentation across image folders
│   ├── 4_reconstruct_void_fraction.py  # Calculate void fraction from masks
│   ├── 5_track_bubbles_kinematics.py   # Bubble movement and breakup tracking
│   ├── evaluate_unet_metrics.py      # Validation and metric reporting
│   └── README.md                     # Model training documentation
│
├── dataset/                           # Local training dataset (not tracked)
│   ├── images/
│   └── masks/
│
├── images/                            # Local raw / processed image data (not tracked)
├── models/                            # Saved model checkpoints (not tracked)
├── results/                           # CSV outputs from analysis (not tracked)
├── main.py                            # Placeholder/default script
├── .gitignore                         # Ignores data, weights, generated outputs
├── .pre-commit-config.yaml           # Formatting / quality hooks
├── README.md                          # Project overview
└── .venv/                             # Local virtual environment (not tracked)
```

## Requirements

This project is Python-based and expects a local virtual environment.

Required runtime and libraries:

- Python 3.10+
- OpenCV
- NumPy
- Pandas
- SciPy
- PyTorch
- Albumentations
- segmentation-models-pytorch
- Label Studio (for manual annotation refinement)

Example installation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install opencv-python numpy pandas scipy torch albumentations segmentation-models-pytorch
```

If you plan to use Label Studio for annotation review, follow the official Label Studio setup for your environment and run it locally or in Docker.

## Typical workflow

### 1. Prepare the image data

Use the scripts in `image_scripts/` to crop, clean, and generate initial segmentation masks before annotation.

```bash
cd image_scripts
python roi_crop_pre_processing.py
python 1_generate_masks.py
python 2_create_preannotations.py
```

This produces initial bubble pre-annotations that can be imported into Label Studio.

### 2. Refine labels in Label Studio

Import the generated JSON into Label Studio and correct the predicted masks for a small set of representative frames.

Use the exported or corrected annotations as training ground truth for the model.

### 3. Convert annotations into training datasets

```bash
cd model_training
python 1_convert_annotations.py --json ../image_scripts/preannotations.json --dataset ../dataset
```

This step prepares the binary image/mask pairs expected by the training pipeline.

### 4. Train the segmentation model

```bash
python 2_train_u_net.py \
  --dataset ../dataset \
  --output-model ../models/best_bubble_unet.pth \
  --epochs 40 \
  --batch-size 8 \
  --lr 1e-3
```

The training script uses a U-Net with a ResNet-34 encoder and saves the best checkpoint based on validation loss.

### 5. Run inference on full sequences

```bash
python 3_batch_inference.py \
  --input ../images/raw_frames \
  --output ../images/segmented_masks \
  --model ../models/best_bubble_unet.pth
```

This creates binary segmentation masks for each input frame.

### 6. Reconstruct void fraction

```bash
python 4_reconstruct_void_fraction.py \
  --masks ../images/segmented_masks \
  --output-csv ../results/void_fraction_results.csv \
  --pixel-size 0.067 \
  --diameter 12.0
```

The script estimates the volumetric gas fraction from connected components and converts pixel geometry into physical values using the supplied calibration parameters.

### 7. Track bubble motion and breakup events

```bash
python 5_track_bubbles_kinematics.py \
  --masks ../images/segmented_masks \
  --output-dir ../results \
  --fps 2000 \
  --pixel-size 0.067 \
  --max-disp 50
```

This produces CSV outputs such as:

- `bubble_velocities.csv`
- `bubble_breakups.csv`

## Output artifacts

The repository is organized to keep generated data outside version control. Typical artifacts include:

- `dataset/images/` and `dataset/masks/` for training data
- `models/*.pth` for trained weights
- `results/*.csv` for analysis outputs
- `images/` or similar local folders with raw videos, crops, and predicted masks

## Notes

- Input data folders and model checkpoints are intentionally not tracked by git.
- Many scripts expect explicit file paths and may require adjusting constants such as ROI cropping, pixel calibration, pipe diameter, and flow rate.
- This project is research-oriented and assumes domain-specific knowledge of the two-phase flow setup.

## Suggested next steps

- Add a `requirements.txt` file for reproducible installs
- Add unit tests around preprocessing and mask export routines
- Standardize folder names across scripts for easier automation
- Add a single CLI entry point to orchestrate the end-to-end workflow

## License

This project does not currently declare a repository license. If you intend to distribute or publish this code, add an appropriate license file and update the project metadata accordingly.
