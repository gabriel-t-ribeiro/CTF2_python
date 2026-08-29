# Model training pipeline

This folder contains the deep-learning and post-processing workflow for bubble segmentation and flow analysis in the
CTF2 Python project.

The scripts here take labeled image pairs, train a U-Net model, run inference on full image sequences, and convert
segmentation outputs into physical measurements such as void fraction and bubble dynamics.

## Folder purpose

The training pipeline is organized into five main stages:

1. Annotation conversion
2. Model training
3. Batch inference
4. Void-fraction reconstruction
5. Bubble tracking and kinematics

## Scripts

### 1_convert_annotations.py

Converts exported annotation data into the binary mask format expected by the training process.

Typical use:

```bash
python 1_convert_annotations.py --json exports/experiment_1_export.json --dataset ../dataset
```

Purpose:

- prepare image/mask pairs for the training dataset
- ensure ground-truth labels are binary and aligned with image frames

### 2_train_u_net.py

Trains a U-Net segmentation model on grayscale input images and binary bubble masks.

Typical use:

```bash
python 2_train_u_net.py \
  --dataset ../dataset \
  --output-model ../models/best_bubble_unet.pth \
  --epochs 40 \
  --batch-size 8 \
  --lr 1e-3
```

Key features:

- 80/20 train/validation split
- ResNet-34 encoder backbone
- grayscale single-channel input
- Dice + BCE loss
- automatic checkpoint saving for the best validation result

### 3_batch_inference.py

Runs inference over a folder of raw images and saves binary segmentation masks.

Typical use:

```bash
python 3_batch_inference.py \
  --input ../images/raw_frames \
  --output ../images/segmented_masks \
  --model ../models/best_bubble_unet.pth
```

Purpose:

- generate full-mask predictions for each frame
- prepare outputs for downstream physics analysis

### 4_reconstruct_void_fraction.py

Computes the gas volume and void fraction from binary masks using geometric assumptions and pipe calibration parameters.

Typical use:

```bash
python 4_reconstruct_void_fraction.py \
  --masks ../images/segmented_masks \
  --output-csv ../results/void_fraction_results.csv \
  --pixel-size 0.067 \
  --diameter 12.0
```

Outputs:

- a CSV with frame-wise gas volume and void fraction values
- summary metrics: mean, min, max, and standard deviation of alpha
- optional Rouhani-Axelsson theoretical benchmark comparison (`--vapor-quality`, `--mass-flux`)

### 5_track_bubbles_kinematics.py

Tracks bubble centroids across successive mask frames and estimates velocity and breakup events.

Typical use:

```bash
python 5_track_bubbles_kinematics.py \
  --masks ../images/segmented_masks \
  --output-dir ../results \
  --fps 2000 \
  --pixel-size 0.067 \
  --max-disp 50
```

Outputs:

- `bubble_velocities.csv`
- `bubble_breakups.csv`

### evaluate_unet_metrics.py

Evaluates a trained model on the validation split and prints segmentation metrics such as:

- accuracy
- precision
- recall
- specificity
- F1 score / Dice
- IoU / Jaccard
- MCC

Typical use:

```bash
python evaluate_unet_metrics.py \
  --model ../models/best_bubble_unet.pth \
  --dataset ../dataset \
  --output-csv ../results/model_validation_metrics.csv
```

## Expected data layout

The scripts assume a folder structure similar to:

```text
project/
├── dataset/
│   ├── images/
│   └── masks/
├── images/
│   ├── raw_frames/
│   └── segmented_masks/
├── models/
│   └── best_bubble_unet.pth
├── results/
│   ├── void_fraction_results.csv
│   ├── bubble_velocities.csv
│   └── bubble_breakups.csv
└── model_training/
```

## Dependencies

The training pipeline relies on:

- Python 3.10+
- OpenCV
- NumPy
- Pandas
- SciPy
- PyTorch
- Albumentations
- segmentation-models-pytorch

## Recommended workflow

```bash
cd model_training

# 1. Prepare training masks from annotations
python 1_convert_annotations.py --json ../image_scripts/preannotations.json --dataset ../dataset

# 2. Train the segmentation model
python 2_train_u_net.py --dataset ../dataset --output-model ../models/best_bubble_unet.pth

# 3. Assess model quality
python evaluate_unet_metrics.py --model ../models/best_bubble_unet.pth --dataset ../dataset

# 4. Run full-sequence inference
python 3_batch_inference.py --input ../images/raw_frames --output ../images/segmented_masks --model ../models/best_bubble_unet.pth

# 5. Compute void fraction
python 4_reconstruct_void_fraction.py --masks ../images/segmented_masks --output-csv ../results/void_fraction_results.csv --pixel-size 0.067 --diameter 12.0

# 6. Track bubble motion
python 5_track_bubbles_kinematics.py --masks ../images/segmented_masks --output-dir ../results --fps 2000 --pixel-size 0.067 --max-disp 50
```

## Notes

- Most scripts expect explicit paths and calibration values.
- This project is research-oriented and depends on the physical setup of the flow experiment.
- Generated masks, checkpoints, and CSV outputs are usually kept outside version control.

## Related project files

- `image_scripts/` for preprocessing and annotation seed generation
- root `README.md` for project-wide overview
- `.gitignore` for generated dataset and model files
