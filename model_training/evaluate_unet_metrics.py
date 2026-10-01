#!/usr/bin/env python3
import argparse
import ssl
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch

# SSL bypass for macOS pretrained weight downloads if needed
ssl._create_default_https_context = ssl._create_unverified_context


def find_matching_mask(img_path: Path, masks_dir: Path) -> Path:
    """
    Finds the corresponding ground-truth mask file for a given image file.
    Supports identical filename (e.g. frame_0001.png) or stem_mask (e.g. frame_0001_mask.png).
    """
    # 1. Check exact filename match
    candidate = masks_dir / img_path.name
    if candidate.exists():
        return candidate

    # 2. Check stem_mask.png match
    candidate = masks_dir / f"{img_path.stem}_mask.png"
    if candidate.exists():
        return candidate

    # 3. Check matching stem with any image extension
    matches = list(masks_dir.glob(f"{img_path.stem}.*")) + list(
        masks_dir.glob(f"{img_path.stem}_mask.*")
    )
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Ground-truth mask for '{img_path.name}' not found in '{masks_dir}'")


def evaluate_model(model_path: Path, dataset_dir: Path, output_csv: Path):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Running evaluation on device: {device}")

    # 1. Load Trained U-Net Weights
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Get Validation Pairs (Using the 20% validation split)
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"

    all_images = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".bmp"}
        ]
    )

    if not all_images:
        raise FileNotFoundError(f"No validation images found in: {images_dir}")

    if len(all_images) > 1:
        split_idx = max(1, min(len(all_images) - 1, int(0.8 * len(all_images))))
        val_images = all_images[split_idx:]
    else:
        val_images = all_images

    print(f"Evaluating on {len(val_images)} validation frame(s)...")

    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    with torch.no_grad():
        for img_p in val_images:
            mask_p = find_matching_mask(img_p, masks_dir)

            raw_img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)

            if raw_img is None:
                print(f"[SKIP] Could not read image: {img_p.name}")
                continue
            if gt_mask is None:
                print(f"[SKIP] Could not read mask: {mask_p.name}")
                continue

            orig_h, orig_w = raw_img.shape

            # Resize to 512x512 for U-Net forward pass
            resized_img = cv2.resize(raw_img, (512, 512))
            input_tensor = (
                torch.tensor(resized_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
            ).to(device)

            logits = model(input_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

            # Restore original aspect ratio and threshold
            pred_512 = (probs > 0.5).astype(np.uint8)
            pred_mask = cv2.resize(pred_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            gt_binary = (gt_mask > 127).astype(np.uint8)

            # Accumulate pixel counts
            tp = np.sum((pred_mask == 1) & (gt_binary == 1))
            fp = np.sum((pred_mask == 1) & (gt_binary == 0))
            fn = np.sum((pred_mask == 0) & (gt_binary == 1))
            tn = np.sum((pred_mask == 0) & (gt_binary == 0))

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

    # 3. Compute Metrics
    eps = 1e-7
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + eps)
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    specificity = total_tn / (total_tn + total_fp + eps)
    f1_dice = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn + eps)
    jaccard_iou = total_tp / (total_tp + total_fp + total_fn + eps)

    mcc_num = (total_tp * total_tn) - (total_fp * total_fn)
    mcc_den = np.sqrt(
        float(total_tp + total_fp)
        * float(total_tp + total_fn)
        * float(total_tn + total_fp)
        * float(total_tn + total_fn)
    )
    mcc = mcc_num / (mcc_den + eps)

    # 4. Display Report
    print("\n" + "=" * 45)
    print("       U-NET SEGMENTATION PERFORMANCE       ")
    print("=" * 45)
    print(f"Total Pixels Analyzed : {total_tp + total_tn + total_fp + total_fn:,}")
    print(f"True Positives  (TP)  : {total_tp:,}")
    print(f"True Negatives  (TN)  : {total_tn:,}")
    print(f"False Positives (FP)  : {total_fp:,}")
    print(f"False Negatives (FN)  : {total_fn:,}")
    print("-" * 45)
    print(f"Accuracy              : {accuracy:.4f}")
    print(f"Precision             : {precision:.4f}")
    print(f"Recall (Sensitivity)  : {recall:.4f}")
    print(f"Specificity           : {specificity:.4f}")
    print(f"F1-Score (Dice)       : {f1_dice:.4f}")
    print(f"Jaccard Index (IoU)   : {jaccard_iou:.4f}")
    print(f"MCC                   : {mcc:.4f}")
    print("=" * 45)

    # Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "Specificity": specificity,
                "F1_Dice": f1_dice,
                "Jaccard_IoU": jaccard_iou,
                "MCC": mcc,
                "TP": total_tp,
                "TN": total_tn,
                "FP": total_fp,
                "FN": total_fn,
            }
        ]
    ).to_csv(output_csv, index=False)
    print(f"[SAVED] Results exported to: {output_csv}")
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1_Dice": f1_dice,
        "Jaccard_IoU": jaccard_iou,
        "MCC": mcc,
        "TP": total_tp,
        "TN": total_tn,
        "FP": total_fp,
        "FN": total_fn,
    }


def compare_models(output_csv: Path):
    configs = [
        ("run_1_unet", Path("models/run_1_unet.pth"), "dataset (raw with bg)", Path("dataset")),
        (
            "run_2_unet",
            Path("models/run_2_unet.pth"),
            "dataset_2_no_background",
            Path("dataset_2_no_background"),
        ),
        (
            "run_2_with_bg",
            Path("models/run_2_with_bg.pth"),
            "dataset_2 (raw with bg)",
            Path("dataset_2"),
        ),
        # 1-to-1 Controlled Pair: Same 20 frames tested on NO BG vs WITH BG
        (
            "run_2_unet (cross)",
            Path("models/run_2_unet.pth"),
            "dataset_2 (raw with bg)",
            Path("dataset_2"),
        ),
        (
            "run_2_with_bg (cross)",
            Path("models/run_2_with_bg.pth"),
            "dataset_2_no_background",
            Path("dataset_2_no_background"),
        ),
        # Original Model tested on dataset_2 (with background)
        (
            "run_1_unet (cross)",
            Path("models/run_1_unet.pth"),
            "dataset_2 (raw with bg)",
            Path("dataset_2"),
        ),
        (
            "run_1_unet (cross)",
            Path("models/run_1_unet.pth"),
            "dataset_2_no_background",
            Path("dataset_2_no_background"),
        ),
        (
            "run_2_with_bg (cross)",
            Path("models/run_2_with_bg.pth"),
            "dataset (raw with bg)",
            Path("dataset"),
        ),
        (
            "run_2_unet (cross)",
            Path("models/run_2_unet.pth"),
            "dataset (raw with bg)",
            Path("dataset"),
        ),
    ]

    records = []
    print("\n" + "=" * 65)
    print("      RUNNING FULL CROSS-DATASET MODEL COMPARISON MATRIX      ")
    print("=" * 65)
    for model_name, m_path, d_name, d_path in configs:
        if not m_path.exists():
            print(f"[SKIP] Model not found: {m_path}")
            continue
        if not d_path.exists():
            print(f"[SKIP] Dataset not found: {d_path}")
            continue

        print(f"\n--> Evaluating {model_name} on {d_name}...")
        tmp_csv = Path("/tmp/eval_tmp.csv")
        metrics = evaluate_model(m_path, d_path, tmp_csv)
        metrics["Model"] = model_name
        metrics["Dataset"] = d_name
        metrics["Model_Path"] = str(m_path)
        metrics["Dataset_Path"] = str(d_path)
        records.append(metrics)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    cols = [
        "Model",
        "Dataset",
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1_Dice",
        "Jaccard_IoU",
        "MCC",
        "TP",
        "TN",
        "FP",
        "FN",
        "Model_Path",
        "Dataset_Path",
    ]
    df[cols].to_csv(output_csv, index=False)
    print("\n" + "=" * 65)
    print(f"[SAVED] Complete model comparison matrix saved to: {output_csv}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate U-Net validation metrics")
    parser.add_argument(
        "--model",
        "-m",
        type=Path,
        default=Path("models/run_1_unet.pth"),
        help="Path to trained model weights (.pth)",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=Path,
        default=Path("dataset"),
        help="Dataset folder containing images/ and masks/",
    )
    parser.add_argument(
        "--output-csv",
        "-o",
        type=Path,
        default=Path("models/model_validation_metrics.csv"),
        help="Path to output CSV report",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comprehensive comparison matrix between run_1_unet and run_2_unet",
    )
    args = parser.parse_args()

    if args.compare:
        compare_models(args.output_csv)
    else:
        evaluate_model(args.model, args.dataset, args.output_csv)
