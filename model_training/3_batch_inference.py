#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch


def run_inference(
    input_dir: Path,
    output_dir: Path,
    model_weights: Path,
    batch_size: int = 16,
    timing_csv: Path | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Loading model '{model_weights}' on device: {device}...")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.to(device)
    model.eval()

    image_files = sorted(
        [
            p
            for p in input_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".bmp"}
        ]
    )
    if not image_files:
        raise FileNotFoundError(f"No raw images found in {input_dir}")

    total_frames = len(image_files)
    print(
        f"Segmenting {total_frames} frames from '{input_dir}' -> '{output_dir}' (batch_size={batch_size})..."
    )

    start_time = time.perf_counter()
    processed_count = 0

    with torch.no_grad():
        for b_idx in range(0, total_frames, batch_size):
            batch_paths = image_files[b_idx : b_idx + batch_size]
            batch_tensors = []
            orig_shapes = []
            valid_paths = []

            for img_p in batch_paths:
                raw_img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
                if raw_img is None:
                    print(f"  [skip] Could not read image: {img_p.name}")
                    continue
                orig_shapes.append(raw_img.shape)
                valid_paths.append(img_p)
                resized_img = cv2.resize(raw_img, (512, 512))
                batch_tensors.append(resized_img)

            if not batch_tensors:
                continue

            batch_np = np.stack(batch_tensors, axis=0)  # (B, 512, 512)
            input_tensor = (torch.from_numpy(batch_np).unsqueeze(1).float() / 255.0).to(device)

            logits = model(input_tensor)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()  # (B, 512, 512)
            if probs.ndim == 2:
                probs = np.expand_dims(probs, axis=0)

            for p_map, (orig_h, orig_w), img_p in zip(probs, orig_shapes, valid_paths):
                pred_512 = (p_map > 0.5).astype(np.uint8) * 255
                final_mask = cv2.resize(pred_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                out_path = output_dir / f"{img_p.stem}_mask.png"
                cv2.imwrite(str(out_path), final_mask)
                processed_count += 1

            if processed_count % 100 == 0 or processed_count == total_frames:
                elapsed_so_far = time.perf_counter() - start_time
                curr_fps = processed_count / max(1e-5, elapsed_so_far)
                print(
                    f"  Processed [{processed_count}/{total_frames}] "
                    f"({processed_count / total_frames * 100:.1f}%) - Speed: {curr_fps:.1f} FPS"
                )

    if device.type == "mps":
        torch.mps.synchronize()

    total_time = time.perf_counter() - start_time
    fps = processed_count / max(1e-5, total_time)
    ms_per_frame = (total_time / max(1, processed_count)) * 1000.0

    print("\n" + "=" * 55)
    print("         BATCH INFERENCE BENCHMARK REPORT         ")
    print("=" * 55)
    print(f"Input Directory    : {input_dir}")
    print(f"Output Directory   : {output_dir}")
    print(f"Model Checkpoint   : {model_weights}")
    print(f"Frames Processed   : {processed_count:,}")
    print(f"Total Time Taken   : {total_time:.2f} s ({total_time / 60.0:.2f} min)")
    print(f"Throughput Speed   : {fps:.2f} FPS")
    print(f"Latency per Frame  : {ms_per_frame:.2f} ms/frame")
    print("=" * 55 + "\n")

    if timing_csv:
        timing_csv.parent.mkdir(parents=True, exist_ok=True)
        timing_data = {
            "Input_Dir": str(input_dir),
            "Output_Dir": str(output_dir),
            "Model_Weights": str(model_weights),
            "Frames_Processed": processed_count,
            "Total_Time_s": round(total_time, 3),
            "Throughput_FPS": round(fps, 2),
            "Latency_ms_per_frame": round(ms_per_frame, 2),
        }
        df = pd.DataFrame([timing_data])
        if timing_csv.exists():
            df.to_csv(timing_csv, mode="a", header=False, index=False)
        else:
            df.to_csv(timing_csv, index=False)
        print(f"[TIMING LOG] Benchmark recorded to: {timing_csv}")

    return {
        "frames": processed_count,
        "total_time_s": total_time,
        "fps": fps,
        "latency_ms": ms_per_frame,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch U-Net segmentation on raw flow frames")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input folder containing cropped raw images",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output folder for binary segmentation masks",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=Path,
        default=Path("best_bubble_unet.pth"),
        help="Path to trained U-Net weights (.pth)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=16,
        help="Inference batch size",
    )
    parser.add_argument(
        "--timing-csv",
        type=Path,
        default=None,
        help="Optional CSV file to log execution benchmarks",
    )
    args = parser.parse_args()

    run_inference(args.input, args.output, args.model, args.batch_size, args.timing_csv)
