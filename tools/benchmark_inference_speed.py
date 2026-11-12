#!/usr/bin/env python3
import argparse
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


# -------------------------------------------------------------------------
# Define MATLAB-equivalent 4-depth U-Net architecture (28.5M parameters)
# -------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class MatlabUnetEquivalent(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.c1 = DoubleConv(in_ch, 64)
        self.p1 = nn.MaxPool2d(2)
        self.c2 = DoubleConv(64, 128)
        self.p2 = nn.MaxPool2d(2)
        self.c3 = DoubleConv(128, 256)
        self.p3 = nn.MaxPool2d(2)
        self.c4 = DoubleConv(256, 512)
        self.p4 = nn.MaxPool2d(2)
        self.bottom = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.e4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.e3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.e2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.e1 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(self.p1(x1))
        x3 = self.c3(self.p2(x2))
        x4 = self.c4(self.p3(x3))
        b = self.bottom(self.p4(x4))

        d4 = self.e4(torch.cat([self.up4(b), x4], dim=1))
        d3 = self.e3(torch.cat([self.up3(d4), x3], dim=1))
        d2 = self.e2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.e1(torch.cat([self.up1(d2), x1], dim=1))
        return self.final(d1)


def benchmark_forward_pass(
    batch_sizes: tuple[int, ...] = (1, 4, 8, 16), n_warmup: int = 5, n_runs: int = 20
) -> list[dict]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("\n=======================================================")
    print(f"  BENCHMARK 1: PURE NEURAL NETWORK FORWARD PASS ({device})")
    print("=======================================================")

    # 1. MATLAB U-Net Equivalent (28.5M params, input 304x832)
    matlab_model = MatlabUnetEquivalent().to(device).eval()
    matlab_params = sum(p.numel() for p in matlab_model.parameters())
    print(f"MATLAB U-Net: {matlab_params:,} parameters (Input: 304x832)")

    # 2. Python ResNet-34 U-Net (24.4M params, input 512x512)
    python_model = smp.Unet(encoder_name="resnet34", in_channels=1, classes=1).to(device).eval()
    python_params = sum(p.numel() for p in python_model.parameters())
    print(f"Python ResNet-34 U-Net: {python_params:,} parameters (Input: 512x512)")

    results = []

    for bs in batch_sizes:
        # Benchmark Python Model
        x_py = torch.randn(bs, 1, 512, 512, device=device)
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = python_model(x_py)
            if device.type == "mps":
                torch.mps.synchronize()

            t0 = time.perf_counter()
            for _ in range(n_runs):
                _ = python_model(x_py)
            if device.type == "mps":
                torch.mps.synchronize()
            py_total_time = (time.perf_counter() - t0) / n_runs
            py_ms_frame = (py_total_time / bs) * 1000.0
            py_fps = bs / py_total_time

        # Benchmark MATLAB Model
        x_mat = torch.randn(bs, 1, 304, 832, device=device)
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = matlab_model(x_mat)
            if device.type == "mps":
                torch.mps.synchronize()

            t0 = time.perf_counter()
            for _ in range(n_runs):
                _ = matlab_model(x_mat)
            if device.type == "mps":
                torch.mps.synchronize()
            mat_total_time = (time.perf_counter() - t0) / n_runs
            mat_ms_frame = (mat_total_time / bs) * 1000.0
            mat_fps = bs / mat_total_time

        print(f"Batch Size = {bs:2d}:")
        print(f"  Python ResNet-34 : {py_ms_frame:6.2f} ms/frame | {py_fps:6.1f} FPS")
        print(f"  MATLAB 4-D U-Net: {mat_ms_frame:6.2f} ms/frame | {mat_fps:6.1f} FPS")

        results.append(
            {
                "stage": "Forward Pass (Model Only)",
                "batch_size": bs,
                "python_ms_per_frame": py_ms_frame,
                "python_fps": py_fps,
                "matlab_ms_per_frame": mat_ms_frame,
                "matlab_fps": mat_fps,
                "speedup_ratio": mat_ms_frame / py_ms_frame,
            }
        )

    return results


def benchmark_end_to_end(
    input_dir: Path | None = None,
    model_path: Path = Path("models/run_2_unet.pth"),
    max_frames: int = 50,
    batch_sizes: tuple[int, ...] = (1, 16),
) -> list[dict]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("\n=======================================================")
    print("  BENCHMARK 2: END-TO-END BATCH INFERENCE PIPELINE")
    print("  (Disk Read -> Resize -> Forward -> Threshold -> Save)")
    print("=======================================================")

    if input_dir is None:
        input_dir = Path("images/background1/background_removed")
        if not input_dir.exists():
            input_dir = Path("dataset_2/images")

    test_files = sorted(list(input_dir.glob("*.png")) + list(input_dir.glob("*.tif")))[:max_frames]
    num_files = len(test_files)
    if num_files == 0:
        raise FileNotFoundError(f"No .png or .tif frames found in: {input_dir}")
    print(f"Benchmarking on {num_files} actual flow frames from {input_dir}...")

    # Load Python model
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    output_dir = Path(tempfile.mkdtemp(prefix="ctf2_benchmark_"))

    e2e_results = []

    for bs in batch_sizes:
        t_read_resize = 0.0
        t_model = 0.0
        t_post_save = 0.0
        t_start = time.perf_counter()

        with torch.no_grad():
            for b_idx in range(0, num_files, bs):
                batch_paths = test_files[b_idx : b_idx + bs]

                # Step 1: Read & Preprocess
                t0 = time.perf_counter()
                tensors = []
                orig_shapes = []
                for p in batch_paths:
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    orig_shapes.append(img.shape)
                    img_512 = cv2.resize(img, (512, 512))
                    tensors.append(img_512)
                batch_arr = np.stack(tensors, axis=0)
                input_t = (torch.from_numpy(batch_arr).unsqueeze(1).float() / 255.0).to(device)
                t_read_resize += time.perf_counter() - t0

                # Step 2: Model Forward Pass
                t1 = time.perf_counter()
                logits = model(input_t)
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                if probs.ndim == 2:
                    probs = np.expand_dims(probs, axis=0)
                if device.type == "mps":
                    torch.mps.synchronize()
                t_model += time.perf_counter() - t1

                # Step 3: Post-process & Save
                t2 = time.perf_counter()
                for prob_map, (orig_h, orig_w), p in zip(probs, orig_shapes, batch_paths):
                    mask_512 = (prob_map > 0.5).astype(np.uint8) * 255
                    mask_orig = cv2.resize(
                        mask_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                    )
                    out_p = output_dir / f"{p.stem}_mask.png"
                    cv2.imwrite(str(out_p), mask_orig)
                t_post_save += time.perf_counter() - t2

        total_time = time.perf_counter() - t_start
        fps = num_files / total_time
        ms_per_frame = (total_time / num_files) * 1000.0

        print(
            f"\nBatch Size = {bs:2d}: Total Time = {total_time:.2f} s | Latency = {ms_per_frame:.2f} ms/frame | Throughput = {fps:.2f} FPS"
        )
        print(f"  - Disk Read & Resize        : {(t_read_resize / num_files) * 1000:.2f} ms/frame")
        print(f"  - Model Forward Pass (MPS)  : {(t_model / num_files) * 1000:.2f} ms/frame")
        print(f"  - Postprocess & Disk Write  : {(t_post_save / num_files) * 1000:.2f} ms/frame")

        e2e_results.append(
            {
                "stage": f"End-to-End Pipeline (batch_size={bs})",
                "batch_size": bs,
                "total_ms_per_frame": ms_per_frame,
                "fps": fps,
                "io_read_resize_ms": (t_read_resize / num_files) * 1000,
                "model_forward_ms": (t_model / num_files) * 1000,
                "post_save_ms": (t_post_save / num_files) * 1000,
            }
        )

    shutil.rmtree(output_dir)

    return e2e_results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark U-Net forward-pass and end-to-end inference speed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="Frames for the end-to-end benchmark (default: background-removed run, else dataset_2)",
    )
    parser.add_argument("--model", "-m", type=Path, default=Path("models/run_2_unet.pth"))
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("models"))
    parser.add_argument("--max-frames", type=int, default=50, help="Frames in the end-to-end run")
    parser.add_argument("--runs", type=int, default=20, help="Timed forward passes per batch size")
    parser.add_argument(
        "--quick", action="store_true", help="Smoke run: batch size 1 only, 2 timed passes"
    )
    args = parser.parse_args(argv)

    if args.quick:
        fwd_res = benchmark_forward_pass(batch_sizes=(1,), n_warmup=1, n_runs=2)
        e2e_res = benchmark_end_to_end(args.input, args.model, args.max_frames, batch_sizes=(1,))
    else:
        fwd_res = benchmark_forward_pass(n_runs=args.runs)
        e2e_res = benchmark_end_to_end(args.input, args.model, args.max_frames)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fwd_csv = args.output_dir / "benchmark_forward_pass.csv"
    e2e_csv = args.output_dir / "benchmark_end_to_end.csv"
    pd.DataFrame(fwd_res).to_csv(fwd_csv, index=False)
    pd.DataFrame(e2e_res).to_csv(e2e_csv, index=False)
    print(f"\nBenchmark results saved to {fwd_csv} and {e2e_csv}")


if __name__ == "__main__":
    main()
