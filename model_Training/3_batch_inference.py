import argparse
from pathlib import Path
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch


def run_inference(input_dir: Path, output_dir: Path, model_weights: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1
    )
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

    print(
        f"Segmenting {len(image_files)} frames from '{input_dir}' -> '{output_dir}'..."
    )

    with torch.no_grad():
        for i, img_path in enumerate(image_files, 1):
            raw_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            orig_h, orig_w = raw_img.shape

            resized_img = cv2.resize(raw_img, (512, 512))
            input_tensor = (
                torch.tensor(resized_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                / 255.0
            ).to(device)

            logits = model(input_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

            pred_mask = (probs > 0.5).astype(np.uint8) * 255
            final_mask = cv2.resize(
                pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
            )

            out_path = output_dir / f"{img_path.stem}_mask.png"
            cv2.imwrite(str(out_path), final_mask)

            if i % 100 == 0 or i == len(image_files):
                print(f"  Processed [{i}/{len(image_files)}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch U-Net segmentation on raw flow frames"
    )
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
    args = parser.parse_args()

    run_inference(args.input, args.output, args.model)
