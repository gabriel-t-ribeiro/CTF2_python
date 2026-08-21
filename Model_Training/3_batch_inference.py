from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

BASE_DIR = Path(__file__).resolve().parent
INPUT_RAW_DIR = BASE_DIR / "Images" / "images_2"  # Cropped raw frames
OUTPUT_MASK_DIR = BASE_DIR / "segmented_masks"
MODEL_WEIGHTS = BASE_DIR / "best_bubble_unet.pth"

OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)


def run_batch_inference():
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
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.to(device)
    model.eval()

    image_files = sorted(
        [
            p
            for p in INPUT_RAW_DIR.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".tif", ".bmp"}
        ]
    )
    print(f"Running inference on {len(image_files)} frames...")

    with torch.no_grad():
        for i, img_path in enumerate(image_files, 1):
            raw_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            orig_h, orig_w = raw_img.shape

            # Resize to 512x512 for U-Net tensor forward pass
            resized_img = cv2.resize(raw_img, (512, 512))
            input_tensor = (
                torch.tensor(resized_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                / 255.0
            )
            input_tensor = input_tensor.to(device)

            logits = model(input_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

            # Threshold at 0.5 and restore original aspect ratio
            pred_mask = (probs > 0.5).astype(np.uint8) * 255
            final_mask = cv2.resize(
                pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
            )

            out_path = OUTPUT_MASK_DIR / f"{img_path.stem}_mask.png"
            cv2.imwrite(str(out_path), final_mask)

            if i % 100 == 0 or i == len(image_files):
                print(f"Processed [{i}/{len(image_files)}]")

    print(f"All binary masks saved to: {OUTPUT_MASK_DIR}")


if __name__ == "__main__":
    run_batch_inference()
