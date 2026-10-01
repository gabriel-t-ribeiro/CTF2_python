#!/usr/bin/env python3
import argparse
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.utils.data import DataLoader, Dataset


class BubbleDataset(Dataset):
    """
    PyTorch Dataset for loading single-channel bubble images and binary masks.
    Applies data augmentations via Albumentations and ensures 4D batch compatibility (B, C, H, W).
    """

    def __init__(self, image_paths: list[Path], mask_paths: list[Path], transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Load image and mask in grayscale (H, W)
        img = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Could not load image: {self.image_paths[idx]}")
        if mask is None:
            raise FileNotFoundError(f"Could not load mask: {self.mask_paths[idx]}")

        # Binarize mask to 0.0 or 1.0
        mask = (mask > 127).astype(np.float32)

        # 2. Apply Albumentations spatial & intensity transformations
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # 3. Convert to torch.Tensor and enforce (1, H, W) channel dimension
        if isinstance(img, np.ndarray):
            img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
            img = torch.from_numpy(img)
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        else:
            # Albumentations ToTensorV2 produces a 2D Tensor (H, W) for grayscale images; add channel axis
            if img.ndim == 2:
                img = img.unsqueeze(0)
            img = img.float() / 255.0

            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = mask.float()

        return img, mask


class DiceBCELoss(nn.Module):
    """
    Combined Binary Cross-Entropy (BCE) and Dice Loss.
    Addresses extreme foreground-background class imbalance in bubble segmentation.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1.0
        intersection = (probs * targets).sum()
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
        return bce_loss + dice_loss


def train(
    dataset_dir: Path,
    output_model: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int = 0,
):
    train_transform = A.Compose(
        [
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose([A.Resize(512, 512), ToTensorV2()])

    # Retrieve and sort dataset pairs with multi-extension and stem-matching support
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    img_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"
    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Missing 'images' or 'masks' subdirectories inside: {dataset_dir}")

    all_images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in valid_exts])
    if not all_images:
        raise FileNotFoundError(f"No valid image files found inside: {img_dir}")

    all_masks = []
    for img_p in all_images:
        cand = mask_dir / img_p.name
        if not cand.exists():
            cand = mask_dir / f"{img_p.stem}_mask.png"
        if not cand.exists():
            matches = list(mask_dir.glob(f"{img_p.stem}.*")) + list(
                mask_dir.glob(f"{img_p.stem}_mask.*")
            )
            if matches:
                cand = matches[0]
        if not cand.exists():
            raise FileNotFoundError(f"Matching mask for '{img_p.name}' not found in {mask_dir}")
        all_masks.append(cand)

    # 80/20 train/validation split with safety guardrail against ZeroDivisionError
    if len(all_images) > 1:
        split_idx = max(1, min(len(all_images) - 1, int(0.8 * len(all_images))))
        train_imgs, train_msks = all_images[:split_idx], all_masks[:split_idx]
        val_imgs, val_msks = all_images[split_idx:], all_masks[split_idx:]
    else:
        train_imgs, train_msks = all_images, all_masks
        val_imgs, val_msks = all_images, all_masks

    train_dataset = BubbleDataset(train_imgs, train_msks, train_transform)
    val_dataset = BubbleDataset(val_imgs, val_msks, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Training on device: {device}")

    # Initialize U-Net with ResNet-34 backbone pre-trained on ImageNet
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    ).to(device)

    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    is_cuda = device.type == "cuda"
    scaler = (
        torch.amp.GradScaler("cuda", enabled=is_cuda)
        if hasattr(torch.amp, "GradScaler")
        else torch.cuda.amp.GradScaler(enabled=is_cuda)
    )

    import time

    training_start = time.perf_counter()
    best_loss = float("inf")
    output_model.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss = 0.0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", enabled=is_cuda):
                outputs = model(imgs)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * imgs.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                with torch.amp.autocast(device_type="cuda", enabled=is_cuda):
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)
                val_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        epoch_duration = time.perf_counter() - epoch_start

        print(
            f"Epoch [{epoch:02d}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_duration:.2f}s"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), output_model)
            print(f"  --> Saved best model checkpoint to: {output_model}")

    total_training_time = time.perf_counter() - training_start
    print(
        f"[TIMING] Total Training Time: {total_training_time:.2f}s "
        f"({total_training_time / epochs:.2f}s/epoch, "
        f"{total_training_time / (epochs * len(train_dataset)):.4f}s/sample)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net on bubble dataset")
    parser.add_argument(
        "--dataset",
        "-d",
        type=Path,
        default=Path("dataset"),
        help="Root dataset directory (must contain 'images/' and 'masks/')",
    )
    parser.add_argument(
        "--output-model",
        "-o",
        type=Path,
        default=Path("best_bubble_unet.pth"),
        help="Path where the best model weights will be saved",
    )
    parser.add_argument("--epochs", "-e", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes (default: 0 for macOS compatibility)",
    )
    args = parser.parse_args()

    train(
        args.dataset,
        args.output_model,
        args.epochs,
        args.batch_size,
        args.lr,
        num_workers=args.num_workers,
    )
