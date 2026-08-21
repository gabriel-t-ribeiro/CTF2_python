from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


# 1. Dataset Definition
class BubbleDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)

        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        if isinstance(img, np.ndarray):
            img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
            img = torch.tensor(img)
            mask = torch.tensor(mask).unsqueeze(0)
        else:
            img = img.float() / 255.0
            mask = mask.unsqueeze(0).float()

        return img, mask


# 2. Combined Loss
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1.0
        intersection = (probs * targets).sum()
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (
            probs.sum() + targets.sum() + smooth
        )
        return bce_loss + dice_loss


def train_model():
    BASE_DIR = Path(__file__).resolve().parent
    DATASET_DIR = BASE_DIR / "dataset"

    # Augmentations (resized to multiples of 32 for U-Net pooling)
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

    all_images = sorted(list((DATASET_DIR / "images").glob("*.png")))
    all_masks = sorted(list((DATASET_DIR / "masks").glob("*.png")))

    if len(all_images) == 0:
        raise FileNotFoundError(f"No training pairs found in {DATASET_DIR}")

    split_idx = int(0.8 * len(all_images))
    train_loader = DataLoader(
        BubbleDataset(all_images[:split_idx], all_masks[:split_idx], train_transform),
        batch_size=8,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        BubbleDataset(all_images[split_idx:], all_masks[split_idx:], val_transform),
        batch_size=8,
        shuffle=False,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Training on device: {device}")

    # ResNet34 Encoder with pre-trained ImageNet weights
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    ).to(device)

    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    epochs = 40
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(imgs)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * imgs.size(0)

        # Evaluation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)
                val_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        print(
            f"Epoch [{epoch:02d}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), BASE_DIR / "best_bubble_unet.pth")
            print("  --> Model checkpoint saved.")


if __name__ == "__main__":
    train_model()
