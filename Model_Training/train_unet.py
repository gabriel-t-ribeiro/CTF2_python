from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


# 1. Dataset Loader
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

        # Normalize mask to binary float [0.0, 1.0]
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # Ensure image has channel dimension: (1, H, W)
        if isinstance(img, np.ndarray):
            img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
            img = torch.tensor(img)
            mask = torch.tensor(mask).unsqueeze(0)
        else:
            img = img.float() / 255.0
            mask = mask.unsqueeze(0).float()

        return img, mask


# 2. Combined Loss Function
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


# 3. Augmentations (Multiples of 32 required for U-Net pooling)
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

# 4. Training Setup
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
dataset_dir = Path("dataset")

all_images = sorted(list((dataset_dir / "images").glob("*.png")))
all_masks = sorted(list((dataset_dir / "masks").glob("*.png")))

# 80/20 Train/Validation Split
split_idx = int(0.8 * len(all_images))
train_imgs, val_imgs = all_images[:split_idx], all_images[split_idx:]
train_masks, val_masks = all_masks[:split_idx], all_masks[split_idx:]

train_loader = DataLoader(
    BubbleDataset(train_imgs, train_masks, train_transform),
    batch_size=8,
    shuffle=True,
    num_workers=2,
)
val_loader = DataLoader(
    BubbleDataset(val_imgs, val_masks, val_transform), batch_size=8, shuffle=False
)

# Model: ResNet34 Encoder, Single-channel Grayscale Input
model = smp.Unet(
    encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=1
).to(device)

criterion = DiceBCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

# 5. Training Loop
EPOCHS = 40
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * images.size(0)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)

    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_bubble_unet.pth")
        print(" --> Model checkpoint saved.")
