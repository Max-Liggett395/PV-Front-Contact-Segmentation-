"""SEM image dataset for semantic segmentation."""

import os

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES = ["background", "silver", "glass", "silicon", "void", "interfacial_void"]
CLASS_COLORS = [
    (255, 0, 0),      # background - red
    (0, 255, 0),      # silver - green
    (0, 0, 255),      # glass - blue
    (255, 255, 0),    # silicon - yellow
    (255, 0, 255),    # void - magenta
    (0, 255, 255),    # interfacial void - cyan
]
NUM_CLASSES = len(CLASS_NAMES)


class SEMDataset(Dataset):
    """Dataset for SEM semantic segmentation.

    Args:
        image_paths: list of image file paths
        mask_dir: directory containing .npy masks
        transform: albumentations transform pipeline
        in_channels: 1 for grayscale (UNet), 3 for RGB-stacked (DeepLab)
    """

    def __init__(self, image_paths, mask_dir, transform=None, in_channels=1):
        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.transform = transform
        self.in_channels = in_channels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.mask_dir, stem + ".npy")

        # Load grayscale image, resize to 1024x768
        image = Image.open(img_path).convert("L").resize((1024, 768), Image.BILINEAR)
        image = np.array(image, dtype=np.float32)

        # Load mask, resize with nearest to preserve labels
        mask = np.load(mask_path)
        mask = np.array(Image.fromarray(mask).resize((1024, 768), Image.NEAREST))

        # Stack to 3 channels for DeepLab (before augmentation)
        if self.in_channels == 3:
            image = np.stack([image, image, image], axis=-1)  # (H, W, 3)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
            # Ensure tensors after transform
            if not isinstance(image, torch.Tensor):
                image = torch.tensor(image, dtype=torch.float32)
            if image.ndim == 2:
                image = image.unsqueeze(0)
            # For 3-channel: ToTensorV2 outputs (H, W, 3) -> need (3, H, W)
            if image.ndim == 3 and image.shape[0] != self.in_channels:
                image = image.permute(2, 0, 1)
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.long)

        # When transform=None, return numpy arrays (for use with _TransformSubset)
        return image, mask


def get_train_transform(in_channels=1):
    """Training augmentation pipeline matching original notebooks."""
    if in_channels == 3:
        norm = A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    else:
        norm = A.Normalize(mean=(0.5,), std=(0.5,))

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=5, p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        A.GaussianBlur(blur_limit=(1, 3), p=0.2),
        A.GaussNoise(std_range=(0.02, 0.05), p=0.2),
        norm,
        ToTensorV2(),
    ])


def get_strong_train_transform(in_channels=1):
    """Stronger training augmentation preset for harder regularisation.

    Builds on get_train_transform but increases geometric strength and adds
    photometric / dropout augmentations:
        - Rotation increased to 15° (from 5°)
        - ElasticTransform probability increased to 0.3 (from 0.2)
        - RandomBrightnessContrast (p=0.3)
        - CLAHE (p=0.2)
        - CoarseDropout max_holes=8, max_height/width=32 (p=0.3)
        - ShiftScaleRotate shift=0.1, scale=0.1, rotate=15 (p=0.4)
    """
    if in_channels == 3:
        norm = A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    else:
        norm = A.Normalize(mean=(0.5,), std=(0.5,))

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.GaussianBlur(blur_limit=(1, 3), p=0.2),
        A.GaussNoise(std_range=(0.02, 0.05), p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.CLAHE(p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.4),
        norm,
        ToTensorV2(),
    ])


def get_val_transform(in_channels=1):
    """Validation/test transform — normalize only."""
    if in_channels == 3:
        norm = A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    else:
        norm = A.Normalize(mean=(0.5,), std=(0.5,))

    return A.Compose([
        norm,
        ToTensorV2(),
    ])
