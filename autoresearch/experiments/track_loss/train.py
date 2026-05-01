"""
SEM Segmentation Autoresearch — train.py
=========================================
This is the ONLY file the agent modifies.

It contains the full pipeline: data loading, model, training loop, and evaluation.
Each run prints a single summary line at the end:

    RESULT | val_f1=0.XXXX | val_loss=X.XXXX | epoch=N | ...

The agent's goal is to MAXIMIZE val_f1 (macro F1 score).
"""

import os
import sys
import time
import glob
import json
import random
import hashlib
import fcntl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---- Import constants from prepare.py (DO NOT MODIFY these) ----
AUTORESEARCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, AUTORESEARCH_ROOT)
from prepare import (
    IMAGE_DIR, MASK_DIR, NUM_CLASSES, CLASS_NAMES,
    SEED, TRAIN_SPLIT, IMG_H, IMG_W, get_image_mask_pairs,
)

# =====================================================================
# CONFIGURATION — feel free to modify these
# =====================================================================
CONFIG = {
    "architecture": "unet",      # "unet", "deeplabv3", or "deeplabv3plus"
    "in_channels": 1,            # 1=grayscale, 3=RGB-stacked
    "batch_size": 2,
    "max_epochs": 50,            # short budget for fast iteration
    "lr": 5e-4,
    "weight_decay": 0.0,
    "dropout": 0.3,
    "class_weights": [1.0, 1.5, 5.0, 2.0, 4.0, 6.0],  # boost rare classes
    "mixed_precision": False,
    "patience": 25,              # early stopping patience
}


# =====================================================================
# DATASET
# =====================================================================
class SEMDataset(Dataset):
    def __init__(self, pairs, transform=None, in_channels=1):
        self.pairs = pairs
        self.transform = transform
        self.in_channels = in_channels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # Load grayscale, resize
        image = Image.open(img_path).convert("L").resize((IMG_W, IMG_H), Image.BILINEAR)
        image = np.array(image, dtype=np.float32)

        # Load mask, resize with nearest
        mask = np.load(mask_path)
        mask = np.array(Image.fromarray(mask).resize((IMG_W, IMG_H), Image.NEAREST))

        # Stack to 3 channels if needed
        if self.in_channels == 3:
            image = np.stack([image, image, image], axis=-1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if image.ndim == 3 and image.shape[0] != self.in_channels:
            image = image.permute(2, 0, 1)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask


# =====================================================================
# TRANSFORMS
# =====================================================================
def get_train_transform(in_channels=1):
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


def get_val_transform(in_channels=1):
    if in_channels == 3:
        norm = A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    else:
        norm = A.Normalize(mean=(0.5,), std=(0.5,))
    return A.Compose([
        norm,
        ToTensorV2(),
    ])


# =====================================================================
# MODEL
# =====================================================================
class UNet(nn.Module):
    """U-Net with 5 encoder blocks, 4 decoder blocks, skip connections, dropout."""

    def __init__(self, in_channels=1, num_classes=6, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        # Encoder
        self.enc1a = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.enc1b = nn.Conv2d(64, 64, 3, 1, 1)
        self.enc2a = nn.Conv2d(64, 128, 3, 1, 1)
        self.enc2b = nn.Conv2d(128, 128, 3, 1, 1)
        self.enc3a = nn.Conv2d(128, 256, 3, 1, 1)
        self.enc3b = nn.Conv2d(256, 256, 3, 1, 1)
        self.enc4a = nn.Conv2d(256, 512, 3, 1, 1)
        self.enc4b = nn.Conv2d(512, 512, 3, 1, 1)

        # Bottleneck
        self.bottleneck_a = nn.Conv2d(512, 1024, 3, 1, 1)
        self.bottleneck_b = nn.Conv2d(1024, 1024, 3, 1, 1)

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, 2, 0)
        self.dec1a = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.dec1b = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.up2 = nn.ConvTranspose2d(1024, 256, 2, 2, 0)
        self.dec2a = nn.Conv2d(512, 512, 3, 1, 1)
        self.dec2b = nn.Conv2d(512, 512, 3, 1, 1)
        self.up3 = nn.ConvTranspose2d(512, 128, 2, 2, 0)
        self.dec3a = nn.Conv2d(256, 256, 3, 1, 1)
        self.dec3b = nn.Conv2d(256, 256, 3, 1, 1)
        self.up4 = nn.ConvTranspose2d(256, 64, 2, 2, 0)
        self.dec4a = nn.Conv2d(128, 128, 3, 1, 1)
        self.dec4b = nn.Conv2d(128, 128, 3, 1, 1)

        # Output
        self.out = nn.Conv2d(128, num_classes, 1, 1, 0)

    def _downsample(self, x, conv_a, conv_b):
        f = F.relu(conv_a(x))
        f = F.relu(conv_b(f))
        p = self.max_pool(f)
        p = self.dropout(p)
        return f, p

    def _upsample(self, x, skip, up_conv, conv_a, conv_b):
        x = up_conv(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x, skip], dim=1)
        x = self.dropout(x)
        x = F.relu(conv_a(x))
        x = F.relu(conv_b(x))
        return x

    def forward(self, x):
        f1, p1 = self._downsample(x, self.enc1a, self.enc1b)
        f2, p2 = self._downsample(p1, self.enc2a, self.enc2b)
        f3, p3 = self._downsample(p2, self.enc3a, self.enc3b)
        f4, p4 = self._downsample(p3, self.enc4a, self.enc4b)
        bn = F.relu(self.bottleneck_a(p4))
        bn = F.relu(self.bottleneck_b(bn))
        u1 = self._upsample(bn, f4, self.up1, self.dec1a, self.dec1b)
        u2 = self._upsample(u1, f3, self.up2, self.dec2a, self.dec2b)
        u3 = self._upsample(u2, f2, self.up3, self.dec3a, self.dec3b)
        u4 = self._upsample(u3, f1, self.up4, self.dec4a, self.dec4b)
        return self.out(u4)


def create_model(config):
    """Create model from config."""
    arch = config["architecture"].lower()
    in_ch = config["in_channels"]
    n_cls = NUM_CLASSES

    if arch == "unet":
        return UNet(in_channels=in_ch, num_classes=n_cls, dropout=config.get("dropout", 0.3))
    elif arch == "deeplabv3":
        import torchvision.models.segmentation as seg
        model = seg.deeplabv3_resnet50(weights=None)
        model.classifier[4] = nn.Conv2d(256, n_cls, kernel_size=1)
        model.aux_classifier = None
        return model
    elif arch == "deeplabv3plus":
        import torchvision.models.segmentation as seg
        model = seg.deeplabv3_resnet101(weights=None)
        model.classifier[4] = nn.Conv2d(256, n_cls, kernel_size=1)
        model.aux_classifier = None
        return model
    else:
        raise ValueError(f"Unknown architecture: {arch}")


# =====================================================================
# METRICS
# =====================================================================
def compute_metrics(preds, targets):
    """Compute mIoU, F1-macro, pixel accuracy. Returns dict."""
    iou_per_class = []
    f1_per_class = []

    for c in range(NUM_CLASSES):
        pred_c = (preds == c)
        target_c = (targets == c)
        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        tp = intersection
        fp = (pred_c & ~target_c).sum().float()
        fn = (~pred_c & target_c).sum().float()

        iou = (intersection / (union + 1e-8)).item() if union > 0 else float("nan")
        iou_per_class.append(iou)

        precision = (tp / (tp + fp + 1e-8)).item() if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn + 1e-8)).item() if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0
        f1_per_class.append(f1)

    valid_ious = [v for v in iou_per_class if not np.isnan(v)]
    miou = float(np.mean(valid_ious)) if valid_ious else 0.0
    f1_macro = float(np.mean(f1_per_class))
    correct = (preds == targets).sum().float()
    pixel_acc = (correct / targets.numel()).item()

    return {
        "miou": miou,
        "f1_macro": f1_macro,
        "pixel_accuracy": pixel_acc,
        "per_class_iou": iou_per_class,
        "per_class_f1": f1_per_class,
    }


# =====================================================================
# TRAINING LOOP
# =====================================================================
def train():
    t0 = time.time()

    # ---- Device ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ---- Data ----
    pairs = get_image_mask_pairs()
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    in_ch = CONFIG["in_channels"]
    train_tf = get_train_transform(in_ch)
    val_tf = get_val_transform(in_ch)

    n_train = int(TRAIN_SPLIT * len(pairs))
    indices = list(range(len(pairs)))
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(pairs), generator=g).tolist()
    train_pairs = [pairs[i] for i in perm[:n_train]]
    val_pairs = [pairs[i] for i in perm[n_train:]]

    train_ds = SEMDataset(train_pairs, transform=train_tf, in_channels=in_ch)
    val_ds = SEMDataset(val_pairs, transform=val_tf, in_channels=in_ch)

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                            num_workers=2, pin_memory=True)

    print(f"Data: {len(train_pairs)} train / {len(val_pairs)} val")

    # ---- Model ----
    model = create_model(CONFIG)
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {CONFIG['architecture']} | Params: {param_count:,}")

    # ---- Loss ----
    cw = CONFIG.get("class_weights")
    weight = torch.tensor(cw, dtype=torch.float32).to(device) if cw else None
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    # ---- Optimizer ----
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"],
                                 weight_decay=CONFIG["weight_decay"])

    # ---- Mixed precision ----
    use_amp = CONFIG["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None

    # ---- Training ----
    best_f1 = 0.0
    best_epoch = 0
    best_metrics = {}
    patience_counter = 0
    max_epochs = CONFIG["max_epochs"]
    patience = CONFIG["patience"]

    for epoch in range(1, max_epochs + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast(device_type=device.type):
                    logits = model(images)
                    if isinstance(logits, dict):
                        logits = logits["out"]
                    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, NUM_CLASSES)
                    loss = loss_fn(logits_flat, masks.view(-1).long())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                if isinstance(logits, dict):
                    logits = logits["out"]
                logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, NUM_CLASSES)
                loss = loss_fn(logits_flat, masks.view(-1).long())
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
        train_loss /= len(train_loader)

        # -- Validate --
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                if isinstance(logits, dict):
                    logits = logits["out"]
                logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, NUM_CLASSES)
                loss = loss_fn(logits_flat, masks.view(-1).long())
                val_loss += loss.item()
                all_preds.append(logits.argmax(dim=1).cpu())
                all_targets.append(masks.cpu())

        val_loss /= len(val_loader)
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metrics = compute_metrics(all_preds, all_targets)

        f1 = metrics["f1_macro"]
        print(f"Epoch {epoch:3d}/{max_epochs} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | f1={f1:.4f} | miou={metrics['miou']:.4f} | "
              f"px_acc={metrics['pixel_accuracy']:.4f}")

        # -- Early stopping on F1 macro --
        if f1 > best_f1 + 1e-4:
            best_f1 = f1
            best_epoch = epoch
            best_metrics = {**metrics, "val_loss": val_loss, "train_loss": train_loss}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    elapsed = time.time() - t0

    # ---- Final result line (parsed by the autoresearch runner) ----
    per_cls_iou = " | ".join(
        f"{CLASS_NAMES[i]}_iou={best_metrics['per_class_iou'][i]:.4f}"
        for i in range(NUM_CLASSES)
    )
    per_cls_f1 = " | ".join(
        f"{CLASS_NAMES[i]}_f1={best_metrics['per_class_f1'][i]:.4f}"
        for i in range(NUM_CLASSES)
    )
    print(f"\nRESULT | val_f1={best_f1:.4f} | val_miou={best_metrics.get('miou', 0):.4f} | "
          f"val_loss={best_metrics.get('val_loss', 0):.4f} | "
          f"px_acc={best_metrics.get('pixel_accuracy', 0):.4f} | "
          f"epoch={best_epoch}/{max_epochs} | time={elapsed:.0f}s")
    print(f"PER_CLASS_F1 | {per_cls_f1}")
    print(f"PER_CLASS_IOU | {per_cls_iou}")

    # ---- Save result to JSON log ----
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Hash of train.py for tracking which version produced this result
    with open(__file__, "r") as f:
        code_hash = hashlib.md5(f.read().encode()).hexdigest()[:8]

    result = {
        "val_f1": best_f1,
        "val_miou": best_metrics.get("miou", 0),
        "val_loss": best_metrics.get("val_loss", 0),
        "f1_macro": best_metrics.get("f1_macro", 0),
        "pixel_accuracy": best_metrics.get("pixel_accuracy", 0),
        "per_class_f1": best_metrics.get("per_class_f1", []),
        "per_class_iou": best_metrics.get("per_class_iou", []),
        "best_epoch": best_epoch,
        "max_epochs": max_epochs,
        "elapsed_seconds": elapsed,
        "config": CONFIG,
        "code_hash": code_hash,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Append to log
    log_file = os.path.join(log_dir, "results.jsonl")
    with open(log_file, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"Result saved to {log_file}")

    # ---- Also write to shared central log (file-locked) ----
    result["track"] = "track_loss"
    shared_log = os.path.join(AUTORESEARCH_ROOT, "logs", "results.jsonl")
    os.makedirs(os.path.dirname(shared_log), exist_ok=True)
    with open(shared_log, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(result) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)
    print(f"Result also saved to {shared_log}")


if __name__ == "__main__":
    train()
