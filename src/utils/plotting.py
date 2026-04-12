"""Automatic plotting utilities for training runs."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.dataset import CLASS_NAMES, CLASS_COLORS
from src.utils.visualization import mask_to_rgb


def plot_training_curves(history, save_dir):
    """Plot training and validation metric curves.

    Args:
        history: dict with keys 'train_loss', 'val_loss', 'miou',
                 'f1_macro', 'pixel_accuracy', 'lr' — each a list per epoch.
        save_dir: directory to save the plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # --- Loss curves ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.plot(epochs, history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)

    # --- Validation metrics ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["miou"])
    axes[0].set_title("Validation mIoU")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["f1_macro"])
    axes[1].set_title("Validation F1 (macro)")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["pixel_accuracy"])
    axes[2].set_title("Validation Pixel Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "val_metrics.png"), dpi=150)
    plt.close(fig)

    # --- Learning rate ---
    if "lr" in history and history["lr"]:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(epochs, history["lr"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "lr_schedule.png"), dpi=150)
        plt.close(fig)

    print(f"Training curves saved to {save_dir}")


def plot_sample_predictions(model, dataloader, device, save_dir,
                            split_name="val", num_samples=4, num_classes=6):
    """Generate side-by-side plots of input / ground truth / prediction.

    Args:
        model: trained model (already on device).
        dataloader: DataLoader for the split to visualize.
        device: torch device.
        save_dir: directory to save the plots.
        split_name: name for the file (e.g. 'train', 'val').
        num_samples: number of samples to plot.
        num_classes: number of segmentation classes.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    images_list, masks_list, preds_list = [], [], []

    with torch.no_grad():
        for images, masks in dataloader:
            images_dev = images.to(device)
            logits = model(images_dev)
            if isinstance(logits, dict):
                logits = logits["out"]
            preds = logits.argmax(dim=1).cpu()

            for i in range(images.size(0)):
                images_list.append(images[i])
                masks_list.append(masks[i])
                preds_list.append(preds[i])
                if len(images_list) >= num_samples:
                    break
            if len(images_list) >= num_samples:
                break

    n = len(images_list)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        img = images_list[i]
        # Undo normalization for display: img was normalized with mean=0.5, std=0.5
        img = img * 0.5 + 0.5
        img = img.clamp(0, 1)

        # Convert to displayable numpy (H, W) or (H, W, 3)
        if img.shape[0] == 1:
            img_np = img.squeeze(0).numpy()
            axes[i, 0].imshow(img_np, cmap="gray")
        elif img.shape[0] == 3:
            img_np = img.permute(1, 2, 0).numpy()
            # All 3 channels are the same grayscale — just show one
            axes[i, 0].imshow(img_np[:, :, 0], cmap="gray")
        else:
            img_np = img.squeeze(0).numpy()
            axes[i, 0].imshow(img_np, cmap="gray")

        gt_rgb = mask_to_rgb(masks_list[i].numpy())
        pred_rgb = mask_to_rgb(preds_list[i].numpy())

        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(gt_rgb)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(pred_rgb)
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    # Add class legend
    legend_patches = []
    from matplotlib.patches import Patch
    for cls_id in range(min(num_classes, len(CLASS_NAMES))):
        color = np.array(CLASS_COLORS[cls_id]) / 255.0
        legend_patches.append(Patch(facecolor=color, label=CLASS_NAMES[cls_id]))
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=num_classes, fontsize=9, frameon=True)

    fig.suptitle(f"Sample Predictions — {split_name}", fontsize=14)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(os.path.join(save_dir, f"predictions_{split_name}.png"), dpi=150)
    plt.close(fig)

    print(f"{split_name} prediction samples saved to {save_dir}")
