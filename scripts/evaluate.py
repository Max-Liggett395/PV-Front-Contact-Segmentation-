#!/usr/bin/env python3
"""
Testing Script for U-Net SEM Segmentation

Tests a trained model on the test set and computes comprehensive metrics.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --config config/train_config.yaml

Optional arguments:
    --output PATH       Save metrics to JSON file
    --visualize         Generate visualizations for random samples
    --num-viz INT       Number of visualization samples (default: 10)
"""

import argparse
import sys
import json
from pathlib import Path
import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import SEMDataset, compute_dataset_statistics
from src.data.transforms import get_transforms_from_config
from src.evaluation.metrics import compute_all_metrics, compute_confusion_matrix
from src.utils.checkpointing import load_model_for_inference
from src.utils.visualization import save_comparison_grid, tensor_to_numpy
from scripts.train import create_data_splits, get_device, create_model


def predict_sliding_window(model, image, patch_size, stride, device, num_classes):
    """
    Run sliding window inference on a single image.

    Args:
        model: Trained segmentation model
        image: Input tensor of shape (C, H, W)
        patch_size: Size of square patches (e.g., 512)
        stride: Step size between patches (e.g., 256 for 50% overlap)
        device: Torch device
        num_classes: Number of segmentation classes

    Returns:
        Prediction tensor of shape (num_classes, H, W) with averaged logits
    """
    _, h, w = image.shape
    logits_sum = torch.zeros(num_classes, h, w, device=device)
    count = torch.zeros(1, h, w, device=device)

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[:, y:y + patch_size, x:x + patch_size].unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(patch)  # (1, num_classes, patch_size, patch_size)
            logits_sum[:, y:y + patch_size, x:x + patch_size] += output.squeeze(0)
            count[:, y:y + patch_size, x:x + patch_size] += 1

    # Handle right and bottom edges if image isn't evenly divisible
    # Right edge
    if (w - patch_size) % stride != 0:
        x = w - patch_size
        for y in range(0, h - patch_size + 1, stride):
            patch = image[:, y:y + patch_size, x:x + patch_size].unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(patch)
            logits_sum[:, y:y + patch_size, x:x + patch_size] += output.squeeze(0)
            count[:, y:y + patch_size, x:x + patch_size] += 1

    # Bottom edge
    if (h - patch_size) % stride != 0:
        y = h - patch_size
        for x in range(0, w - patch_size + 1, stride):
            patch = image[:, y:y + patch_size, x:x + patch_size].unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(patch)
            logits_sum[:, y:y + patch_size, x:x + patch_size] += output.squeeze(0)
            count[:, y:y + patch_size, x:x + patch_size] += 1

    # Bottom-right corner
    if (w - patch_size) % stride != 0 and (h - patch_size) % stride != 0:
        y = h - patch_size
        x = w - patch_size
        patch = image[:, y:y + patch_size, x:x + patch_size].unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(patch)
        logits_sum[:, y:y + patch_size, x:x + patch_size] += output.squeeze(0)
        count[:, y:y + patch_size, x:x + patch_size] += 1

    # Average overlapping predictions
    logits_avg = logits_sum / count.clamp(min=1)
    return logits_avg


def main():
    parser = argparse.ArgumentParser(description="Test U-Net on test set")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config/train_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save metrics JSON")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization samples")
    parser.add_argument("--num-viz", type=int, default=10,
                        help="Number of samples to visualize")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cuda, mps, cpu)")
    parser.add_argument("--patch-size", type=int, default=None,
                        help="Patch size for sliding window inference (e.g., 512)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride for sliding window inference (default: patch_size // 2)")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Device selection
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Create data splits (get test set)
    print("\nLoading test set...")
    img_dir = config["dataset"]["images_dir"]
    label_dir = config["dataset"]["labels_dir"]

    train_filenames, _, test_filenames = create_data_splits(img_dir, config)
    print(f"Test set: {len(test_filenames)} images")

    # Auto-compute dataset statistics from training set (same normalization as training)
    print("\nComputing dataset statistics from training set...")
    dataset_stats = compute_dataset_statistics(img_dir, train_filenames)
    print(f"  Mean: {dataset_stats['mean']}")
    print(f"  Std:  {dataset_stats['std']}")

    for mode in ("train", "val"):
        if mode in config.get("augmentation", {}):
            if "normalize" in config["augmentation"][mode]:
                config["augmentation"][mode]["normalize"]["mean"] = dataset_stats["mean"]
                config["augmentation"][mode]["normalize"]["std"] = dataset_stats["std"]

    # Create test dataset
    test_transform = get_transforms_from_config(config, mode="val")
    test_dataset = SEMDataset(img_dir, label_dir, test_filenames, transform=test_transform)

    # Create data loader
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 4)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # Initialize model
    print("\nLoading model...")
    model_config = config["model"]
    model = create_model(model_config)

    # Load checkpoint
    model = load_model_for_inference(args.checkpoint, model, device)
    print(f"Model loaded from: {args.checkpoint}")

    # Run inference on test set
    use_sliding_window = args.patch_size is not None
    if use_sliding_window:
        stride = args.stride if args.stride is not None else args.patch_size // 2
        print(f"\nRunning sliding window inference (patch_size={args.patch_size}, stride={stride})...")
    else:
        print("\nRunning inference on test set...")

    all_predictions = []
    all_targets = []
    all_images = []

    model.eval()
    if use_sliding_window:
        # Sliding window: process one image at a time from dataset
        for i in tqdm(range(len(test_dataset)), desc="Testing (sliding window)"):
            image, label = test_dataset[i]
            all_images.append(image.unsqueeze(0))
            all_targets.append(label.unsqueeze(0))

            logits = predict_sliding_window(
                model, image, args.patch_size, stride, device,
                num_classes=model_config["num_classes"]
            )
            all_predictions.append(logits.cpu().unsqueeze(0))
    else:
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                all_predictions.append(outputs.cpu())
                all_targets.append(labels.cpu())
                all_images.append(images.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_images = torch.cat(all_images, dim=0)

    print(f"Total samples tested: {len(all_predictions)}")

    # Compute all metrics
    print("\nComputing metrics...")
    metrics = compute_all_metrics(all_predictions, all_targets, num_classes=model_config["num_classes"])

    # Compute confusion matrix
    confusion_mat = compute_confusion_matrix(all_predictions, all_targets, num_classes=model_config["num_classes"])

    # Print results
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(f"\nOverall Metrics:")
    print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"  Macro F1:       {metrics['f1_macro']:.4f}")
    print(f"  Macro IoU:      {metrics['iou_macro']:.4f}")
    print(f"  Macro Dice:     {metrics['dice_macro']:.4f}")

    class_names = config.get("class_names", {})
    print(f"\nPer-Class F1 Scores:")
    for i in range(model_config["num_classes"]):
        class_name = class_names.get(i, f"Class {i}")
        f1_key = f"f1_class_{i}"
        if f1_key in metrics:
            print(f"  {class_name:20s}: {metrics[f1_key]:.4f}")

    print(f"\nComparison to Paper (Target F1 > 0.95):")
    for i in range(model_config["num_classes"]):
        class_name = class_names.get(i, f"Class {i}")
        f1_key = f"f1_class_{i}"
        if f1_key in metrics:
            f1_value = metrics[f1_key]
            status = "✓" if f1_value > 0.95 else "✗"
            print(f"  {status} {class_name:20s}: {f1_value:.4f}")

    print(f"\nConfusion Matrix:")
    print(confusion_mat)
    print("=" * 70)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metrics_json = {k: float(v) for k, v in metrics.items()}
        metrics_json["confusion_matrix"] = confusion_mat.tolist()

        with open(output_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)

        print(f"\nMetrics saved to: {output_path}")

    if args.visualize:
        print(f"\nGenerating {args.num_viz} visualization samples...")
        viz_dir = Path("results/visualizations")
        viz_dir.mkdir(parents=True, exist_ok=True)

        pred_classes = torch.argmax(all_predictions, dim=1)

        num_samples = min(args.num_viz, len(all_images))
        indices = np.random.choice(len(all_images), num_samples, replace=False)

        class_colors = config.get("class_colors", None)
        if class_colors:
            class_colors = {int(k): tuple(v) for k, v in class_colors.items()}

        for idx_num, idx in enumerate(indices):
            image_np = tensor_to_numpy(all_images[idx])
            gt_np = tensor_to_numpy(all_targets[idx])
            pred_np = tensor_to_numpy(pred_classes[idx])

            save_path = viz_dir / f"test_sample_{idx_num}.png"
            save_comparison_grid(
                image_np, gt_np, pred_np,
                str(save_path),
                class_colors=class_colors
            )

        print(f"Visualizations saved to: {viz_dir}")

    print("\nTesting completed successfully!")


if __name__ == "__main__":
    main()
