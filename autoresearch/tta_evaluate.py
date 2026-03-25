"""Test-Time Augmentation (TTA) evaluator.

Runs inference with multiple augmentations and averages predictions
for a free mIoU boost without retraining.

Usage:
    python -m autoresearch.tta_evaluate \
        --checkpoint logs/runs/ar-exp15/checkpoints/best.pt \
        --data-config configs/data/merged.yaml \
        --model-config configs/model/deeplabv3plus_resnet101_pretrained.yaml
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import SEMDataModule
from src.data.dataset import get_val_transform
from src.evaluation.metrics import compute_metrics
from src.models import create_model
from src.utils.config import load_config


def predict_with_tta(model, images, device, num_classes):
    """Run TTA: original + hflip + vflip + hflip+vflip, average logits."""
    model.eval()
    all_logits = []

    with torch.no_grad():
        # Original
        logits = _forward(model, images.to(device))
        all_logits.append(logits)

        # Horizontal flip
        flipped_h = torch.flip(images, dims=[3]).to(device)
        logits_h = _forward(model, flipped_h)
        logits_h = torch.flip(logits_h, dims=[3])
        all_logits.append(logits_h)

        # Vertical flip
        flipped_v = torch.flip(images, dims=[2]).to(device)
        logits_v = _forward(model, flipped_v)
        logits_v = torch.flip(logits_v, dims=[2])
        all_logits.append(logits_v)

        # Both flips
        flipped_hv = torch.flip(images, dims=[2, 3]).to(device)
        logits_hv = _forward(model, flipped_hv)
        logits_hv = torch.flip(logits_hv, dims=[2, 3])
        all_logits.append(logits_hv)

    # Average logits
    avg_logits = torch.stack(all_logits).mean(dim=0)
    return avg_logits


def _forward(model, images):
    output = model(images)
    if isinstance(output, dict):
        output = output["out"]
    return output


def main():
    parser = argparse.ArgumentParser(description="TTA Evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--model-config", required=True)
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Model
    model_cfg = load_config(args.model_config)
    model = create_model(model_cfg)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Get checkpoint metrics for comparison
    ckpt_metrics = ckpt.get("metrics", {})
    print(f"Checkpoint metrics: miou={ckpt_metrics.get('miou', '?'):.4f}")

    # Data
    data_cfg = load_config(args.data_config)
    data_cfg["in_channels"] = model_cfg.get("in_channels", 1)
    dm = SEMDataModule(data_cfg)
    dm.setup()
    val_loader = dm.val_dataloader()

    num_classes = model_cfg.get("num_classes", 6)

    # Evaluate WITHOUT TTA (baseline)
    print("\nEvaluating WITHOUT TTA...")
    all_preds_no_tta = []
    all_targets = []
    for images, masks in tqdm(val_loader, desc="No TTA"):
        with torch.no_grad():
            logits = _forward(model, images.to(device))
        preds = logits.argmax(dim=1).cpu()
        all_preds_no_tta.append(preds)
        all_targets.append(masks)

    all_preds_no_tta = torch.cat(all_preds_no_tta)
    all_targets_cat = torch.cat(all_targets)
    metrics_no_tta = compute_metrics(all_preds_no_tta, all_targets_cat, num_classes)

    # Evaluate WITH TTA
    print("\nEvaluating WITH TTA (4 augmentations)...")
    all_preds_tta = []
    all_targets2 = []
    for images, masks in tqdm(val_loader, desc="TTA"):
        avg_logits = predict_with_tta(model, images, device, num_classes)
        preds = avg_logits.argmax(dim=1).cpu()
        all_preds_tta.append(preds)
        all_targets2.append(masks)

    all_preds_tta = torch.cat(all_preds_tta)
    all_targets2_cat = torch.cat(all_targets2)
    metrics_tta = compute_metrics(all_preds_tta, all_targets2_cat, num_classes)

    # Print comparison
    print(f"\n{'='*60}")
    print(f"TTA RESULTS")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'No TTA':<12} {'With TTA':<12} {'Delta':<12}")
    print(f"{'-'*60}")
    for key in ["miou", "f1_macro", "pixel_accuracy"]:
        v1 = metrics_no_tta.get(key, 0)
        v2 = metrics_tta.get(key, 0)
        delta = v2 - v1
        d_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"{key:<20} {v1:<12.4f} {v2:<12.4f} {d_str}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
