"""Evaluate all best checkpoints with both global and per-image metrics."""

import os
import sys

import torch

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import SEMDataModule
from src.models import create_model
from src.evaluation.metrics import compute_metrics
from src.utils.config import load_config


# Map each run name to its experiment config
RUNS = {
    "unet-d83": "configs/experiment/unet_dataset83.yaml",
    "unet-d116": "configs/experiment/unet_dataset116.yaml",
    "unet-merged": "configs/experiment/baseline_merged.yaml",
    "deeplabv3-d83": "configs/experiment/deeplabv3_dataset83.yaml",
    "deeplabv3-d116": "configs/experiment/deeplabv3_dataset116.yaml",
    "deeplabv3-merged": "configs/experiment/deeplabv3_merged.yaml",
    "deeplabv3plus-d83": "configs/experiment/deeplabv3plus_dataset83.yaml",
    "deeplabv3plus-d116": "configs/experiment/deeplabv3plus_dataset116.yaml",
    "deeplabv3plus-merged": "configs/experiment/deeplabv3plus_merged.yaml",
}


def evaluate_run(run_name, exp_config_path, device):
    """Load best checkpoint for a run and compute metrics on val set."""
    checkpoint_path = f"logs/runs/{run_name}/checkpoints/best.pt"
    if not os.path.isfile(checkpoint_path):
        print(f"  SKIP {run_name}: no checkpoint at {checkpoint_path}")
        return None

    # Load configs
    exp_cfg = load_config(exp_config_path)
    data_cfg = load_config(os.path.join("configs", exp_cfg["data"] + ".yaml"))
    model_cfg = load_config(os.path.join("configs", exp_cfg["model"] + ".yaml"))

    data_cfg["in_channels"] = model_cfg.get("in_channels", 1)
    data_overrides = exp_cfg.get("data_overrides", {})
    data_cfg.update(data_overrides)

    # Setup data (same seed/split as training)
    dm = SEMDataModule(data_cfg)
    dm.setup()
    val_loader = dm.val_dataloader()

    # Create model and load weights
    model = create_model(model_cfg)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    best_epoch = ckpt.get("epoch", "?")
    num_classes = model_cfg.get("num_classes", 6)

    # Run inference
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            logits = model(images)
            if isinstance(logits, dict):
                logits = logits["out"]
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_targets.append(masks)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metrics = compute_metrics(all_preds, all_targets, num_classes)
    metrics["best_epoch"] = best_epoch
    return metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    results = {}
    for run_name, exp_config in RUNS.items():
        print(f"Evaluating {run_name}...")
        metrics = evaluate_run(run_name, exp_config, device)
        if metrics is not None:
            results[run_name] = metrics

    # Print results table
    print("\n" + "=" * 120)
    print(f"{'Run':<25} {'Epoch':>5}  "
          f"{'Global mIoU':>11} {'Global F1':>10} {'Global Acc':>10}  "
          f"{'Img mIoU':>10} {'Img F1':>10} {'Img Acc':>10}")
    print("-" * 120)

    for run_name in RUNS:
        if run_name not in results:
            continue
        m = results[run_name]
        print(f"{run_name:<25} {m['best_epoch']:>5}  "
              f"{m['miou']:>11.4f} {m['f1_macro']:>10.4f} {m['pixel_accuracy']:>10.4f}  "
              f"{m['img_miou']:>10.4f} {m['img_f1_macro']:>10.4f} {m['img_pixel_accuracy']:>10.4f}")

    print("=" * 120)


if __name__ == "__main__":
    main()
