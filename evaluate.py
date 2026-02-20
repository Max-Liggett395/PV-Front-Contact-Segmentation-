"""Evaluation entrypoint."""

import argparse
import os

import torch

from src.data import SEMDataModule
from src.models import create_model
from src.evaluation import Evaluator
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate SEM segmentation model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data-config", type=str, required=True, help="Path to data config")
    parser.add_argument("--output", type=str, default="logs/results.json", help="Output path for results")
    args = parser.parse_args()

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg = checkpoint["config"].get("model", checkpoint.get("config", {}))

    # If model config isn't in checkpoint, infer from data config
    if "architecture" not in model_cfg:
        print("Warning: model config not found in checkpoint, using default UNet ResNet34")
        model_cfg = load_config("configs/model/unet_resnet34.yaml")

    # Create model and load weights
    model = create_model(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Data
    data_cfg = load_config(args.data_config)
    dm = SEMDataModule(data_cfg)
    dm.setup()

    # Evaluate
    evaluator = Evaluator(model, device, num_classes=model_cfg.get("num_classes", 6))
    metrics = evaluator.run(dm.test_dataloader())
    evaluator.save_results(metrics, args.output)

    print(f"\nmIoU: {metrics['miou']:.4f}")
    print(f"F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"Pixel accuracy: {metrics['pixel_accuracy']:.4f}")


if __name__ == "__main__":
    main()
