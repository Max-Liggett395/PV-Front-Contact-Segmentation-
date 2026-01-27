#!/usr/bin/env python3
"""
Inference Script for U-Net SEM Segmentation

Run inference on a single image or directory of images.

Usage:
    python scripts/inference.py --checkpoint checkpoints/best_model.pth \\
                                --input path/to/image.png \\
                                --output results/prediction.png

    python scripts/inference.py --checkpoint checkpoints/best_model.pth \\
                                --input-dir data/images/ \\
                                --output-dir results/predictions/
"""

import argparse
import sys
from pathlib import Path
import yaml

import numpy as np
import torch
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.unet import UNet
from src.data.transforms import MINIMAL_AUGMENTATION
from src.utils.checkpointing import load_model_for_inference
from src.utils.visualization import save_single_prediction, mask_to_rgb
from scripts.train import get_device


def load_and_preprocess_image(image_path: str, transform=None) -> torch.Tensor:
    """
    Load and preprocess a single image for inference.

    Args:
        image_path (str): Path to image file
        transform: Albumentations transform

    Returns:
        torch.Tensor: Preprocessed image tensor [1, C, H, W]
    """
    image = Image.open(image_path).convert("L")
    image = image.resize((1024, 768), Image.BILINEAR)
    image = np.array(image)

    if transform:
        augmented = transform(image=image)
        image = augmented["image"]

    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)

    if image.ndim == 2:
        image = image.unsqueeze(0)

    image = image.unsqueeze(0)

    return image


def predict_single_image(model, image_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    Run inference on a single image.

    Args:
        model: Trained U-Net model
        image_tensor (torch.Tensor): Preprocessed image [1, C, H, W]
        device: Device to run inference on

    Returns:
        np.ndarray: Predicted mask [H, W] with class indices
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)

        prediction = torch.argmax(output, dim=1)
        prediction = prediction.squeeze(0)
        prediction = prediction.cpu().numpy()

    return prediction


def main():
    parser = argparse.ArgumentParser(description="Run inference on SEM images")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config/train_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to single input image")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Path to directory of input images")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save output (for single image)")
    parser.add_argument("--output-dir", type=str, default="results/inference",
                        help="Directory to save outputs (for directory input)")
    parser.add_argument("--overlay", action="store_true",
                        help="Save as overlay on original image (default: mask only)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Transparency for overlay (0-1)")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cuda, mps, cpu)")
    args = parser.parse_args()

    if args.input is None and args.input_dir is None:
        parser.error("Either --input or --input-dir must be specified")

    if args.input and args.input_dir:
        parser.error("Only one of --input or --input-dir can be specified")

    if args.input and not args.output:
        parser.error("--output must be specified when using --input")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = get_device(args.device)
    print(f"Using device: {device}")

    print("\nLoading model...")
    model_config = config["model"]
    model = UNet(
        in_channels=model_config["in_channels"],
        num_classes=model_config["num_classes"],
        dropout=model_config["dropout"]
    )

    model = load_model_for_inference(args.checkpoint, model, device)
    print(f"Model loaded from: {args.checkpoint}")

    transform = MINIMAL_AUGMENTATION

    class_colors = config.get("class_colors", None)
    if class_colors:
        class_colors = {int(k): tuple(v) for k, v in class_colors.items()}

    if args.input:
        print(f"\nRunning inference on: {args.input}")

        image_tensor = load_and_preprocess_image(args.input, transform)

        original_image = Image.open(args.input).convert("L")
        original_image = original_image.resize((1024, 768), Image.BILINEAR)
        original_image = np.array(original_image)

        prediction = predict_single_image(model, image_tensor, device)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_single_prediction(
            original_image,
            prediction,
            str(output_path),
            class_colors=class_colors,
            overlay=args.overlay,
            alpha=args.alpha
        )

        print(f"Prediction saved to: {output_path}")

    elif args.input_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            image_files.extend(input_dir.glob(ext))

        image_files = sorted(image_files)
        print(f"\nFound {len(image_files)} images in {input_dir}")

        for img_path in image_files:
            print(f"Processing: {img_path.name}")

            image_tensor = load_and_preprocess_image(str(img_path), transform)

            original_image = Image.open(img_path).convert("L")
            original_image = original_image.resize((1024, 768), Image.BILINEAR)
            original_image = np.array(original_image)

            prediction = predict_single_image(model, image_tensor, device)

            output_path = output_dir / f"{img_path.stem}_prediction.png"
            save_single_prediction(
                original_image,
                prediction,
                str(output_path),
                class_colors=class_colors,
                overlay=args.overlay,
                alpha=args.alpha
            )

        print(f"\nAll predictions saved to: {output_dir}")

    print("\nInference completed successfully!")


if __name__ == "__main__":
    main()
