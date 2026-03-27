"""Run inference on SEM images and save colored segmentation masks.

Usage:
    # Single image
    python predict.py --checkpoint logs/runs/my-run/checkpoints/best.pt \
        --experiment configs/experiment/autoresearch/ar_exp15_pretrained_cosine.yaml \
        --input image.tif --output predictions/

    # Directory of images
    python predict.py --checkpoint logs/runs/my-run/checkpoints/best.pt \
        --experiment configs/experiment/autoresearch/ar_exp15_pretrained_cosine.yaml \
        --input images/ --output predictions/

    # Save raw class indices as .npy instead of colored PNGs
    python predict.py --checkpoint logs/runs/my-run/checkpoints/best.pt \
        --experiment configs/experiment/autoresearch/ar_exp15_pretrained_cosine.yaml \
        --input images/ --output predictions/ --save-npy
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.models import create_model
from src.utils.config import load_config

CLASSES = ["background", "silver", "glass", "silicon", "void", "interfacial_void"]
COLORS = [
    (255, 0, 0),      # background - red
    (0, 255, 0),      # silver - green
    (0, 0, 255),      # glass - blue
    (255, 255, 0),    # silicon - yellow
    (255, 0, 255),    # void - magenta
    (0, 255, 255),    # interfacial_void - cyan
]

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def load_model(checkpoint_path: str, experiment_path: str, device: torch.device):
    """Load model from checkpoint + experiment config."""
    exp_cfg = load_config(experiment_path)
    model_cfg = load_config(os.path.join("configs", exp_cfg["model"] + ".yaml"))

    model = create_model(model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    in_channels = model_cfg.get("in_channels", 1)
    return model, in_channels


def preprocess_image(image_path: str, in_channels: int) -> torch.Tensor:
    """Load and preprocess an image to match training transforms."""
    img = Image.open(image_path).convert("L")  # grayscale
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # normalize same as training

    if in_channels == 1:
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    else:
        # Stack grayscale to 3 channels
        tensor = torch.from_numpy(arr).unsqueeze(0).repeat(1, 3, 1, 1)  # (1, 3, H, W)

    return tensor


def predict_image(model, tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """Run inference and return class index map (H, W)."""
    tensor = tensor.to(device)
    with torch.no_grad():
        output = model(tensor)
        if isinstance(output, dict):
            output = output["out"]
        pred = output.argmax(dim=1).squeeze().cpu().numpy()
    return pred


def save_colored_mask(pred: np.ndarray, output_path: str):
    """Save prediction as a colored PNG."""
    color_mask = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for i, color in enumerate(COLORS):
        color_mask[pred == i] = color
    Image.fromarray(color_mask).save(output_path)


def save_overlay(image_path: str, pred: np.ndarray, output_path: str, alpha: float = 0.5):
    """Save prediction overlaid on the original image."""
    img = np.array(Image.open(image_path).convert("RGB"))
    color_mask = np.zeros_like(img)
    for i, color in enumerate(COLORS):
        color_mask[pred == i] = color
    blended = (img * (1 - alpha) + color_mask * alpha).astype(np.uint8)
    Image.fromarray(blended).save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Run inference on SEM images")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--experiment", type=str, required=True, help="Path to experiment config (.yaml)")
    parser.add_argument("--input", type=str, required=True, help="Image file or directory of images")
    parser.add_argument("--output", type=str, required=True, help="Output directory for predictions")
    parser.add_argument("--save-npy", action="store_true", help="Save raw class indices as .npy files")
    parser.add_argument("--overlay", action="store_true", help="Also save overlay on original image")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto | cuda | cpu | mps")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    model, in_channels = load_model(args.checkpoint, args.experiment, device)
    print(f"Model loaded from {args.checkpoint}")

    # Collect input images
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    else:
        raise FileNotFoundError(f"Input not found: {args.input}")

    print(f"Found {len(image_paths)} image(s)")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Run inference
    for img_path in image_paths:
        tensor = preprocess_image(str(img_path), in_channels)
        pred = predict_image(model, tensor, device)

        stem = img_path.stem

        # Save colored mask
        mask_path = os.path.join(args.output, f"{stem}_pred.png")
        save_colored_mask(pred, mask_path)

        # Save raw numpy array
        if args.save_npy:
            npy_path = os.path.join(args.output, f"{stem}_pred.npy")
            np.save(npy_path, pred)

        # Save overlay
        if args.overlay:
            overlay_path = os.path.join(args.output, f"{stem}_overlay.png")
            save_overlay(str(img_path), pred, overlay_path)

        print(f"  {img_path.name} -> {stem}_pred.png")

    print(f"\nDone. {len(image_paths)} prediction(s) saved to {args.output}/")


if __name__ == "__main__":
    main()
