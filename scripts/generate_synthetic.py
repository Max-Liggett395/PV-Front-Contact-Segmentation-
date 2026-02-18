#!/usr/bin/env python3
"""
Generate synthetic training data via morphological operations on segmentation masks.

For each original mask, generates N variants by randomly eroding or dilating
per-class binary masks, then reconstructing the multi-class mask using the
DRAW_PRIORITY ordering from generate_masks.py.

Images are symlinked (or copied on Windows) to the originals — only masks change.

Usage:
    python scripts/generate_synthetic.py \
        --masks-dir data/masks/npy \
        --images-dir data/images \
        --output-dir data/synthetic \
        --num-variants 3 \
        --seed 42
"""

import argparse
import json
import os
import platform
import shutil
from pathlib import Path

import cv2
import numpy as np

# Draw priority from generate_masks.py: higher priority overwrites lower
# Order: Silver(1) -> Silicon(3) -> Glass(2) -> Void(4) -> Interfacial Void(5)
DRAW_PRIORITY = [1, 3, 2, 4, 5]

# Per-class morphological parameters
# (kernel_min, kernel_max, max_iters, p_operate, dilate_bias)
CLASS_PARAMS = {
    1: {"kernel_range": (3, 7), "max_iters": 2, "p_operate": 0.6, "dilate_bias": 0.5},  # Silver
    2: {"kernel_range": (3, 5), "max_iters": 1, "p_operate": 0.5, "dilate_bias": 0.5},  # Glass
    3: {"kernel_range": (3, 7), "max_iters": 2, "p_operate": 0.6, "dilate_bias": 0.5},  # Silicon
    4: {"kernel_range": (3, 7), "max_iters": 2, "p_operate": 0.7, "dilate_bias": 0.6},  # Void
    5: {"kernel_range": (3, 5), "max_iters": 1, "p_operate": 0.5, "dilate_bias": 0.5},  # Interfacial Void
}

CLASS_NAMES = {
    0: "Background",
    1: "Silver",
    2: "Glass",
    3: "Silicon",
    4: "Void",
    5: "Interfacial Void",
}


def morph_class_mask(binary_mask: np.ndarray, params: dict, rng: np.random.Generator) -> np.ndarray:
    """Apply random morphological operation to a single-class binary mask."""
    if rng.random() > params["p_operate"]:
        return binary_mask

    k_min, k_max = params["kernel_range"]
    k_size = int(rng.integers(k_min, k_max + 1))
    # Ensure odd kernel
    if k_size % 2 == 0:
        k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    n_iters = int(rng.integers(1, params["max_iters"] + 1))

    if rng.random() < params["dilate_bias"]:
        return cv2.dilate(binary_mask, kernel, iterations=n_iters)
    else:
        return cv2.erode(binary_mask, kernel, iterations=n_iters)


def generate_variant(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a single synthetic mask variant.

    For each non-background class, extract binary mask, apply random
    morphological operation, then reconstruct using draw priority.
    """
    h, w = mask.shape
    result = np.zeros((h, w), dtype=np.uint8)

    # Process classes in draw priority order (lower priority first)
    for class_id in DRAW_PRIORITY:
        if class_id not in CLASS_PARAMS:
            continue

        binary = (mask == class_id).astype(np.uint8)
        if binary.sum() == 0:
            continue

        modified = morph_class_mask(binary, CLASS_PARAMS[class_id], rng)
        result[modified > 0] = class_id

    return result


def find_image_for_stem(images_dir: Path, stem: str) -> Path | None:
    """Find the image file matching a mask stem."""
    for ext in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]:
        candidate = images_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def compute_class_distribution(masks: list[np.ndarray], num_classes: int = 6) -> dict:
    """Compute per-class pixel percentages across a list of masks."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for m in masks:
        for c in range(num_classes):
            counts[c] += (m == c).sum()
    total = counts.sum()
    if total == 0:
        return {c: 0.0 for c in range(num_classes)}
    return {c: float(counts[c]) / total * 100 for c in range(num_classes)}


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data via morphological mask augmentation"
    )
    parser.add_argument("--masks-dir", type=str, default="data/masks/npy",
                        help="Directory containing original .npy masks")
    parser.add_argument("--images-dir", type=str, default="data/images",
                        help="Directory containing original images")
    parser.add_argument("--output-dir", type=str, default="data/synthetic",
                        help="Output directory for synthetic data")
    parser.add_argument("--num-variants", type=int, default=3,
                        help="Number of synthetic variants per original mask")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--mag-map", type=str, default="data/magnification_map.json",
                        help="Path to magnification map JSON (for synthetic map generation)")
    args = parser.parse_args()

    masks_dir = Path(args.masks_dir)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    if not masks_dir.exists():
        print(f"Error: Masks directory not found: {masks_dir}")
        return
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return

    # Create output directories
    out_images_dir = output_dir / "images"
    out_masks_dir = output_dir / "masks" / "npy"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    # Load magnification map if available
    mag_map_path = Path(args.mag_map)
    orig_mag_map = {}
    if mag_map_path.exists():
        with open(mag_map_path, "r") as f:
            orig_mag_map = json.load(f)
        print(f"Loaded magnification map: {len(orig_mag_map)} entries")

    rng = np.random.default_rng(args.seed)

    # Collect all mask files
    mask_files = sorted(masks_dir.glob("*.npy"))
    print(f"Found {len(mask_files)} original masks")
    print(f"Generating {args.num_variants} variants each = {len(mask_files) * args.num_variants} synthetic samples")

    original_masks = []
    synthetic_masks = []
    synth_mag_map = {}
    generated = 0
    warnings = []
    use_symlinks = platform.system() != "Windows"

    for mask_path in mask_files:
        stem = mask_path.stem
        mask = np.load(mask_path)
        original_masks.append(mask)

        # Find corresponding image
        img_path = find_image_for_stem(images_dir, stem)
        if img_path is None:
            print(f"  Warning: No image found for {stem}, skipping")
            continue

        # Get magnification for this stem
        parent_mag = orig_mag_map.get(stem)

        for i in range(args.num_variants):
            synth_stem = f"{stem}_synth{i:03d}"

            # Generate morphologically modified mask
            variant = generate_variant(mask, rng)

            # Validate: check no class was completely eroded away
            orig_classes = set(np.unique(mask))
            synth_classes = set(np.unique(variant))
            lost = orig_classes - synth_classes
            if lost:
                lost_names = [CLASS_NAMES.get(c, f"Class {c}") for c in lost]
                warnings.append(f"{synth_stem}: lost classes {lost_names}")

            # Save modified mask
            np.save(out_masks_dir / f"{synth_stem}.npy", variant)
            synthetic_masks.append(variant)

            # Symlink (or copy) original image
            out_img_path = out_images_dir / f"{synth_stem}{img_path.suffix}"
            if not out_img_path.exists():
                if use_symlinks:
                    os.symlink(img_path.resolve(), out_img_path)
                else:
                    shutil.copy2(img_path, out_img_path)

            # Add to synthetic magnification map
            if parent_mag is not None:
                synth_mag_map[synth_stem] = parent_mag

            generated += 1

        if (generated // args.num_variants) % 20 == 0 and generated > 0:
            print(f"  Processed {generated // args.num_variants}/{len(mask_files)} masks...")

    # Save synthetic magnification map
    if synth_mag_map:
        mag_out_path = output_dir / "magnification_map.json"
        with open(mag_out_path, "w") as f:
            json.dump(synth_mag_map, f, indent=2)
        print(f"\nSaved synthetic magnification map: {len(synth_mag_map)} entries -> {mag_out_path}")

    print(f"\nGeneration complete: {generated} synthetic samples")
    print(f"  Masks: {out_masks_dir}")
    print(f"  Images: {out_images_dir} ({'symlinks' if use_symlinks else 'copies'})")

    # Print per-class distribution comparison
    print("\nPer-class pixel distribution (%):")
    orig_dist = compute_class_distribution(original_masks)
    synth_dist = compute_class_distribution(synthetic_masks)
    print(f"  {'Class':<20s} {'Original':>10s} {'Synthetic':>10s} {'Delta':>10s}")
    print(f"  {'-'*50}")
    for c in range(6):
        name = CLASS_NAMES.get(c, f"Class {c}")
        o = orig_dist[c]
        s = synth_dist[c]
        delta = s - o
        print(f"  {name:<20s} {o:>9.2f}% {s:>9.2f}% {delta:>+9.2f}%")

    # Print warnings
    if warnings:
        print(f"\nWarnings ({len(warnings)} masks had classes completely eroded):")
        for w in warnings:
            print(f"  {w}")
    else:
        print("\nNo classes were completely eroded in any synthetic mask.")


if __name__ == "__main__":
    main()
