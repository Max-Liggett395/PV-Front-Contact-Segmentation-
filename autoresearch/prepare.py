"""
Data preparation for SEM segmentation autoresearch.
DO NOT MODIFY — this file is fixed. The agent only modifies train.py.

Verifies the merged dataset (130 images + masks) is ready and prints stats.
"""

import os
import sys
import glob
import numpy as np
from PIL import Image
from collections import Counter

# ---- Constants (importable by train.py) ----
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "merged")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
MASK_DIR = os.path.join(DATA_DIR, "masks")

NUM_CLASSES = 6
CLASS_NAMES = ["background", "silver", "glass", "silicon", "void", "interfacial_void"]

# Fixed train/val split seed
SEED = 42
TRAIN_SPLIT = 0.85

# Image dimensions after resize
IMG_H, IMG_W = 768, 1024


def get_image_mask_pairs():
    """Return sorted list of (image_path, mask_path) pairs."""
    image_paths = sorted(
        glob.glob(os.path.join(IMAGE_DIR, "*.png"))
        + glob.glob(os.path.join(IMAGE_DIR, "*.PNG"))
        + glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
        + glob.glob(os.path.join(IMAGE_DIR, "*.JPG"))
    )
    pairs = []
    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(MASK_DIR, stem + ".npy")
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
    return pairs


def main():
    print("=" * 60)
    print("SEM Segmentation Autoresearch — Data Preparation")
    print("=" * 60)

    # Check directories
    for d, name in [(IMAGE_DIR, "Image dir"), (MASK_DIR, "Mask dir")]:
        if not os.path.isdir(d):
            print(f"ERROR: {name} not found: {d}")
            sys.exit(1)
        print(f"  {name}: {d}")

    # Find pairs
    pairs = get_image_mask_pairs()
    print(f"\n  Found {len(pairs)} image/mask pairs")

    if len(pairs) == 0:
        print("ERROR: No matching image/mask pairs found!")
        sys.exit(1)

    # Check a sample image and mask
    img_path, mask_path = pairs[0]
    img = Image.open(img_path).convert("L")
    mask = np.load(mask_path)
    print(f"\n  Sample image: {os.path.basename(img_path)}")
    print(f"    Original size: {img.size}")
    print(f"    Mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"    Mask classes present: {sorted(np.unique(mask).tolist())}")

    # Class distribution across all masks
    print("\n  Computing class distribution across all masks...")
    total_pixels = Counter()
    for _, mask_path in pairs:
        m = np.load(mask_path)
        m_resized = np.array(Image.fromarray(m).resize((IMG_W, IMG_H), Image.NEAREST))
        unique, counts = np.unique(m_resized, return_counts=True)
        for u, c in zip(unique, counts):
            total_pixels[int(u)] += int(c)

    total = sum(total_pixels.values())
    print(f"\n  Class distribution ({total:,} total pixels across {len(pairs)} images):")
    for c in range(NUM_CLASSES):
        count = total_pixels.get(c, 0)
        pct = 100.0 * count / total if total > 0 else 0
        print(f"    {c} ({CLASS_NAMES[c]:>18s}): {count:>12,} pixels ({pct:5.1f}%)")

    # Train/val split sizes
    train_size = int(TRAIN_SPLIT * len(pairs))
    val_size = len(pairs) - train_size
    print(f"\n  Train/val split: {train_size} / {val_size} (seed={SEED})")

    print("\n" + "=" * 60)
    print("Data preparation complete. Ready for autoresearch.")
    print("=" * 60)


if __name__ == "__main__":
    main()
