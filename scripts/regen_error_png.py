"""Regenerate every predictions/<stem>/error.png with a class-color legend.

No model/GPU needed: reads the existing prediction.png (color -> class index),
loads the GT .npy mask, and rewrites error.png using the updated
save_error_on_input (includes legend + % wrong in the title).

Usage:
    python scripts/regen_error_png.py \
        --data configs/data/merged.yaml \
        --predictions predictions
"""

import argparse
import os
import sys
import types

import numpy as np

if not hasattr(np, "_core"):
    core_mod = types.ModuleType("numpy._core")
    for _name in dir(np.core):
        setattr(core_mod, _name, getattr(np.core, _name))
    sys.modules["numpy._core"] = core_mod
    import numpy.core.multiarray, numpy.core.numeric, numpy.core._multiarray_umath
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
    sys.modules["numpy._core.numeric"] = np.core.numeric
    sys.modules["numpy._core._multiarray_umath"] = np.core._multiarray_umath

import matplotlib
matplotlib.use("Agg")
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import CLASS_NAMES, CLASS_COLORS, NUM_CLASSES
from src.utils.config import load_config
# Reuse the updated legend-aware renderer from the main builder
from scripts.build_prediction_dirs import save_error_on_input


def color_to_index(rgb):
    """Convert an (H,W,3) uint8 RGB colored mask back to (H,W) class indices."""
    out = np.full(rgb.shape[:2], -1, dtype=np.int64)
    for i, c in enumerate(CLASS_COLORS):
        m = (rgb[..., 0] == c[0]) & (rgb[..., 1] == c[1]) & (rgb[..., 2] == c[2])
        out[m] = i
    if (out == -1).any():
        # Stray pixels — shouldn't happen if prediction.png came from our colorize()
        out[out == -1] = 0
    return out


def load_gt_mask(mask_path, target_shape):
    """Load a class-index GT mask .npy and resize to target (H, W) with nearest-neighbor."""
    m = np.load(mask_path)
    if m.shape != target_shape:
        m = np.array(Image.fromarray(m).resize((target_shape[1], target_shape[0]), Image.NEAREST))
    return m.astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Data config (for mask_dir)")
    ap.add_argument("--predictions", default="predictions")
    args = ap.parse_args()

    data_cfg = load_config(args.data)
    mask_dir = data_cfg["mask_dir"]

    stems = sorted(
        d for d in os.listdir(args.predictions)
        if os.path.isdir(os.path.join(args.predictions, d))
        and os.path.isfile(os.path.join(args.predictions, d, "prediction.png"))
    )
    print(f"Regenerating error.png for {len(stems)} per-image dirs")

    for i, stem in enumerate(stems):
        dir_path = os.path.join(args.predictions, stem)
        img_path = os.path.join(dir_path, "image.png")
        pred_path = os.path.join(dir_path, "prediction.png")
        mask_path = os.path.join(mask_dir, stem + ".npy")
        err_path = os.path.join(dir_path, "error.png")

        if not os.path.exists(mask_path):
            print(f"  [{i+1:>3}/{len(stems)}] {stem}  SKIP (no GT mask at {mask_path})")
            continue

        img_u8 = np.array(Image.open(img_path).convert("L"))
        pred_rgb = np.array(Image.open(pred_path).convert("RGB"))
        pred = color_to_index(pred_rgb)
        gt = load_gt_mask(mask_path, pred.shape)

        save_error_on_input(img_u8, gt, pred, err_path)
        print(f"  [{i+1:>3}/{len(stems)}] {stem}")

    print(f"\nDone. Rewrote {len(stems)} error.png files.")


if __name__ == "__main__":
    main()
