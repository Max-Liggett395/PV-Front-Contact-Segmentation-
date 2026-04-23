"""Generate per-image error maps for a trained segmentation model.

For each validation image: produces a 4-panel PNG (input | GT | pred | errors)
plus a per-class error breakdown panel. Also emits:
    - summary.png        : global confusion matrix + per-class IoU bars
    - ranked.csv         : val images sorted by miou ascending (worst first)

Default target: deeplabv3plus-d83 (best checkpoint with saved weights).

Usage:
    python scripts/error_maps.py \
        --checkpoint logs/runs/deeplabv3plus-d83/checkpoints/best.pt \
        --experiment configs/experiment/deeplabv3plus_dataset83.yaml \
        --output predictions/error_maps_dlv3p_d83
"""

import argparse
import csv
import os
import sys
import types

# Shim for checkpoints saved with numpy >= 2.0 when loaded under numpy < 2.0
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
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import SEMDataModule
from src.data.dataset import CLASS_NAMES, CLASS_COLORS, NUM_CLASSES
from src.evaluation.metrics import _compute_single
from src.models import create_model
from src.utils.config import load_config


def colorize(mask):
    """Turn an (H, W) class-index array into an (H, W, 3) RGB image."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, color in enumerate(CLASS_COLORS):
        out[mask == i] = color
    return out


def error_overlay(gt, pred):
    """RGB image where pixels are colored by the (wrong) predicted class.
    Correct pixels are dark gray."""
    out = np.full((*gt.shape, 3), 32, dtype=np.uint8)  # dark gray
    wrong = gt != pred
    for i, color in enumerate(CLASS_COLORS):
        m = wrong & (pred == i)
        out[m] = color
    return out


def denormalize_display(image_tensor):
    """Take the model-input tensor (normalized) and return an (H, W) uint8 image for display."""
    arr = image_tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]  # (C, H, W) -> (H, W) using first channel (grayscale input)
    arr = arr * 0.5 + 0.5  # undo Normalize(mean=0.5, std=0.5)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def make_legend_handles():
    """matplotlib patch handles for the class legend."""
    from matplotlib.patches import Patch
    return [
        Patch(facecolor=np.array(c) / 255.0, edgecolor="black", label=n)
        for n, c in zip(CLASS_NAMES, CLASS_COLORS)
    ]


def plot_per_image(image, gt, pred, miou, per_class_iou, stem, out_path):
    """Save the 4-panel + per-class bar figure for a single image."""
    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 0.8], height_ratios=[1, 0.4])

    ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(image, cmap="gray"); ax0.set_title("input"); ax0.axis("off")
    ax1 = fig.add_subplot(gs[0, 1]); ax1.imshow(colorize(gt)); ax1.set_title("ground truth"); ax1.axis("off")
    ax2 = fig.add_subplot(gs[0, 2]); ax2.imshow(colorize(pred)); ax2.set_title(f"prediction (miou={miou:.3f})"); ax2.axis("off")
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.imshow(image, cmap="gray")
    err = error_overlay(gt, pred)
    wrong_mask = (gt != pred)
    overlay = np.zeros((*gt.shape, 4), dtype=np.float32)
    overlay[..., :3] = err / 255.0
    overlay[..., 3] = wrong_mask.astype(np.float32) * 0.7
    ax3.imshow(overlay)
    wrong_frac = wrong_mask.mean() * 100
    ax3.set_title(f"errors ({wrong_frac:.1f}% wrong, colored by predicted class)")
    ax3.axis("off")

    ax_leg = fig.add_subplot(gs[0, 4])
    ax_leg.legend(handles=make_legend_handles(), loc="center", frameon=False, fontsize=11)
    ax_leg.axis("off")

    # Per-class IoU bar + per-class error frequency
    ax_b = fig.add_subplot(gs[1, :])
    x = np.arange(NUM_CLASSES)
    iou_vals = [0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in per_class_iou]
    # Fraction of GT pixels of class c that were misclassified
    miss_frac = []
    for c in range(NUM_CLASSES):
        gtc = (gt == c)
        if gtc.sum() == 0:
            miss_frac.append(0.0)
        else:
            miss_frac.append(((gtc) & (pred != c)).sum() / gtc.sum())
    w = 0.38
    bars1 = ax_b.bar(x - w/2, iou_vals, w, label="IoU", color=[np.array(c)/255.0 for c in CLASS_COLORS])
    bars2 = ax_b.bar(x + w/2, miss_frac, w, label="missed-GT fraction", color="black", alpha=0.5)
    ax_b.set_xticks(x); ax_b.set_xticklabels(CLASS_NAMES, rotation=15)
    ax_b.set_ylim(0, 1.05); ax_b.set_ylabel("value")
    ax_b.legend(loc="upper right")
    ax_b.set_title(f"{stem}  —  per-class")

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_summary(confusion, per_class_iou, per_class_f1, out_path):
    """Global confusion matrix (row-normalized) + per-class IoU/F1 bars."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Row-normalize: each row = GT class, sums to 1
    row_sums = confusion.sum(axis=1, keepdims=True).clip(min=1)
    norm = confusion / row_sums
    ax = axes[0]
    im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("ground truth")
    ax.set_title("row-normalized confusion (recall per class)")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{norm[i,j]:.2f}", ha="center", va="center",
                    color="white" if norm[i,j] < 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    x = np.arange(NUM_CLASSES); w = 0.38
    ax.bar(x - w/2, per_class_iou, w, label="IoU", color=[np.array(c)/255.0 for c in CLASS_COLORS])
    ax.bar(x + w/2, per_class_f1, w, label="F1", color="black", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=15)
    ax.set_ylim(0, 1.05); ax.set_ylabel("value"); ax.legend()
    ax.set_title("per-class IoU / F1 (global)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def get_val_stems(dm):
    """Pull original filenames for the val split so we can label outputs."""
    subset = dm.val_dataset.subset  # torch.utils.data.Subset
    base = subset.dataset  # SEMDataset
    return [os.path.splitext(os.path.basename(base.image_paths[i]))[0] for i in subset.indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="logs/runs/deeplabv3plus-d83/checkpoints/best.pt")
    parser.add_argument("--experiment", default="configs/experiment/deeplabv3plus_dataset83.yaml")
    parser.add_argument("--output", default="predictions/error_maps_dlv3p_d83")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N val images (for debugging)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    exp_cfg = load_config(args.experiment)
    model_cfg = load_config(os.path.join("configs", exp_cfg["model"] + ".yaml"))
    data_cfg = load_config(os.path.join("configs", exp_cfg["data"] + ".yaml"))
    data_cfg["in_channels"] = model_cfg.get("in_channels", 1)
    data_cfg.update(exp_cfg.get("data_overrides", {}))
    data_cfg["batch_size"] = 1  # one image at a time for per-image reporting
    data_cfg["num_workers"] = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = create_model(model_cfg)
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch','?')})")

    dm = SEMDataModule(data_cfg)
    dm.setup()
    stems = get_val_stems(dm)

    loader = dm.val_dataloader()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    ranked_rows = []  # (stem, miou, f1, pxacc, per_class_iou...)

    with torch.no_grad():
        for idx, (img, gt) in enumerate(loader):
            if args.limit is not None and idx >= args.limit:
                break
            stem = stems[idx] if idx < len(stems) else f"val_{idx:03d}"
            img = img.to(device)
            out = model(img)
            if isinstance(out, dict):
                out = out["out"]
            pred = out.argmax(dim=1)  # (1, H, W)

            pred_np = pred[0].cpu().numpy().astype(np.int64)
            gt_np = gt[0].cpu().numpy().astype(np.int64)

            # Per-image metrics
            miou, f1, pxacc, per_class_iou = _compute_single(
                pred[0].cpu(), gt[0].cpu(), NUM_CLASSES,
            )

            # Accumulate global confusion
            m = (gt_np >= 0) & (gt_np < NUM_CLASSES)
            np.add.at(confusion, (gt_np[m], pred_np[m]), 1)

            disp = denormalize_display(img[0])
            out_path = os.path.join(args.output, f"{stem}.png")
            plot_per_image(disp, gt_np, pred_np, miou, per_class_iou, stem, out_path)

            ranked_rows.append([stem, miou, f1, pxacc] + [
                (v if not (isinstance(v, float) and np.isnan(v)) else "")
                for v in per_class_iou
            ])
            print(f"  [{idx+1:>3}/{len(loader)}] {stem:40s} miou={miou:.4f}")

    # Global per-class metrics from accumulated confusion
    per_class_iou_g = []
    per_class_f1_g = []
    for c in range(NUM_CLASSES):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        iou = tp / (tp + fp + fn + 1e-8) if (tp + fp + fn) > 0 else 0.0
        prec = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec + 1e-8) if (prec + rec) > 0 else 0.0
        per_class_iou_g.append(float(iou))
        per_class_f1_g.append(float(f1))

    plot_summary(
        confusion,
        per_class_iou_g,
        per_class_f1_g,
        os.path.join(args.output, "summary.png"),
    )
    print(f"Global miou: {np.mean(per_class_iou_g):.4f}")
    for n, v in zip(CLASS_NAMES, per_class_iou_g):
        print(f"  {n:18s} iou={v:.4f}")

    # Write ranked CSV (worst-first)
    ranked_rows.sort(key=lambda r: r[1])
    csv_path = os.path.join(args.output, "ranked.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "miou", "f1_macro", "pixel_accuracy"] +
                   [f"iou_{n}" for n in CLASS_NAMES])
        for row in ranked_rows:
            w.writerow(row)
    print(f"\nWrote {len(ranked_rows)} per-image panels + summary.png + ranked.csv to {args.output}/")


if __name__ == "__main__":
    main()
