"""SmoothGrad pixel-resolved saliency for a segmentation model.

For each image + target class c, compute:
    SG(x) = (1/N) sum_i |grad_x score_c(x + eps_i)|
where score_c = mean logit for class c over GT==c pixels (or whole image if absent),
and eps_i ~ N(0, sigma^2 * (x_max - x_min)^2).

Outputs one figure per image with: input, GT, pred in row 1, then per-class
SmoothGrad heatmaps (input resolution) in rows 2-3.

Usage:
    python scripts/smoothgrad.py \
        --checkpoint logs/runs/deeplabv3plus-d83/checkpoints/best.pt \
        --experiment configs/experiment/deeplabv3plus_dataset83.yaml \
        --data configs/data/merged.yaml \
        --stems-from predictions/error_maps_dlv3p_d83_on_merged/ranked.csv \
        --top-k 5 \
        --output predictions/smoothgrad_dlv3p_d83_on_merged
"""

import argparse
import csv
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
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import CLASS_NAMES, CLASS_COLORS, NUM_CLASSES, get_val_transform
from src.models import create_model
from src.utils.config import load_config


def colorize(mask):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, c in enumerate(CLASS_COLORS):
        out[mask == i] = c
    return out


def load_image_mask(img_path, mask_path, in_channels):
    img = Image.open(img_path).convert("L").resize((1024, 768), Image.BILINEAR)
    img_arr = np.array(img, dtype=np.float32)
    mask = np.load(mask_path)
    mask = np.array(Image.fromarray(mask).resize((1024, 768), Image.NEAREST))
    model_in = np.stack([img_arr] * 3, axis=-1) if in_channels == 3 else img_arr
    tf = get_val_transform(in_channels)
    out = tf(image=model_in, mask=mask)
    tensor = out["image"]
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor, dtype=torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 3 and in_channels == 3 and tensor.shape[0] != 3:
        tensor = tensor.permute(2, 0, 1)
    return tensor.unsqueeze(0), img_arr.astype(np.uint8), mask.astype(np.int64)


def class_score(logits, c, gt_mask, device):
    """Mean logit for class c over GT==c pixels; fallback to whole-image mean."""
    m = (gt_mask == c)
    if m.sum() == 0:
        return logits[0, c].mean()
    mt = torch.from_numpy(m.astype(np.float32)).to(device)
    if mt.shape != logits.shape[-2:]:
        mt = F.interpolate(mt[None, None], size=logits.shape[-2:], mode="nearest")[0, 0]
    return (logits[0, c] * mt).sum() / (mt.sum() + 1e-8)


def smoothgrad(model, x, gt_mask, device, n_samples=16, sigma_frac=0.15):
    """Return (NUM_CLASSES, H, W) float32 saliency maps normalized per-class to [0,1]."""
    x = x.to(device)
    x_min, x_max = x.min().item(), x.max().item()
    sigma = sigma_frac * (x_max - x_min)
    H, W = x.shape[-2:]
    sal = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)

    for c in range(NUM_CLASSES):
        agg = torch.zeros(x.shape[-2:], dtype=torch.float32, device=device)
        for _ in range(n_samples):
            noise = torch.randn_like(x) * sigma
            x_noisy = (x + noise).detach().requires_grad_(True)
            out = model(x_noisy)
            if isinstance(out, dict):
                out = out["out"]
            score = class_score(out, c, gt_mask, device)
            model.zero_grad(set_to_none=True)
            if x_noisy.grad is not None:
                x_noisy.grad.zero_()
            score.backward()
            g = x_noisy.grad.detach().abs()
            # collapse channel dim (max across channels is more informative than mean for saliency)
            g_map = g[0].max(dim=0).values
            agg += g_map
        agg = agg / n_samples
        arr = agg.cpu().numpy()
        mx = arr.max()
        if mx > 0:
            arr = arr / mx
        sal[c] = arr
    return sal


def predict(model, x, device):
    with torch.no_grad():
        out = model(x.to(device))
        if isinstance(out, dict):
            out = out["out"]
        return out.argmax(dim=1)[0].cpu().numpy()


def plot_sg(img_u8, gt_mask, pred_mask, sal, stem, out_path):
    ncols = max(3, NUM_CLASSES)
    fig, axes = plt.subplots(3, ncols, figsize=(4 * ncols, 10))

    axes[0, 0].imshow(img_u8, cmap="gray"); axes[0, 0].set_title("input"); axes[0, 0].axis("off")
    axes[0, 1].imshow(colorize(gt_mask)); axes[0, 1].set_title("GT"); axes[0, 1].axis("off")
    axes[0, 2].imshow(colorize(pred_mask)); axes[0, 2].set_title("pred"); axes[0, 2].axis("off")
    for j in range(3, ncols):
        axes[0, j].axis("off")

    for c in range(NUM_CLASSES):
        ax_ov = axes[1, c]
        ax_ov.imshow(img_u8, cmap="gray")
        ax_ov.imshow(sal[c], cmap="hot", alpha=0.55, vmin=0, vmax=1)
        ax_ov.set_title(f"SG: {CLASS_NAMES[c]}"); ax_ov.axis("off")

        ax_raw = axes[2, c]
        ax_raw.imshow(sal[c], cmap="hot", vmin=0, vmax=1)
        ax_raw.set_title("SG heatmap"); ax_raw.axis("off")
    for c in range(NUM_CLASSES, ncols):
        axes[1, c].axis("off"); axes[2, c].axis("off")

    fig.suptitle(f"{stem}  (SmoothGrad, n=16)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def load_ranked_stems(csv_path, k):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r["miou"]))
    return [r["stem"] for r in rows[:k]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--data", default=None)
    p.add_argument("--stems-from", default=None)
    p.add_argument("--stems", nargs="*", default=None)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--sigma-frac", type=float, default=0.15)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    exp_cfg = load_config(args.experiment)
    model_cfg = load_config(os.path.join("configs", exp_cfg["model"] + ".yaml"))
    data_cfg = load_config(args.data) if args.data else load_config(os.path.join("configs", exp_cfg["data"] + ".yaml"))
    in_channels = model_cfg.get("in_channels", 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_cfg)
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    for pr in model.parameters():
        pr.requires_grad_(False)  # we only need grads w.r.t. input
    print(f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch','?')}) on {device}")

    if args.stems_from:
        stems = load_ranked_stems(args.stems_from, args.top_k)
    elif args.stems:
        stems = args.stems
    else:
        raise SystemExit("Need --stems-from or --stems")
    print(f"Target stems: {stems}")
    print(f"SmoothGrad: n_samples={args.n_samples} sigma_frac={args.sigma_frac}")

    image_dir = data_cfg["image_dir"]
    mask_dir = data_cfg["mask_dir"]

    for stem in stems:
        img_path = None
        for ext in (".png", ".PNG", ".jpg", ".JPG", ".tif", ".tiff"):
            c = os.path.join(image_dir, stem + ext)
            if os.path.exists(c):
                img_path = c; break
        mask_path = os.path.join(mask_dir, stem + ".npy")
        if img_path is None or not os.path.exists(mask_path):
            print(f"  skip {stem}: missing files"); continue

        x, img_u8, gt = load_image_mask(img_path, mask_path, in_channels)
        pred = predict(model, x, device)
        sal = smoothgrad(model, x, gt, device, n_samples=args.n_samples, sigma_frac=args.sigma_frac)
        out_path = os.path.join(args.output, f"{stem}_sg.png")
        plot_sg(img_u8, gt, pred, sal, stem, out_path)
        print(f"  {stem} -> {out_path}")

    print(f"\nDone. SmoothGrad maps in {args.output}/")


if __name__ == "__main__":
    main()
