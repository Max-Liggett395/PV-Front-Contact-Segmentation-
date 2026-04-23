"""Grad-CAM saliency maps for a segmentation model.

For each target image: hooks the backbone's last conv block, computes the
gradient of the sum of logits for each target class (summed over pixels GT-labeled
that class, or over the whole image if that class is absent) w.r.t. the feature
map, then forms the standard Grad-CAM heatmap (weighted-sum of activations,
ReLU, upsample).

Produces per-image figures with a row of per-class CAMs overlaid on the input.

Usage:
    python scripts/saliency_maps.py \
        --checkpoint logs/runs/deeplabv3plus-d83/checkpoints/best.pt \
        --experiment configs/experiment/deeplabv3plus_dataset83.yaml \
        --data configs/data/merged.yaml \
        --stems-from predictions/error_maps_dlv3p_d83_on_merged/ranked.csv \
        --top-k 5 \
        --output predictions/saliency_dlv3p_d83_on_merged
"""

import argparse
import csv
import os
import sys
import types

import numpy as np

# numpy._core shim for checkpoints pickled with numpy>=2.0 under numpy<2.0
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
    if in_channels == 3:
        model_in = np.stack([img_arr, img_arr, img_arr], axis=-1)
    else:
        model_in = img_arr
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


def pick_cam_layer(model):
    """Return the conv/module whose activations+gradients feed Grad-CAM."""
    # torchvision deeplabv3 / deeplabv3+: backbone is a ResNet; layer4 is the last conv block.
    if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
        return model.backbone.layer4
    # SMP or custom: try to find a last conv by walking named_modules
    last_conv = None
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No conv layer found for CAM target")
    return last_conv


def gradcam(model, input_tensor, gt_mask, device):
    """Return (C_classes, H, W) CAM heatmap array in [0,1] plus logits argmax pred."""
    model.eval()
    target_layer = pick_cam_layer(model)

    activations = {}
    gradients = {}

    def fwd_hook(_m, _i, o):
        activations["v"] = o.detach()

    def bwd_hook(_m, grad_in, grad_out):
        gradients["v"] = grad_out[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        input_tensor = input_tensor.to(device).requires_grad_(False)
        out = model(input_tensor)
        if isinstance(out, dict):
            out = out["out"]
        logits = out  # (1, C, H, W)
        pred = logits.argmax(dim=1)[0].cpu().numpy()

        H, W = logits.shape[-2:]
        cams = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
        for c in range(NUM_CLASSES):
            model.zero_grad(set_to_none=True)
            # Score = mean logit for class c over regions where GT == c,
            # or over entire image if that class is absent (so we still get a CAM).
            mask_c = (gt_mask == c)
            if mask_c.sum() == 0:
                score = logits[0, c].mean()
            else:
                mt = torch.from_numpy(mask_c.astype(np.float32)).to(device)
                # Resize mask to logits spatial size if needed
                if mt.shape != logits.shape[-2:]:
                    mt = F.interpolate(mt[None, None], size=logits.shape[-2:], mode="nearest")[0, 0]
                score = (logits[0, c] * mt).sum() / (mt.sum() + 1e-8)
            score.backward(retain_graph=(c < NUM_CLASSES - 1))

            A = activations["v"][0]      # (Cfeat, h, w)
            G = gradients["v"][0]        # (Cfeat, h, w)
            weights = G.mean(dim=(1, 2)) # (Cfeat,)
            cam = (weights[:, None, None] * A).sum(dim=0)  # (h, w)
            cam = torch.relu(cam)
            cam = F.interpolate(cam[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            cam = cam.cpu().numpy()
            cmax = cam.max()
            if cmax > 0:
                cam = cam / cmax
            cams[c] = cam
    finally:
        h1.remove(); h2.remove()

    return cams, pred


def plot_cams(img_u8, gt_mask, pred_mask, cams, stem, out_path):
    """3-row figure: (row1) input, GT, pred; (row2) per-class CAM overlay; (row3) per-class CAM heatmap."""
    ncols = max(3, NUM_CLASSES)
    fig, axes = plt.subplots(3, ncols, figsize=(4 * ncols, 10))

    # Row 0: input / GT / pred / (padding)
    axes[0, 0].imshow(img_u8, cmap="gray"); axes[0, 0].set_title("input"); axes[0, 0].axis("off")
    axes[0, 1].imshow(colorize(gt_mask)); axes[0, 1].set_title("GT"); axes[0, 1].axis("off")
    axes[0, 2].imshow(colorize(pred_mask)); axes[0, 2].set_title("pred"); axes[0, 2].axis("off")
    for j in range(3, ncols):
        axes[0, j].axis("off")

    # Rows 1-2: per-class CAM overlay + raw CAM
    for c in range(NUM_CLASSES):
        ax_ov = axes[1, c]
        ax_ov.imshow(img_u8, cmap="gray")
        ax_ov.imshow(cams[c], cmap="jet", alpha=0.5, vmin=0, vmax=1)
        ax_ov.set_title(f"CAM: {CLASS_NAMES[c]}"); ax_ov.axis("off")

        ax_raw = axes[2, c]
        ax_raw.imshow(cams[c], cmap="jet", vmin=0, vmax=1)
        ax_raw.set_title(f"CAM heatmap"); ax_raw.axis("off")
    for c in range(NUM_CLASSES, ncols):
        axes[1, c].axis("off"); axes[2, c].axis("off")

    fig.suptitle(stem, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def load_ranked_stems(csv_path, k):
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    rows.sort(key=lambda r: float(r["miou"]))
    return [r["stem"] for r in rows[:k]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--data", default=None, help="Data config (defaults to experiment's)")
    p.add_argument("--stems-from", default=None, help="ranked.csv from error_maps.py — takes --top-k worst")
    p.add_argument("--stems", nargs="*", default=None, help="Explicit list of image stems")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)

    exp_cfg = load_config(args.experiment)
    model_cfg = load_config(os.path.join("configs", exp_cfg["model"] + ".yaml"))
    if args.data is not None:
        data_cfg = load_config(args.data)
    else:
        data_cfg = load_config(os.path.join("configs", exp_cfg["data"] + ".yaml"))
    in_channels = model_cfg.get("in_channels", 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_cfg)
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    for p_ in model.parameters():
        p_.requires_grad_(True)  # need grads for CAM
    print(f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch','?')}) on {device}")

    # Resolve stems
    if args.stems_from:
        stems = load_ranked_stems(args.stems_from, args.top_k)
    elif args.stems:
        stems = args.stems
    else:
        raise SystemExit("Provide --stems-from ranked.csv OR --stems <stems...>")
    print(f"Target stems: {stems}")

    image_dir = data_cfg["image_dir"]
    mask_dir = data_cfg["mask_dir"]

    for stem in stems:
        img_path = None
        for ext in (".png", ".PNG", ".jpg", ".JPG", ".tif", ".tiff"):
            c = os.path.join(image_dir, stem + ext)
            if os.path.exists(c):
                img_path = c; break
        if img_path is None:
            print(f"  skip {stem}: no matching image file")
            continue
        mask_path = os.path.join(mask_dir, stem + ".npy")
        if not os.path.exists(mask_path):
            print(f"  skip {stem}: no mask")
            continue

        tensor, img_u8, gt_mask = load_image_mask(img_path, mask_path, in_channels)
        cams, pred = gradcam(model, tensor, gt_mask, device)
        out_path = os.path.join(args.output, f"{stem}_cam.png")
        plot_cams(img_u8, gt_mask, pred, cams, stem, out_path)
        print(f"  {stem} -> {out_path}")

    print(f"\nDone. Saliency maps in {args.output}/")


if __name__ == "__main__":
    main()
