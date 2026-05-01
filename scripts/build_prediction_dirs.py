"""Build organized per-image prediction directories.

For each image in the dataset, produces:
    predictions/<stem>/
        image.png
        ground_truth.png
        prediction.png
        error.png
        saliency/<class>.png     # Grad-CAM overlay, 6 files
        smoothgrad/<class>.png   # SmoothGrad overlay, 6 files

Usage:
    python scripts/build_prediction_dirs.py \
        --checkpoint logs/runs/deeplabv3plus-d83/checkpoints/best.pt \
        --experiment configs/experiment/deeplabv3plus_dataset83.yaml \
        --data configs/data/merged.yaml \
        --output predictions
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


def error_rgba(gt, pred):
    """RGBA overlay: wrong pixels colored by predicted class, correct pixels transparent."""
    wrong = gt != pred
    rgb = np.zeros((*gt.shape, 3), dtype=np.uint8)
    for i, c in enumerate(CLASS_COLORS):
        rgb[wrong & (pred == i)] = c
    rgba = np.zeros((*gt.shape, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = (wrong.astype(np.uint8)) * 200
    return rgba


def save_image_png(arr, path):
    """Save a numpy array as PNG. Handles grayscale (H,W), RGB (H,W,3), RGBA (H,W,4)."""
    if arr.ndim == 2:
        Image.fromarray(arr.astype(np.uint8), mode="L").save(path)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        Image.fromarray(arr.astype(np.uint8), mode="RGBA").save(path)
    else:
        raise ValueError(f"Unsupported shape {arr.shape}")


def save_overlay_heatmap(img_u8, heat, path, cmap="jet", alpha=0.5):
    """Save a figure: grayscale input with per-pixel heatmap overlay."""
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(img_u8, cmap="gray")
    ax.imshow(heat, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_error_on_input(img_u8, gt, pred, path):
    """Save error overlay (wrong pixels colored by predicted class) on top of input,
    with a right-side legend mapping each color to its class name."""
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.imshow(img_u8, cmap="gray")
    ax.imshow(error_rgba(gt, pred))
    wrong_frac = (gt != pred).mean() * 100
    ax.set_title(f"errors: {wrong_frac:.1f}% wrong, colored by predicted class",
                 fontsize=11)
    ax.axis("off")
    handles = [
        Patch(facecolor=np.array(c) / 255.0, edgecolor="black", label=n)
        for n, c in zip(CLASS_NAMES, CLASS_COLORS)
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=10, title="predicted class")
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


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


def pick_cam_layer(model):
    if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
        return model.backbone.layer4
    last_conv = None
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No conv layer for CAM target")
    return last_conv


def class_score(logits, c, gt_mask, device):
    m = (gt_mask == c)
    if m.sum() == 0:
        return logits[0, c].mean()
    mt = torch.from_numpy(m.astype(np.float32)).to(device)
    if mt.shape != logits.shape[-2:]:
        mt = F.interpolate(mt[None, None], size=logits.shape[-2:], mode="nearest")[0, 0]
    return (logits[0, c] * mt).sum() / (mt.sum() + 1e-8)


def compute_gradcam(model, x, gt, device):
    """Return (C,H,W) Grad-CAM heatmaps at input resolution, normalized per-class to [0,1]."""
    target_layer = pick_cam_layer(model)
    activations, gradients = {}, {}
    h1 = target_layer.register_forward_hook(lambda _m, _i, o: activations.__setitem__("v", o.detach()))
    h2 = target_layer.register_full_backward_hook(lambda _m, _gi, go: gradients.__setitem__("v", go[0].detach()))
    try:
        x = x.to(device)
        # Enable grads on backbone for CAM
        for p in model.parameters():
            p.requires_grad_(True)
        out = model(x)
        if isinstance(out, dict):
            out = out["out"]
        logits = out
        H, W = logits.shape[-2:]
        cams = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
        for c in range(NUM_CLASSES):
            model.zero_grad(set_to_none=True)
            score = class_score(logits, c, gt, device)
            score.backward(retain_graph=(c < NUM_CLASSES - 1))
            A = activations["v"][0]
            G = gradients["v"][0]
            w = G.mean(dim=(1, 2))
            cam = torch.relu((w[:, None, None] * A).sum(dim=0))
            cam = F.interpolate(cam[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            arr = cam.detach().cpu().numpy()
            mx = arr.max()
            cams[c] = arr / mx if mx > 0 else arr
    finally:
        h1.remove(); h2.remove()
    return cams


def compute_smoothgrad(model, x, gt, device, n_samples=8, sigma_frac=0.15):
    x = x.to(device)
    x_min, x_max = x.min().item(), x.max().item()
    sigma = sigma_frac * (x_max - x_min)
    H, W = x.shape[-2:]
    sal = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)

    # Freeze params — only input needs grad
    for p in model.parameters():
        p.requires_grad_(False)
    for c in range(NUM_CLASSES):
        agg = torch.zeros((H, W), dtype=torch.float32, device=device)
        for _ in range(n_samples):
            noise = torch.randn_like(x) * sigma
            x_n = (x + noise).detach().requires_grad_(True)
            out = model(x_n)
            if isinstance(out, dict):
                out = out["out"]
            score = class_score(out, c, gt, device)
            if x_n.grad is not None:
                x_n.grad.zero_()
            score.backward()
            g = x_n.grad.detach().abs()
            agg += g[0].max(dim=0).values
        arr = (agg / n_samples).cpu().numpy()
        mx = arr.max()
        sal[c] = arr / mx if mx > 0 else arr
    return sal


def list_image_mask_pairs(data_cfg):
    import glob
    image_dir = data_cfg["image_dir"]
    mask_dir = data_cfg["mask_dir"]
    paths = []
    for ext in ("*.png", "*.PNG", "*.jpg", "*.JPG"):
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    paths = sorted(p for p in paths
                   if os.path.exists(os.path.join(mask_dir, os.path.splitext(os.path.basename(p))[0] + ".npy")))
    return [(p, os.path.join(mask_dir, os.path.splitext(os.path.basename(p))[0] + ".npy")) for p in paths]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--output", default="predictions")
    ap.add_argument("--sg-samples", type=int, default=8)
    ap.add_argument("--sg-sigma", type=float, default=0.15)
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip stems whose dir already has all 16 expected files")
    args = ap.parse_args()

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
    print(f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch','?')}) on {device}")

    pairs = list_image_mask_pairs(data_cfg)
    print(f"Building {len(pairs)} per-image directories under {args.output}/")

    for i, (img_path, mask_path) in enumerate(pairs):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_dir = os.path.join(args.output, stem)
        sal_dir = os.path.join(out_dir, "saliency")
        sg_dir = os.path.join(out_dir, "smoothgrad")

        expected = (
            [os.path.join(out_dir, f) for f in ("image.png", "ground_truth.png", "prediction.png", "error.png")]
            + [os.path.join(sal_dir, f"{n}.png") for n in CLASS_NAMES]
            + [os.path.join(sg_dir, f"{n}.png") for n in CLASS_NAMES]
        )
        if args.skip_existing and all(os.path.exists(p) for p in expected):
            print(f"  [{i+1:>3}/{len(pairs)}] {stem}  (skip — complete)")
            continue

        os.makedirs(sal_dir, exist_ok=True)
        os.makedirs(sg_dir, exist_ok=True)

        x, img_u8, gt = load_image_mask(img_path, mask_path, in_channels)

        # Prediction
        with torch.no_grad():
            for p in model.parameters():
                p.requires_grad_(False)
            out = model(x.to(device))
            if isinstance(out, dict):
                out = out["out"]
            pred = out.argmax(dim=1)[0].cpu().numpy().astype(np.int64)

        # Base images
        save_image_png(img_u8, os.path.join(out_dir, "image.png"))
        save_image_png(colorize(gt), os.path.join(out_dir, "ground_truth.png"))
        save_image_png(colorize(pred), os.path.join(out_dir, "prediction.png"))
        save_error_on_input(img_u8, gt, pred, os.path.join(out_dir, "error.png"))

        # Grad-CAM (needs param grads)
        cams = compute_gradcam(model, x, gt, device)
        for c, name in enumerate(CLASS_NAMES):
            save_overlay_heatmap(img_u8, cams[c], os.path.join(sal_dir, f"{name}.png"), cmap="jet", alpha=0.55)

        # SmoothGrad (freezes params inside)
        sg = compute_smoothgrad(model, x, gt, device, n_samples=args.sg_samples, sigma_frac=args.sg_sigma)
        for c, name in enumerate(CLASS_NAMES):
            save_overlay_heatmap(img_u8, sg[c], os.path.join(sg_dir, f"{name}.png"), cmap="hot", alpha=0.6)

        print(f"  [{i+1:>3}/{len(pairs)}] {stem}")

    print(f"\nDone. Per-image dirs in {args.output}/")


if __name__ == "__main__":
    main()
