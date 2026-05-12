"""Re-evaluate the five new113 best.pt checkpoints and emit a per-class F1 table.

Writes a Markdown file at logs/per_class_f1_new113.md with one row per model and
one column per class. Uses the same val split as training (seed=42, 0.85 train).
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import SEMDataModule
from src.data.dataset import CLASS_NAMES
from src.evaluation.metrics import compute_metrics
from src.models import create_model
from src.utils.config import load_config


RUNS = [
    ("dlv3",      "smp-dlv3-new113",      "configs/experiment/smp_deeplabv3_new113.yaml"),
    ("dlv3p",     "smp-dlv3p-new113",     "configs/experiment/smp_deeplabv3plus_new113.yaml"),
    ("unet",      "smp-unet-new113",      "configs/experiment/smp_unet_new113.yaml"),
    ("unetpp",    "smp-unetpp-new113",    "configs/experiment/smp_unetpp_new113.yaml"),
    ("segformer", "smp-segformer-new113", "configs/experiment/smp_segformer_new113.yaml"),
]


def evaluate_run(run_dir, exp_config, device):
    ckpt_path = os.path.join("logs", "runs", run_dir, "checkpoints", "best.pt")
    if not os.path.isfile(ckpt_path):
        return None, f"missing {ckpt_path}"

    exp_cfg = load_config(exp_config)
    data_cfg = load_config(os.path.join("configs", exp_cfg["data"] + ".yaml"))
    model_cfg = load_config(os.path.join("configs", exp_cfg["model"] + ".yaml"))

    data_cfg["in_channels"] = model_cfg.get("in_channels", 1)
    data_cfg.update(exp_cfg.get("data_overrides", {}))

    dm = SEMDataModule(data_cfg)
    dm.setup()

    model = create_model(model_cfg)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    num_classes = model_cfg.get("num_classes", 6)
    preds, targets = [], []
    with torch.no_grad():
        for images, masks in dm.val_dataloader():
            images = images.to(device)
            logits = model(images)
            if isinstance(logits, dict):
                logits = logits["out"]
            preds.append(logits.argmax(dim=1).cpu())
            targets.append(masks)

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    metrics = compute_metrics(preds, targets, num_classes)
    metrics["best_epoch"] = ckpt.get("epoch", "?")
    return metrics, None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    rows = []
    for label, run_dir, exp_cfg in RUNS:
        print(f"Evaluating {label}...")
        m, err = evaluate_run(run_dir, exp_cfg, device)
        if err:
            print(f"  WARN: {err}")
            rows.append((label, None))
        else:
            rows.append((label, m))

    classes = CLASS_NAMES[:6]
    header = "| Model | " + " | ".join(classes) + " | Macro F1 |"
    sep = "|" + "|".join(["---"] * (len(classes) + 2)) + "|"
    body = []
    for label, m in rows:
        if m is None:
            cells = [label] + ["-"] * (len(classes) + 1)
        else:
            f1s = [f"{m['per_class_f1'][i]:.4f}" for i in range(len(classes))]
            cells = [label] + f1s + [f"{m['f1_macro']:.4f}"]
        body.append("| " + " | ".join(cells) + " |")

    lines = [
        "# Per-Class F1 (val split, seed=42)",
        "",
        header,
        sep,
        *body,
    ]
    out_path = "logs/per_class_f1_new113.md"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
