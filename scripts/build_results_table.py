"""Assemble the final metrics table from each run's best.pt checkpoint.

Reads logs/runs/<run>/checkpoints/best.pt for each run, pulls the val metrics
that compute_metrics() saved at the best epoch, and prints / writes a Markdown
table with: Macro F1, Micro F1, Pixel Accuracy, mIoU.

Note: for multiclass single-label segmentation, Micro F1 == Pixel Accuracy by
construction (every pixel contributes exactly one TP or one FP+FN). Both are
shown for completeness.

Usage:
    python scripts/build_results_table.py \
        unet=logs/runs/smp-unet-new113 \
        unetpp=logs/runs/smp-unetpp-new113 \
        dlv3=logs/runs/smp-dlv3-new113 \
        dlv3p=logs/runs/smp-dlv3p-new113 \
        segformer=logs/runs/smp-segformer-new113 \
        --out logs/results_new113.md
"""

import argparse
import os
import sys

import torch


COLUMNS = [
    ("Macro F1", "f1_macro"),
    ("Micro F1", "f1_micro"),
    ("Pixel Acc", "pixel_accuracy"),
    ("mIoU", "miou"),
]


def load_best_metrics(run_dir):
    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    if not os.path.exists(ckpt_path):
        return None, f"missing: {ckpt_path}"
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    metrics = ckpt.get("metrics", {})
    epoch = ckpt.get("epoch", "?")
    # Fill in f1_micro if the checkpoint pre-dates the metrics update.
    if "f1_micro" not in metrics and "pixel_accuracy" in metrics:
        metrics["f1_micro"] = metrics["pixel_accuracy"]
    return {"epoch": epoch, **metrics}, None


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "runs",
        nargs="+",
        help="label=path/to/run_dir entries, e.g. unet=logs/runs/smp-unet-new113",
    )
    ap.add_argument("--out", default=None, help="Optional Markdown output path.")
    args = ap.parse_args()

    rows = []
    for arg in args.runs:
        if "=" not in arg:
            print(f"  ERROR: expected label=path, got {arg}", file=sys.stderr)
            sys.exit(2)
        label, path = arg.split("=", 1)
        m, err = load_best_metrics(path)
        if err:
            print(f"  WARN {label}: {err}", file=sys.stderr)
            rows.append((label, "?", {k: None for _, k in COLUMNS}))
        else:
            rows.append((label, m.get("epoch", "?"), m))

    header = "| Model | Best Epoch | " + " | ".join(c for c, _ in COLUMNS) + " |"
    sep = "|" + "|".join(["---"] * (len(COLUMNS) + 2)) + "|"
    body = []
    for label, epoch, m in rows:
        cells = [label, str(epoch)] + [fmt(m.get(k)) for _, k in COLUMNS]
        body.append("| " + " | ".join(cells) + " |")

    lines = [
        header,
        sep,
        *body,
        "",
        "_Micro F1 == Pixel Accuracy for multiclass single-label segmentation by construction._",
    ]
    table = "\n".join(lines)
    print(table)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table + "\n")
        print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
