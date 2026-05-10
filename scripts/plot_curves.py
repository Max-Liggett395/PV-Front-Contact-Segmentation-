"""Plot training curves (loss, val mIoU, val pixel acc, val macro F1) from SLURM logs.

Each run shows up as one line per metric. Best epoch (by val_loss) is marked with a dot.

Usage:
    python scripts/plot_curves.py logs/slurm/sem-train-3277728.out
    python scripts/plot_curves.py logs/slurm/sem-train-*.out --out curves.png
    python scripts/plot_curves.py \
        unet=logs/slurm/sem-unet-new113-1234.out \
        dlv3=logs/slurm/sem-dlv3-new113-1235.out \
        --out compare.png

    # Also write one per-model plot per run to viz/training_curves/<label>.png
    python scripts/plot_curves.py unet=...out dlv3=...out \
        --out compare.png --per-model-dir viz/training_curves
"""

import argparse
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_NUM = r"([0-9.eE+-]+)"
EPOCH_RE = re.compile(rf"Epoch\s+(\d+)/\d+\s*\|\s*train_loss:\s*{_NUM}")
_FIELD_RES = {
    "miou": re.compile(rf"\bmiou:\s*{_NUM}"),
    "f1_macro": re.compile(rf"\bf1_macro:\s*{_NUM}"),
    "pixel_accuracy": re.compile(rf"\bpixel_accuracy:\s*{_NUM}"),
    "val_loss": re.compile(rf"\bval_loss:\s*{_NUM}"),
}


def parse_log(path):
    out = {k: [] for k in ("epoch", "train_loss", "miou", "f1_macro", "pixel_accuracy", "val_loss")}
    with open(path) as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if not m:
                continue
            fields = {k: r.search(line) for k, r in _FIELD_RES.items()}
            if not all(fields.values()):
                continue
            out["epoch"].append(int(m.group(1)))
            out["train_loss"].append(float(m.group(2)))
            for k, fm in fields.items():
                out[k].append(float(fm.group(1)))
    return out


def _set_titles(ax_loss, ax_miou, ax_pa, ax_f1):
    ax_loss.set_title("Loss"); ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("loss")
    ax_miou.set_title("Val mIoU"); ax_miou.set_xlabel("epoch"); ax_miou.set_ylabel("mIoU")
    ax_pa.set_title("Val Pixel Accuracy"); ax_pa.set_xlabel("epoch"); ax_pa.set_ylabel("pixel acc")
    ax_f1.set_title("Val Macro F1"); ax_f1.set_xlabel("epoch"); ax_f1.set_ylabel("macro F1")


def _plot_run(axes, d, label, color=None):
    """Plot one run's curves into the (2,2) axes. Returns the color used."""
    ax_loss, ax_miou = axes[0]
    ax_pa, ax_f1 = axes[1]
    ep = d["epoch"]
    line, = ax_loss.plot(
        ep, d["train_loss"], label=f"{label} train", alpha=0.6,
        **({"color": color} if color else {}),
    )
    c = color or line.get_color()
    ax_loss.plot(ep, d["val_loss"], color=c, linestyle="--", label=f"{label} val")
    ax_miou.plot(ep, d["miou"], color=c, label=label)
    ax_pa.plot(ep, d["pixel_accuracy"], color=c, label=label)
    ax_f1.plot(ep, d["f1_macro"], color=c, label=label)
    # Mark best val_loss epoch on every panel.
    i = min(range(len(d["val_loss"])), key=lambda k: d["val_loss"][k])
    ax_loss.scatter([ep[i]], [d["val_loss"][i]], color=c, zorder=5)
    ax_miou.scatter([ep[i]], [d["miou"][i]], color=c, zorder=5)
    ax_pa.scatter([ep[i]], [d["pixel_accuracy"][i]], color=c, zorder=5)
    ax_f1.scatter([ep[i]], [d["f1_macro"][i]], color=c, zorder=5)
    return c, i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="paths or label=path")
    ap.add_argument("--out", default="curves.png")
    ap.add_argument(
        "--per-model-dir",
        default=None,
        help="If set, also write one per-run plot to <dir>/<label>.png",
    )
    args = ap.parse_args()

    runs = []
    for arg in args.inputs:
        if "=" in arg and not os.path.exists(arg):
            label, path = arg.split("=", 1)
        else:
            path = arg
            label = os.path.splitext(os.path.basename(path))[0]
        runs.append((label, path))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _set_titles(axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1])

    if args.per_model_dir:
        os.makedirs(args.per_model_dir, exist_ok=True)

    for label, path in runs:
        d = parse_log(path)
        if not d["epoch"]:
            print(f"  WARN: no epochs parsed from {path}", file=sys.stderr)
            continue
        color, best_i = _plot_run(axes, d, label)
        ep = d["epoch"]
        print(
            f"  {label}: {len(ep)} epochs, best val_loss={d['val_loss'][best_i]:.4f} @ epoch {ep[best_i]}"
            f" (miou={d['miou'][best_i]:.4f}, f1_macro={d['f1_macro'][best_i]:.4f},"
            f" pixel_acc={d['pixel_accuracy'][best_i]:.4f})"
        )

        if args.per_model_dir:
            pm_fig, pm_axes = plt.subplots(2, 2, figsize=(11, 8))
            _set_titles(pm_axes[0, 0], pm_axes[0, 1], pm_axes[1, 0], pm_axes[1, 1])
            _plot_run(pm_axes, d, label, color=color)
            pm_fig.suptitle(f"Training history — {label}", fontsize=13)
            for ax in pm_axes.flat:
                ax.legend(fontsize=8)
            pm_fig.tight_layout()
            pm_path = os.path.join(args.per_model_dir, f"{label}.png")
            pm_fig.savefig(pm_path, dpi=150)
            plt.close(pm_fig)
            print(f"    -> {pm_path}")

    for ax in axes.flat:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
