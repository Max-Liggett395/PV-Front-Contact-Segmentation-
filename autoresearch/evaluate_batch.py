"""Evaluate a batch of autoresearch experiments and print leaderboard.

Usage:
    python -m autoresearch.evaluate_batch --results-dir autoresearch/results
    python -m autoresearch.evaluate_batch --results-dir /runpod-volume/results
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Class names for per-class display
CLASS_NAMES = ["background", "silver", "glass", "silicon", "void", "interfacial_void"]

BASELINE_MIOU = 0.781


def load_results(results_dir):
    """Load all result JSON files from a directory."""
    results = []
    for f in sorted(Path(results_dir).glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
            data["_file"] = str(f)
            results.append(data)
    return results


def print_leaderboard(results):
    """Print ranked leaderboard sorted by val_miou."""
    ranked = sorted(results, key=lambda r: r.get("val_miou", 0), reverse=True)

    print("\n" + "=" * 90)
    print("AUTORESEARCH LEADERBOARD")
    print("=" * 90)
    print(f"{'Rank':<5} {'Run Name':<35} {'val_miou':<10} {'F1':<10} {'Acc':<10} {'Epoch':<7} {'Time':<8} {'Delta'}")
    print("-" * 90)

    for i, r in enumerate(ranked, 1):
        miou = r.get("val_miou", 0)
        delta = miou - BASELINE_MIOU
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        marker = " ***" if delta > 0 else ""

        print(
            f"{i:<5} "
            f"{r.get('run_name', '???'):<35} "
            f"{miou:<10.4f} "
            f"{r.get('val_f1_macro', 0):<10.4f} "
            f"{r.get('pixel_accuracy', 0):<10.4f} "
            f"{r.get('best_epoch', '?'):<7} "
            f"{r.get('elapsed_seconds', 0)/60:<8.1f} "
            f"{delta_str}{marker}"
        )

    print("-" * 90)
    print(f"Baseline: val_miou = {BASELINE_MIOU:.4f} (DeepLabV3+ RN101, merged dataset)")
    print(f"*** = improvement over baseline")
    print()

    # Per-class breakdown for top 3
    print("PER-CLASS IoU (top 3):")
    print(f"{'Run Name':<35} ", end="")
    for name in CLASS_NAMES:
        print(f"{name:<16} ", end="")
    print()
    print("-" * 130)

    for r in ranked[:3]:
        per_class = r.get("per_class_iou", [])
        print(f"{r.get('run_name', '???'):<35} ", end="")
        for iou in per_class:
            print(f"{iou:<16.4f} ", end="")
        print()

    print()

    # Winners
    winners = [r for r in ranked if r.get("val_miou", 0) > BASELINE_MIOU]
    if winners:
        print(f"WINNERS ({len(winners)} experiments beat baseline):")
        for r in winners:
            delta = r["val_miou"] - BASELINE_MIOU
            print(f"  + {r['run_name']}: val_miou={r['val_miou']:.4f} (+{delta:.4f})")
    else:
        print("NO WINNERS — no experiment beat the baseline in this batch.")

    return ranked


def main():
    global BASELINE_MIOU

    parser = argparse.ArgumentParser(description="Evaluate autoresearch batch")
    parser.add_argument("--results-dir", type=str, default="autoresearch/results")
    parser.add_argument("--baseline-miou", type=float, default=BASELINE_MIOU)
    args = parser.parse_args()

    BASELINE_MIOU = args.baseline_miou

    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        sys.exit(1)

    print(f"Loaded {len(results)} experiment results from {args.results_dir}")
    print_leaderboard(results)


if __name__ == "__main__":
    main()
