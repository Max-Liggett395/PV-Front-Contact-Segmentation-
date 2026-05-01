"""
Aggregate and compare all autoresearch experiment results.
Reads logs/results.jsonl and prints a ranked comparison table.
"""

import json
import os

CLASS_NAMES = ["background", "silver", "glass", "silicon", "void", "interfacial_void"]

def load_results():
    log_file = os.path.join(os.path.dirname(__file__), "logs", "results.jsonl")
    if not os.path.exists(log_file):
        print(f"No results file found at {log_file}")
        return []
    results = []
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def print_results(results):
    if not results:
        print("No results to display.")
        return

    # Sort by val_f1 descending
    results.sort(key=lambda r: r.get("val_f1", 0), reverse=True)

    print("=" * 120)
    print("AUTORESEARCH RESULTS — Ranked by val_f1")
    print("=" * 120)

    header = f"{'Rank':>4}  {'Track':<22} {'val_f1':>7} {'val_miou':>8} {'px_acc':>7} {'epoch':>7} {'time':>6}  {'Architecture':<20} {'code_hash':<10}"
    print(header)
    print("-" * 120)

    for i, r in enumerate(results, 1):
        track = r.get("track", "baseline")
        config = r.get("config", {})
        arch = config.get("architecture", "?")
        epoch_str = f"{r.get('best_epoch', '?')}/{r.get('max_epochs', '?')}"
        elapsed = r.get("elapsed_seconds", 0)
        time_str = f"{elapsed/60:.0f}m" if elapsed else "?"
        print(f"{i:>4}  {track:<22} {r.get('val_f1', 0):>7.4f} {r.get('val_miou', 0):>8.4f} "
              f"{r.get('pixel_accuracy', 0):>7.4f} {epoch_str:>7} {time_str:>6}  {arch:<20} {r.get('code_hash', '?'):<10}")

    # Per-class F1 breakdown for top 5
    print("\n" + "=" * 120)
    print("PER-CLASS F1 — Top 5 Experiments")
    print("=" * 120)

    cls_header = f"{'Rank':>4}  {'Track':<22} " + "  ".join(f"{c:>15}" for c in CLASS_NAMES)
    print(cls_header)
    print("-" * 120)

    for i, r in enumerate(results[:5], 1):
        track = r.get("track", "baseline")
        per_f1 = r.get("per_class_f1", [0] * 6)
        f1_str = "  ".join(f"{v:>15.4f}" for v in per_f1)
        print(f"{i:>4}  {track:<22} {f1_str}")

    # Group by track
    print("\n" + "=" * 120)
    print("BEST PER TRACK")
    print("=" * 120)

    tracks = {}
    for r in results:
        t = r.get("track", "baseline")
        if t not in tracks or r.get("val_f1", 0) > tracks[t].get("val_f1", 0):
            tracks[t] = r

    for track, r in sorted(tracks.items(), key=lambda x: x[1].get("val_f1", 0), reverse=True):
        config = r.get("config", {})
        print(f"\n  {track}: val_f1={r.get('val_f1', 0):.4f} | val_miou={r.get('val_miou', 0):.4f}")
        print(f"    Config: {json.dumps(config, indent=None)}")

    print("\n" + "=" * 120)
    best = results[0]
    print(f"OVERALL BEST: val_f1={best.get('val_f1', 0):.4f} from track={best.get('track', '?')}")
    print(f"  Config: {json.dumps(best.get('config', {}))}")
    print("=" * 120)


if __name__ == "__main__":
    results = load_results()
    print_results(results)
