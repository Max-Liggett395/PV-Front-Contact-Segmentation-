"""Build the 113-image dataset (data/new_113) from the VIA-CSV checkpoint.

Reads CSV polygons/polylines from /tmp/113_Image_Checkpoint, rasterizes them
into uint8 masks (6 classes) at the original image resolution, and saves
images as .png + masks as .npy under data/new_113/.

Class scheme matches the rest of the codebase (see autoresearch/prepare.py):
  0 background
  1 silver  (also accepts typo "sliver")
  2 glass
  3 silicon
  4 void
  5 interfacial_void  (CSV "interfacial void")
"""

import csv
import json
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

SRC = "/tmp/113_Image_Checkpoint"
CSV_PATH = os.path.join(SRC, "113_Image_Checkpoint.csv")
DST_IMG = "data/new_113/images"
DST_MASK = "data/new_113/masks"

CLASS_MAP = {
    "silver": 1,
    "sliver": 1,            # observed typo in CSV
    "glass": 2,
    "silicon": 3,
    "void": 4,
    "interfacial void": 5,
    "interfacial_void": 5,
}


def normalize_type(t: str) -> str:
    return t.strip().lower().replace("\n", "").strip()


def main() -> int:
    os.makedirs(DST_IMG, exist_ok=True)
    os.makedirs(DST_MASK, exist_ok=True)

    # Group rows by filename so we paint all regions onto a single mask.
    by_file: dict[str, list[dict]] = {}
    with open(CSV_PATH, newline="") as f:
        for r in csv.DictReader(f):
            by_file.setdefault(r["filename"], []).append(r)

    skipped_unknown_type: Counter[str] = Counter()
    skipped_short: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()

    for fname, rows in sorted(by_file.items()):
        src_img = os.path.join(SRC, fname)
        if not os.path.exists(src_img):
            print(f"  MISSING image: {fname}", file=sys.stderr)
            continue

        # Save image as a normalised .png with the same stem.
        img = Image.open(src_img).convert("RGBA")
        stem = os.path.splitext(fname)[0]
        out_img = os.path.join(DST_IMG, stem + ".png")
        img.save(out_img)

        W, H = img.size
        mask_im = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(mask_im)

        # Bucket by class so we can paint in a deterministic order — bulk
        # materials first (silver/silicon/glass), then defects (void,
        # interfacial_void) on top.  Otherwise a large silver polygon drawn
        # late in the CSV would clobber the small voids drawn first.
        PAINT_ORDER = [1, 3, 2, 4, 5]  # silver, silicon, glass, void, interfacial_void
        buckets: dict[int, list[tuple[str, dict]]] = {c: [] for c in PAINT_ORDER}

        for r in rows:
            try:
                shape = json.loads(r["region_shape_attributes"]) if r["region_shape_attributes"] else {}
                attrs = json.loads(r["region_attributes"]) if r["region_attributes"] else {}
            except Exception:
                continue
            tname = normalize_type(attrs.get("type", ""))
            cls = CLASS_MAP.get(tname)
            if cls is None:
                skipped_unknown_type[tname or "<missing>"] += 1
                continue
            buckets[cls].append((shape.get("name", ""), shape))

        for cls in PAINT_ORDER:
            for sname, shape in buckets[cls]:
                shape_counts[sname] += 1
                if sname in ("polygon", "polyline"):
                    xs = shape.get("all_points_x", [])
                    ys = shape.get("all_points_y", [])
                    if len(xs) < 3 or len(ys) < 3 or len(xs) != len(ys):
                        skipped_short[sname] += 1
                        continue
                    pts = list(zip(xs, ys))
                    # Polylines in the source delineate blobs the annotator
                    # didn't bother to close — fill them like polygons.
                    draw.polygon(pts, fill=cls)
                elif sname == "rect":
                    x = int(shape.get("x") or 0)
                    y = int(shape.get("y") or 0)
                    w = int(shape.get("width") or 0)
                    h = int(shape.get("height") or 0)
                    if w <= 0 or h <= 0:
                        skipped_short[sname] += 1
                        continue
                    draw.rectangle([x, y, x + w, y + h], fill=cls)
                else:
                    skipped_short[sname or "<missing>"] += 1

        mask = np.array(mask_im, dtype=np.uint8)
        np.save(os.path.join(DST_MASK, stem + ".npy"), mask)

    n_img = len(os.listdir(DST_IMG))
    n_mask = len(os.listdir(DST_MASK))
    print(f"Wrote {n_img} images to {DST_IMG}")
    print(f"Wrote {n_mask} masks to {DST_MASK}")
    print(f"Shape counts: {shape_counts}")
    if skipped_unknown_type:
        print(f"Skipped regions w/ unknown type: {skipped_unknown_type}")
    if skipped_short:
        print(f"Skipped degenerate shapes: {skipped_short}")

    # Sanity check: aggregate class distribution
    total = Counter()
    for f in sorted(os.listdir(DST_MASK)):
        m = np.load(os.path.join(DST_MASK, f))
        u, c = np.unique(m, return_counts=True)
        for k, v in zip(u.tolist(), c.tolist()):
            total[k] += v
    s = sum(total.values())
    names = ["background", "silver", "glass", "silicon", "void", "interfacial_void"]
    print("Class distribution:")
    for c in range(6):
        n = total.get(c, 0)
        print(f"  {c} {names[c]:>18s}: {n:>12,} ({100.0 * n / s:5.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
