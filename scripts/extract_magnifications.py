#!/usr/bin/env python3
"""
Extract magnification values from SEM image metadata bars via OCR.

Crops the bottom ~12% of each image (metadata bar region), runs pytesseract
OCR, and regex-matches the magnification value. Handles two formats:
  - "Mag = 10.00 KX" (kilox magnification, stored as-is in KX)
  - "Mag = 950 X" (plain magnification, converted to KX by dividing by 1000)

Cross-validates with filename-based magnification where available.
All values are stored in KX units.

Outputs: data/magnification_map.json  {"filename_stem": magnification_float_in_KX, ...}

Usage:
    python scripts/extract_magnifications.py
    python scripts/extract_magnifications.py --images-dir data/images --output data/magnification_map.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytesseract
from PIL import Image, ImageFilter


def extract_mag_from_filename(stem: str) -> float | None:
    """Try to extract magnification (in KX) from filename patterns like '_10K_', '_2K_', '_5K_', '_2.5K_', '_4K1'."""
    # Match patterns like _10K_, _2K_, _3.5K_, _4K1 (trailing digit, not 'K' suffix)
    match = re.search(r'_(\d+\.?\d*)K[_\d]', stem, re.IGNORECASE)
    if match:
        return float(match.group(1))
    # Match patterns at end of filename like _4K1
    match = re.search(r'_(\d+\.?\d*)K(\d*)$', stem, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def extract_mag_from_ocr(image_path: Path, crop_fraction: float = 0.12) -> tuple[float | None, str | None]:
    """
    OCR the bottom metadata bar and extract magnification value in KX units.

    Returns:
        (mag_value_in_KX, unit_string) or (None, None) on failure
    """
    img = Image.open(image_path).convert("L")
    w, h = img.size

    # Crop bottom region (metadata bar)
    crop_h = int(h * crop_fraction)
    bottom = img.crop((0, h - crop_h, w, h))

    # Enhance for better OCR: sharpen and increase contrast
    bottom = bottom.filter(ImageFilter.SHARPEN)

    # Try OCR with different configs for robustness
    configs = [
        '--psm 6',   # Assume uniform block of text
        '--psm 11',  # Sparse text
        '--psm 3',   # Fully automatic
    ]

    for config in configs:
        text = pytesseract.image_to_string(bottom, config=config)

        # Pattern 1: "Mag = 10.00 KX" or "Mag = 10.00 K X" (KX units)
        match = re.search(r'Mag\s*=?\s*(\d+\.?\d*)\s*K\s*X', text, re.IGNORECASE)
        if match:
            return float(match.group(1)), "KX"

        # Pattern 2: "Mag = 950 X" (plain X units, no K prefix)
        match = re.search(r'Mag\s*=?\s*(\d+\.?\d*)\s*X', text, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            # Distinguish KX from X: if there was no "K" before "X", it's plain X
            # Double-check this isn't a KX match we missed
            return val, "X"

    return None, None


def resolve_magnification(stem: str, ocr_mag: float | None, ocr_unit: str | None,
                          filename_mag: float | None) -> float | None:
    """
    Resolve the final magnification value in KX units, handling OCR decimal misreads.

    OCR commonly drops the decimal point (e.g., "10.00" → "1000"), so we cross-validate
    with filename-based values and apply heuristic corrections.
    """
    if ocr_mag is not None and ocr_unit == "KX":
        # OCR read KX format — but might have dropped decimal point
        # Real SEM magnifications in KX are typically 1-50, never > 100
        if ocr_mag > 50 and filename_mag is not None:
            # Filename available: prefer it (more reliable for this case)
            return filename_mag
        elif ocr_mag > 50:
            # No filename ref: heuristic — divide by 100 to restore decimal
            return ocr_mag / 100.0
        else:
            return ocr_mag

    elif ocr_mag is not None and ocr_unit == "X":
        # Plain X units (e.g., 950 X) — convert to KX
        return ocr_mag / 1000.0

    elif filename_mag is not None:
        return filename_mag

    return None


def main():
    parser = argparse.ArgumentParser(description="Extract magnifications from SEM images via OCR")
    parser.add_argument("--images-dir", type=str, default="data/images",
                        help="Directory containing SEM images")
    parser.add_argument("--output", type=str, default="data/magnification_map.json",
                        help="Output JSON file path")
    parser.add_argument("--crop-fraction", type=float, default=0.12,
                        help="Fraction of image height to crop from bottom")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_path = Path(args.output)

    image_files = sorted([
        p for p in images_dir.iterdir()
        if p.suffix.lower() in ['.png', '.jpg', '.jpeg']
    ])

    print(f"Found {len(image_files)} images in {images_dir}")

    mag_map = {}
    failures = []

    for img_path in image_files:
        stem = img_path.stem
        filename_mag = extract_mag_from_filename(stem)
        ocr_mag, ocr_unit = extract_mag_from_ocr(img_path, args.crop_fraction)

        resolved = resolve_magnification(stem, ocr_mag, ocr_unit, filename_mag)

        if resolved is not None:
            mag_map[stem] = round(resolved, 2)
        else:
            failures.append(stem)
            print(f"  ERROR: Could not determine magnification for {stem}")

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(mag_map, f, indent=2, sort_keys=True)

    # Print summary
    print(f"\nResults saved to {output_path}")
    print(f"  Successfully extracted: {len(mag_map)}/{len(image_files)}")

    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for name in failures:
            print(f"    - {name}")

    # Magnification distribution
    mag_counts = Counter(mag_map.values())
    print(f"\nMagnification distribution:")
    for mag in sorted(mag_counts.keys()):
        print(f"  {mag:>6.2f} KX: {mag_counts[mag]:>3d} images")

    # Return exit code based on success
    if len(mag_map) < len(image_files):
        print(f"\nWARNING: {len(image_files) - len(mag_map)} images without magnification!")
        sys.exit(1)
    else:
        print(f"\nAll {len(image_files)} images have magnification values.")
        sys.exit(0)


if __name__ == "__main__":
    main()
