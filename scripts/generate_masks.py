#!/usr/bin/env python3
"""Generate segmentation masks from VIA JSON annotation files."""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import argparse


# Class mapping from annotation labels to numeric values
# Ordered by pixel prevalence: silver > silicon > background > void > interfacial void > glass
CLASS_MAP = {
    "silver": 0,
    "ag": 0,
    "silicon": 1,
    "si": 1,
    "background": 2,
    "void": 3,
    "interfacial void": 4,
    "interfacial_void": 4,
    "glass": 5,
}

# Drawing priority: lower values drawn first, higher values drawn on top
# Hierarchy: background < silver < silicon < glass < void < interfacial void
DRAW_PRIORITY = {
    2: 0,  # background
    0: 1,  # silver
    1: 2,  # silicon
    5: 3,  # glass
    3: 4,  # void
    4: 5,  # interfacial void
}


def parse_via_json(json_path: Path) -> dict:
    """Parse VIA JSON annotation file.

    Args:
        json_path: Path to JSON file

    Returns:
        Dictionary with filename and list of regions
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # VIA format has a single key that combines filename and size
    key = list(data.keys())[0]
    annotation = data[key]

    return {
        'filename': annotation['filename'],
        'regions': annotation.get('regions', [])
    }


def get_class_from_attributes(region_attrs: dict) -> str:
    """Extract class label from region attributes.

    Handles multiple VIA annotation formats:
    - {'element': 'void'}
    - {'type': 'void'}
    - {'void': 'glass'} - key is 'void' but value is the actual class
    """
    # Try 'element' key first
    if 'element' in region_attrs:
        return region_attrs['element'].lower().strip()

    # Try 'type' key
    if 'type' in region_attrs:
        return region_attrs['type'].lower().strip()

    # Handle format where key is 'void' but value is the actual class
    if 'void' in region_attrs:
        value = region_attrs['void'].lower().strip()
        # If value is a known class, use it; otherwise the class is 'void'
        if value in CLASS_MAP:
            return value
        return 'void'

    # Check if any key itself is a class name
    for key in region_attrs:
        if key.lower().strip() in CLASS_MAP:
            return key.lower().strip()

    return ''


def create_mask_from_regions(regions: list, image_shape: tuple) -> np.ndarray:
    """Create segmentation mask from polygon regions.

    Args:
        regions: List of region dictionaries from VIA JSON
        image_shape: (height, width) of the image

    Returns:
        Numpy array mask with class indices
    """
    height, width = image_shape
    # Initialize with background class (2)
    mask = np.full((height, width), CLASS_MAP["background"], dtype=np.uint8)

    # Sort regions by draw priority so voids are drawn on top of base materials
    def get_region_priority(region):
        region_attrs = region.get('region_attributes', {})
        element = get_class_from_attributes(region_attrs)
        if element and element in CLASS_MAP:
            class_idx = CLASS_MAP[element]
            return DRAW_PRIORITY.get(class_idx, 0)
        return 0

    sorted_regions = sorted(regions, key=get_region_priority)

    for region in sorted_regions:
        shape_attrs = region.get('shape_attributes', {})
        region_attrs = region.get('region_attributes', {})

        # Get class label using flexible parsing
        element = get_class_from_attributes(region_attrs)
        if not element or element not in CLASS_MAP:
            continue

        class_idx = CLASS_MAP[element]

        # Get polygon points
        shape_name = shape_attrs.get('name', '')

        if shape_name in ['polygon', 'polyline']:
            points_x = shape_attrs.get('all_points_x', [])
            points_y = shape_attrs.get('all_points_y', [])

            if len(points_x) < 3 or len(points_y) < 3:
                continue

            # Create polygon points array
            points = np.array(list(zip(points_x, points_y)), dtype=np.int32)

            # Fill polygon on mask
            cv2.fillPoly(mask, [points], class_idx)

        elif shape_name == 'rect':
            x = int(shape_attrs.get('x', 0))
            y = int(shape_attrs.get('y', 0))
            w = int(shape_attrs.get('width', 0))
            h = int(shape_attrs.get('height', 0))
            mask[y:y+h, x:x+w] = class_idx

        elif shape_name == 'circle':
            cx = int(shape_attrs.get('cx', 0))
            cy = int(shape_attrs.get('cy', 0))
            r = int(shape_attrs.get('r', 0))
            cv2.circle(mask, (cx, cy), r, class_idx, -1)

    return mask


def generate_masks(images_dir: Path, labels_dir: Path, output_dir: Path, visualize: bool = False):
    """Generate masks for all images.

    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing JSON label files
        output_dir: Directory to save generated masks
        visualize: If True, also save visualization PNGs
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectory for .npy files
    npy_dir = output_dir / 'npy'
    npy_dir.mkdir(exist_ok=True)

    if visualize:
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)

    # Get all JSON files
    json_files = sorted(labels_dir.glob('*.json'))
    print(f"Found {len(json_files)} JSON label files")

    # Class colors for visualization (BGR for OpenCV)
    class_colors = {
        0: (0, 255, 0),      # Silver - Green
        1: (0, 255, 255),    # Silicon - Yellow
        2: (0, 0, 255),      # Background - Red
        3: (255, 0, 255),    # Void - Magenta
        4: (255, 255, 0),    # Interfacial Void - Cyan
        5: (255, 0, 0),      # Glass - Blue
    }

    processed = 0
    skipped = 0

    for json_path in json_files:
        # Parse JSON
        annotation = parse_via_json(json_path)
        image_filename = annotation['filename']

        # Find corresponding image (handle case sensitivity)
        image_path = None
        for ext in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']:
            candidate = images_dir / (json_path.stem + ext)
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            print(f"Warning: No image found for {json_path.name}, skipping")
            skipped += 1
            continue

        # Load image to get dimensions
        image = Image.open(image_path)
        width, height = image.size

        # Create mask
        mask = create_mask_from_regions(annotation['regions'], (height, width))

        # Save mask as .npy in npy subdirectory
        mask_filename = json_path.stem + '.npy'
        np.save(npy_dir / mask_filename, mask)

        # Create visualization if requested
        if visualize:
            # Create colored mask
            viz = np.zeros((height, width, 3), dtype=np.uint8)
            for class_idx, color in class_colors.items():
                viz[mask == class_idx] = color

            # Blend with original image
            image_array = np.array(image.convert('RGB'))
            blended = cv2.addWeighted(image_array, 0.5, viz, 0.5, 0)

            viz_filename = json_path.stem + '_mask.png'
            cv2.imwrite(str(viz_dir / viz_filename), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

        processed += 1
        if processed % 20 == 0:
            print(f"Processed {processed}/{len(json_files)} masks...")

    print(f"\nDone! Processed {processed} masks, skipped {skipped}")
    print(f"Masks (.npy) saved to: {npy_dir}")
    if visualize:
        print(f"Visualizations saved to: {viz_dir}")

    # Print class distribution
    print("\nClass mapping (by prevalence):")
    print("  0: Silver (Ag)")
    print("  1: Silicon (Si)")
    print("  2: Background (unlabeled)")
    print("  3: Void")
    print("  4: Interfacial Void")
    print("  5: Glass")


def main():
    parser = argparse.ArgumentParser(description='Generate segmentation masks from VIA JSON annotations')
    parser.add_argument('--images', type=str, default='data/images', help='Path to images directory')
    parser.add_argument('--labels', type=str, default='data/labels', help='Path to JSON labels directory')
    parser.add_argument('--output', type=str, default='data/masks', help='Path to output masks directory')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization images')

    args = parser.parse_args()

    # Convert to Path objects
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    output_dir = Path(args.output)

    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return

    if not labels_dir.exists():
        print(f"Error: Labels directory not found: {labels_dir}")
        return

    generate_masks(images_dir, labels_dir, output_dir, args.visualize)


if __name__ == '__main__':
    main()
