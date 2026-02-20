# SEM Dataset Report

## Overview

| Property | Value |
|----------|-------|
| Total images | 130 |
| Total label files | 199 (116 JSON + 83 NPY) |
| Images size on disk | 70 MB |
| Labels size on disk | 66 MB |
| Total size on disk | 136 MB |

## Data Sources

### Source 1: New Dataset (from `old-sem`)

- **Images:** 116 (PNG, JPG)
- **Labels:** 116 VIA JSON files (polygon/polyline annotations)
- **Label format:** VIA JSON (VGG Image Annotator), keyed by `{filename}{filesize}`

### Source 2: Old Dataset (from `images.zip` / `labels.zip`)

- **Images:** 61 PNG files (47 overlap with Source 1, 14 unique)
- **Labels:** 83 NPY files (semantic segmentation masks)
- **Label format:** NumPy arrays, shape `(768, 1024)`, 6 integer classes (0–5)

### Overlap

| Category | Count |
|----------|-------|
| Images with both JSON and NPY labels | 69 |
| Images with JSON label only | 47 |
| Images with NPY label only | 14 |
| Images with no label | 0 |
| Labels with no image | 0 |

## Images

- **Formats:** PNG (108), JPG (22)
- **Resolutions:** ~1024x768, with slight variation (1024x768 to 1030x776)
- **Naming conventions:**
  - Source 1: `{date or sample}_{cellID}_{treatment}_{magnification}_{index}.{ext}`
  - Source 2: `{manufacturer}.{cellID}.{wafer}.{index}.png`

## Annotations

### Source 1 — VIA JSON (polygon regions)

#### Class Distribution

| Class | Region Count | % of Total |
|-------|-------------|------------|
| void | 5,299 | 69.4% |
| glass | 1,001 | 13.1% |
| interfacial void | 740 | 9.7% |
| silver | 466 | 6.1% |
| silicon | 127 | 1.7% |
| **Total** | **7,633** | **100%** |

#### Shape Types

| Shape | Count |
|-------|-------|
| polygon | 7,366 |
| polyline | 265 |
| circle | 2 |

#### Regions Per Image

| Statistic | Value |
|-----------|-------|
| Min | 8 |
| Max | 366 |
| Mean | 65.8 |

### Source 2 — NPY Masks (semantic segmentation)

Integer-valued masks with 6 classes (0–5). Each mask is 768x1024.

#### Class Pixel Distribution

| Class ID | Pixel Count | % of Total |
|----------|-------------|------------|
| 0 | 17,404,992 | 26.7% |
| 1 | 16,581,995 | 25.4% |
| 2 | 2,324,466 | 3.6% |
| 3 | 21,677,542 | 33.2% |
| 4 | 3,433,028 | 5.3% |
| 5 | 3,851,833 | 5.9% |

#### Class ID Mapping

Established by cross-referencing the 69 overlapping images:

| Class ID | Label | Confidence |
|----------|-------|------------|
| 0 | background | — |
| 1 | silver | 84.0% |
| 2 | glass | 91.7% |
| 3 | silicon | 99.8% |
| 4 | void | 98.0% |
| 5 | interfacial void | 98.3% |

## Merged Dataset

**Location:** `data/merged/`

Unified dataset with all 130 images and NPY semantic segmentation masks.

| Property | Value |
|----------|-------|
| Total images | 130 |
| Total masks | 130 (all NPY) |
| Mask shape | (768, 1024) |
| Classes | 6 (0–5) |
| Size on disk | 168 MB |

### Construction

- 83 masks taken directly from Source 2 (existing NPY files)
- 47 masks converted from Source 1 JSON polygons using the class ID mapping above
- Rendering order: silicon → silver → glass → void → interfacial void (later layers overwrite earlier)

## Notes

- Source 1 annotations are heavily imbalanced toward `void` (69.4% of regions).
- Source 2 mask classes are more evenly distributed, with silicon being the largest (33.2%).
- Image resolutions are near-uniform (~1024x768) but not pixel-exact in Source 1.
- The silver class mapping has lower confidence (84.0%) due to void regions often overlapping silver areas (15.2% of silver polygon pixels mapped to class 4/void in the NPY masks).
