"""Investigate overlap between notebook and pipeline train/val splits.

Key question: when we evaluated the notebook checkpoint on the pipeline's seed=42
val split and got 0.93 mIoU — were those val images actually in the notebook's
TRAINING set? That would mean the model memorized them.
"""

import os
import sys

import torch
from torch.utils.data import random_split, Dataset
from PIL import Image
import numpy as np

# Reproduce the notebook's dataset class exactly
class NotebookSEMDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        # sorted, matching our reproduction script Test 1
        self.image_list = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.endswith('.png') or f.endswith('.jpg')
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        return self.image_list[idx]


DATA_DIR = "/home/mis60/CSE_MSE_RXF131/cradle-members/mds3/mis60/mds3-advman-2/25-mds3-data-segmentation/data/sem"

ds = NotebookSEMDataset(
    img_dir=os.path.join(DATA_DIR, "images"),
    label_dir=os.path.join(DATA_DIR, "labels"),
)

all_files = [os.path.basename(ds.image_list[i]) for i in range(len(ds))]
print(f"Total images: {len(ds)}")

train_size = int(0.85 * len(ds))
val_size = len(ds) - train_size

# ============================================================
# Notebook split (no seed) — this depends on PyTorch's RNG state
# We can't reproduce the EXACT notebook split since we don't know
# the RNG state. But Test 1 in reproduce_notebook.py DID reproduce
# the metrics (0.9501 vs 0.9496), which means the model performs
# ~0.95 on ANY 13-image subset, not just a specific one.
#
# Let's verify by testing many random splits.
# ============================================================

# Pipeline split (seed=42)
_, pipe_val = random_split(ds, [train_size, val_size],
                           generator=torch.Generator().manual_seed(42))
pipe_val_files = set(os.path.basename(ds.image_list[i]) for i in pipe_val.indices)
pipe_train_indices = set(range(len(ds))) - set(pipe_val.indices)
pipe_train_files = set(os.path.basename(ds.image_list[i]) for i in pipe_train_indices)

print(f"\nPipeline (seed=42) val files ({len(pipe_val_files)}):")
for f in sorted(pipe_val_files):
    print(f"  {f}")

# Try many no-seed splits to see what the notebook COULD have used
print(f"\n{'='*80}")
print("Testing 100 random no-seed splits to understand variance:")
print(f"{'='*80}")

overlaps_with_pipe_train = []
for i in range(100):
    _, test_val = random_split(ds, [train_size, val_size])
    test_val_files = set(os.path.basename(ds.image_list[j]) for j in test_val.indices)
    test_train_files = set(all_files) - test_val_files

    # How many of pipeline's val images were in this random split's training set?
    overlap = pipe_val_files & test_train_files
    overlaps_with_pipe_train.append(len(overlap))

print(f"\nOf the 13 pipeline val images, how many land in a random split's TRAIN set?")
print(f"  Min: {min(overlaps_with_pipe_train)}")
print(f"  Max: {max(overlaps_with_pipe_train)}")
print(f"  Mean: {np.mean(overlaps_with_pipe_train):.1f}")
print(f"  (If high, it means the notebook model likely TRAINED on most of pipeline's val images)")

# Now check: the notebook split from Test 1 had these val indices:
# [11, 16, 24, 30, 33, 35, 36, 47, 48, 59, 62, 78, 82]
nb_val_indices = [11, 16, 24, 30, 33, 35, 36, 47, 48, 59, 62, 78, 82]
nb_val_files = set(os.path.basename(ds.image_list[i]) for i in nb_val_indices)
nb_train_files = set(all_files) - nb_val_files

print(f"\n{'='*80}")
print("Notebook (Test 1) split analysis:")
print(f"{'='*80}")
print(f"Notebook val files: {sorted(nb_val_files)}")
print(f"Pipeline val files: {sorted(pipe_val_files)}")

common_val = nb_val_files & pipe_val_files
print(f"\nImages in BOTH val sets: {len(common_val)}")
for f in sorted(common_val):
    print(f"  {f}")

pipe_val_in_nb_train = pipe_val_files & nb_train_files
print(f"\nPipeline val images that were in notebook's TRAINING set: {len(pipe_val_in_nb_train)}/13")
for f in sorted(pipe_val_in_nb_train):
    print(f"  {f}")

nb_val_in_pipe_train = nb_val_files & pipe_train_files
print(f"\nNotebook val images that were in pipeline's TRAINING set: {len(nb_val_in_pipe_train)}/13")
for f in sorted(nb_val_in_pipe_train):
    print(f"  {f}")

print(f"\n{'='*80}")
print("CONCLUSION:")
print(f"{'='*80}")
print(f"The notebook model trained on {len(pipe_val_in_nb_train)}/13 of the pipeline's val images.")
print(f"So when we evaluated the notebook checkpoint on the pipeline's val split (Test 2)")
print(f"and got mIoU=0.93, {len(pipe_val_in_nb_train)} of those 13 'val' images were images")
print(f"the model had memorized during training.")
print(f"")
print(f"Similarly, {len(nb_val_in_pipe_train)}/13 of the notebook's own val images were in")
print(f"the pipeline's training set.")
print(f"")
print(f"With 85/15 split on 83 images, ANY random split will have ~85% of another")
print(f"split's val images in its training set. This is expected and means cross-split")
print(f"evaluation is measuring memorization, not generalization.")
