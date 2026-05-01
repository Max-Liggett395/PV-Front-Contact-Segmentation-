"""Reproduce the notebook's results and systematically identify the cause of the gap.

Tests every variable:
1. Notebook checkpoint vs pipeline checkpoint
2. Notebook model class (LazyConv2d) vs pipeline model class
3. Raw [0-255] pixels vs normalized [-1,1] pixels
4. Per-image metrics (sklearn, like notebook) vs global metrics (like pipeline)
5. Different random splits (no seed vs seed=42)
6. Notebook data path vs pipeline data path
"""

import os
import sys
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from sklearn.metrics import jaccard_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.unet import UNet as PipelineUNet
from src.evaluation.metrics import compute_metrics

# ============================================================
# Notebook's exact UNet (LazyConv2d version, copied from cell-0)
# ============================================================
class NotebookUNet(nn.Module):
    def __init__(self):
        super(NotebookUNet, self).__init__()
        self.enc1 = nn.LazyConv2d(64, 3, 1, 1)
        self.enc1b = nn.LazyConv2d(64, 3, 1, 1)
        self.enc2 = nn.LazyConv2d(128, 3, 1, 1)
        self.enc2b = nn.LazyConv2d(128, 3, 1, 1)
        self.enc3 = nn.LazyConv2d(256, 3, 1, 1)
        self.enc3b = nn.LazyConv2d(256, 3, 1, 1)
        self.enc4 = nn.LazyConv2d(512, 3, 1, 1)
        self.enc4b = nn.LazyConv2d(512, 3, 1, 1)
        self.dropout = nn.Dropout(p=0.3)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.enc5 = nn.LazyConv2d(1024, 3, 1, 1)
        self.enc5b = nn.LazyConv2d(1024, 3, 1, 1)
        self.dec1 = nn.LazyConvTranspose2d(512, 2, 2, 0)
        self.dec1b = nn.LazyConvTranspose2d(512, 2, 2, 0)
        self.dec2 = nn.LazyConvTranspose2d(256, 2, 2, 0)
        self.dec2b = nn.LazyConvTranspose2d(256, 2, 2, 0)
        self.dec3 = nn.LazyConvTranspose2d(128, 2, 2, 0)
        self.dec3b = nn.LazyConvTranspose2d(128, 2, 2, 0)
        self.dec4 = nn.LazyConvTranspose2d(64, 2, 2, 0)
        self.dec4b = nn.LazyConvTranspose2d(64, 2, 2, 0)
        self.conv1a = nn.LazyConv2d(64, 3, 1, 1)
        self.conv1b = nn.LazyConv2d(64, 3, 1, 1)
        self.conv2a = nn.LazyConv2d(128, 3, 1, 1)
        self.conv2b = nn.LazyConv2d(128, 3, 1, 1)
        self.conv3a = nn.LazyConv2d(256, 3, 1, 1)
        self.conv3b = nn.LazyConv2d(256, 3, 1, 1)
        self.conv4a = nn.LazyConv2d(512, 3, 1, 1)
        self.conv4b = nn.LazyConv2d(512, 3, 1, 1)
        self.conv5a = nn.LazyConv2d(1024, 3, 1, 1)
        self.conv5b = nn.LazyConv2d(1024, 3, 1, 1)
        self.out = nn.LazyConv2d(6, 1, 1, 0)
        self.forward = self.build_unet_model

    def double_conv_block1(self, x):
        return F.relu(self.enc1b(F.relu(self.enc1(x))))
    def double_conv_block2(self, x):
        return F.relu(self.enc2b(F.relu(self.enc2(x))))
    def double_conv_block3(self, x):
        return F.relu(self.enc3b(F.relu(self.enc3(x))))
    def double_conv_block4(self, x):
        return F.relu(self.enc4b(F.relu(self.enc4(x))))
    def double_conv_block5(self, x):
        return F.relu(self.enc5b(F.relu(self.enc5(x))))

    def up_double_conv_block1(self, x):
        return F.relu(self.conv5b(F.relu(self.conv5a(x))))
    def up_double_conv_block2(self, x):
        return F.relu(self.conv4b(F.relu(self.conv4a(x))))
    def up_double_conv_block3(self, x):
        return F.relu(self.conv3b(F.relu(self.conv3a(x))))
    def up_double_conv_block4(self, x):
        return F.relu(self.conv2b(F.relu(self.conv2a(x))))

    def downsample_block(self, x, conv_fn):
        f = conv_fn(x)
        p = self.dropout(self.max_pool(f))
        return f, p

    def upsample_block(self, x, conv_features, dec_fn, up_conv_fn):
        x = dec_fn(x)
        diffY = conv_features.size()[2] - x.size()[2]
        diffX = conv_features.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, conv_features], dim=1)
        x = self.dropout(x)
        x = up_conv_fn(x)
        return x

    def build_unet_model(self, input):
        f1, p1 = self.downsample_block(input, self.double_conv_block1)
        f2, p2 = self.downsample_block(p1, self.double_conv_block2)
        f3, p3 = self.downsample_block(p2, self.double_conv_block3)
        f4, p4 = self.downsample_block(p3, self.double_conv_block4)
        bottleneck = self.double_conv_block5(p4)
        u6 = self.upsample_block(bottleneck, f4, self.dec1, self.up_double_conv_block1)
        u7 = self.upsample_block(u6, f3, self.dec2, self.up_double_conv_block2)
        u8 = self.upsample_block(u7, f2, self.dec3, self.up_double_conv_block3)
        u9 = self.upsample_block(u8, f1, self.dec4, self.up_double_conv_block4)
        return self.out(u9)


# ============================================================
# Notebook's exact dataset (copied from cell-2)
# ============================================================
class NotebookSEMDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_list = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.endswith('.png') or f.endswith('.jpg')
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        label_name = os.path.join(
            self.label_dir,
            os.path.splitext(os.path.basename(img_path))[0] + '.npy'
        )
        image = Image.open(img_path).convert("L").resize((1024, 768), Image.BILINEAR)
        label = np.load(label_name)
        label = Image.fromarray(label).resize((1024, 768), Image.NEAREST)
        image = np.array(image)
        label = np.array(label)

        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image, label = augmented["image"], augmented["mask"]

        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        return image, label


# ============================================================
# Evaluation functions
# ============================================================
def eval_sklearn_per_image(model, dataloader, device):
    """Notebook-style: per-image sklearn metrics, then average."""
    model.eval()
    iou_scores, f1_scores, pixel_accs = [], [], []

    with torch.no_grad():
        for image, label in dataloader:
            image = image.to(device)
            output = model(image)
            predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            label_np = label.squeeze(0).cpu().numpy().astype(np.uint8)

            iou = jaccard_score(label_np.flatten(), predicted_mask.flatten(), average="macro")
            f1 = f1_score(label_np.flatten(), predicted_mask.flatten(), average="macro")
            pixel_acc = (predicted_mask == label_np).sum() / label_np.size

            iou_scores.append(iou)
            f1_scores.append(f1)
            pixel_accs.append(pixel_acc)

    return {
        "sklearn_img_miou": np.mean(iou_scores),
        "sklearn_img_f1": np.mean(f1_scores),
        "sklearn_img_acc": np.mean(pixel_accs),
    }


def eval_global(model, dataloader, device, num_classes=6):
    """Pipeline-style: global pooled metrics."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_targets.append(masks)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    return compute_metrics(all_preds, all_targets, num_classes)


# ============================================================
# Main
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    NOTEBOOK_DATA = "/home/mis60/CSE_MSE_RXF131/cradle-members/mds3/mis60/mds3-advman-2/25-mds3-data-segmentation/data/sem"
    NOTEBOOK_CKPT = "/home/mis60/CSE_MSE_RXF131/cradle-members/mds3/mis60/mds3-advman-2/25-mds3-data-segmentation/models/sem/unet-83/checkpoints/best_model.pth"
    PIPELINE_CKPT = "/home/mis60/sem/logs/runs/unet-d83/checkpoints/best.pt"
    PIPELINE_DATA_IMG = "/home/mis60/sem/data/images"
    PIPELINE_DATA_MASK = "/home/mis60/sem/data/labels"

    # --------------------------------------------------------
    # TEST 1: Reproduce notebook exactly
    #   Notebook model + notebook checkpoint + notebook data + no transform + no seed
    # --------------------------------------------------------
    print("=" * 80)
    print("TEST 1: Exact notebook reproduction")
    print("  Model: NotebookUNet (LazyConv2d)")
    print("  Checkpoint: notebook best_model.pth")
    print("  Data: notebook path, no seed, no transform")
    print("=" * 80)

    full_ds = NotebookSEMDataset(
        img_dir=os.path.join(NOTEBOOK_DATA, "images"),
        label_dir=os.path.join(NOTEBOOK_DATA, "labels"),
        transform=None,
    )
    print(f"  Dataset size: {len(full_ds)}")

    # Notebook split: 85/15, NO seed
    train_size = int(0.85 * len(full_ds))
    val_size = len(full_ds) - train_size
    _, val_ds = random_split(full_ds, [train_size, val_size])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # Load notebook model
    nb_model = NotebookUNet().to(device)
    # Initialize lazy modules with a dummy forward pass
    with torch.no_grad():
        dummy = torch.randn(1, 1, 768, 1024).to(device)
        _ = nb_model(dummy)
    nb_model.load_state_dict(torch.load(NOTEBOOK_CKPT, map_location=device, weights_only=False))
    nb_model.eval()

    sk_metrics = eval_sklearn_per_image(nb_model, val_loader, device)
    gl_metrics = eval_global(nb_model, val_loader, device)
    print(f"\n  sklearn per-image:  mIoU={sk_metrics['sklearn_img_miou']:.4f}  F1={sk_metrics['sklearn_img_f1']:.4f}  Acc={sk_metrics['sklearn_img_acc']:.4f}")
    print(f"  global pooled:      mIoU={gl_metrics['miou']:.4f}  F1={gl_metrics['f1_macro']:.4f}  Acc={gl_metrics['pixel_accuracy']:.4f}")
    print(f"  pipeline per-image: mIoU={gl_metrics['img_miou']:.4f}  F1={gl_metrics['img_f1_macro']:.4f}  Acc={gl_metrics['img_pixel_accuracy']:.4f}")

    # Get val indices for inspection
    val_indices = val_ds.indices
    val_filenames = [os.path.basename(full_ds.image_list[i]) for i in val_indices]
    print(f"\n  Val split indices: {sorted(val_indices)}")
    print(f"  Val filenames: {sorted(val_filenames)}")

    # --------------------------------------------------------
    # TEST 2: Notebook checkpoint + seed=42 split (pipeline's split)
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST 2: Notebook checkpoint on pipeline's val split (seed=42)")
    print("=" * 80)

    _, val_ds_seeded = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    val_loader_seeded = DataLoader(val_ds_seeded, batch_size=1, shuffle=False)

    val_indices_seeded = val_ds_seeded.indices
    val_filenames_seeded = [os.path.basename(full_ds.image_list[i]) for i in val_indices_seeded]
    print(f"  Val split indices: {sorted(val_indices_seeded)}")
    print(f"  Val filenames: {sorted(val_filenames_seeded)}")
    print(f"  Overlap with Test 1: {len(set(val_indices) & set(val_indices_seeded))} / {val_size} images")

    sk_metrics2 = eval_sklearn_per_image(nb_model, val_loader_seeded, device)
    gl_metrics2 = eval_global(nb_model, val_loader_seeded, device)
    print(f"\n  sklearn per-image:  mIoU={sk_metrics2['sklearn_img_miou']:.4f}  F1={sk_metrics2['sklearn_img_f1']:.4f}  Acc={sk_metrics2['sklearn_img_acc']:.4f}")
    print(f"  global pooled:      mIoU={gl_metrics2['miou']:.4f}  F1={gl_metrics2['f1_macro']:.4f}  Acc={gl_metrics2['pixel_accuracy']:.4f}")
    print(f"  pipeline per-image: mIoU={gl_metrics2['img_miou']:.4f}  F1={gl_metrics2['img_f1_macro']:.4f}  Acc={gl_metrics2['img_pixel_accuracy']:.4f}")

    # --------------------------------------------------------
    # TEST 3: Notebook checkpoint on ALL 83 images (train+val)
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST 3: Notebook checkpoint on ALL 83 images (is it memorization?)")
    print("=" * 80)

    all_loader = DataLoader(full_ds, batch_size=1, shuffle=False)
    sk_metrics3 = eval_sklearn_per_image(nb_model, all_loader, device)
    gl_metrics3 = eval_global(nb_model, all_loader, device)
    print(f"\n  sklearn per-image:  mIoU={sk_metrics3['sklearn_img_miou']:.4f}  F1={sk_metrics3['sklearn_img_f1']:.4f}  Acc={sk_metrics3['sklearn_img_acc']:.4f}")
    print(f"  global pooled:      mIoU={gl_metrics3['miou']:.4f}  F1={gl_metrics3['f1_macro']:.4f}  Acc={gl_metrics3['pixel_accuracy']:.4f}")
    print(f"  pipeline per-image: mIoU={gl_metrics3['img_miou']:.4f}  F1={gl_metrics3['img_f1_macro']:.4f}  Acc={gl_metrics3['img_pixel_accuracy']:.4f}")

    # --------------------------------------------------------
    # TEST 4: Pipeline checkpoint (seed=42 split) on raw [0-255] pixels
    #   To isolate: is the gap from the checkpoint or from normalization?
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST 4: Pipeline checkpoint on raw [0-255] pixels (seed=42 split)")
    print("  (To test if normalization matters)")
    print("=" * 80)

    pipe_model = PipelineUNet(in_channels=1, num_classes=6, dropout=0.3).to(device)
    pipe_ckpt = torch.load(PIPELINE_CKPT, map_location=device, weights_only=False)
    pipe_model.load_state_dict(pipe_ckpt["model_state_dict"])
    pipe_model.eval()

    # Pipeline's val split on raw data (no normalize)
    pipe_full_ds = NotebookSEMDataset(
        img_dir=PIPELINE_DATA_IMG,
        label_dir=PIPELINE_DATA_MASK,
        transform=None,
    )
    pipe_train_size = int(0.85 * len(pipe_full_ds))
    pipe_val_size = len(pipe_full_ds) - pipe_train_size
    _, pipe_val_raw = random_split(
        pipe_full_ds, [pipe_train_size, pipe_val_size],
        generator=torch.Generator().manual_seed(42),
    )
    pipe_raw_loader = DataLoader(pipe_val_raw, batch_size=1, shuffle=False)

    sk_metrics4 = eval_sklearn_per_image(pipe_model, pipe_raw_loader, device)
    gl_metrics4 = eval_global(pipe_model, pipe_raw_loader, device)
    print(f"\n  sklearn per-image:  mIoU={sk_metrics4['sklearn_img_miou']:.4f}  F1={sk_metrics4['sklearn_img_f1']:.4f}  Acc={sk_metrics4['sklearn_img_acc']:.4f}")
    print(f"  global pooled:      mIoU={gl_metrics4['miou']:.4f}  F1={gl_metrics4['f1_macro']:.4f}  Acc={gl_metrics4['pixel_accuracy']:.4f}")
    print(f"  pipeline per-image: mIoU={gl_metrics4['img_miou']:.4f}  F1={gl_metrics4['img_f1_macro']:.4f}  Acc={gl_metrics4['img_pixel_accuracy']:.4f}")

    # --------------------------------------------------------
    # TEST 5: Pipeline checkpoint on normalized data (seed=42) — baseline
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST 5: Pipeline checkpoint on normalized data (seed=42) — baseline")
    print("=" * 80)

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    val_transform = A.Compose([
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])
    pipe_full_ds_norm = NotebookSEMDataset(
        img_dir=PIPELINE_DATA_IMG,
        label_dir=PIPELINE_DATA_MASK,
        transform=val_transform,
    )
    _, pipe_val_norm = random_split(
        pipe_full_ds_norm, [pipe_train_size, pipe_val_size],
        generator=torch.Generator().manual_seed(42),
    )
    pipe_norm_loader = DataLoader(pipe_val_norm, batch_size=1, shuffle=False)

    sk_metrics5 = eval_sklearn_per_image(pipe_model, pipe_norm_loader, device)
    gl_metrics5 = eval_global(pipe_model, pipe_norm_loader, device)
    print(f"\n  sklearn per-image:  mIoU={sk_metrics5['sklearn_img_miou']:.4f}  F1={sk_metrics5['sklearn_img_f1']:.4f}  Acc={sk_metrics5['sklearn_img_acc']:.4f}")
    print(f"  global pooled:      mIoU={gl_metrics5['miou']:.4f}  F1={gl_metrics5['f1_macro']:.4f}  Acc={gl_metrics5['pixel_accuracy']:.4f}")
    print(f"  pipeline per-image: mIoU={gl_metrics5['img_miou']:.4f}  F1={gl_metrics5['img_f1_macro']:.4f}  Acc={gl_metrics5['img_pixel_accuracy']:.4f}")

    # --------------------------------------------------------
    # TEST 6: Notebook checkpoint on notebook data with SORTED file list
    #   The notebook uses os.listdir (unsorted). Our NotebookSEMDataset sorts.
    #   This could change which indices map to which files after random_split.
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST 6: Notebook checkpoint, UNSORTED file list (exact os.listdir order)")
    print("=" * 80)

    class UnsortedSEMDataset(NotebookSEMDataset):
        def __init__(self, img_dir, label_dir, transform=None):
            self.img_dir = img_dir
            self.label_dir = label_dir
            # Use os.listdir order — NOT sorted
            self.image_list = [
                os.path.join(img_dir, f)
                for f in os.listdir(img_dir)
                if f.endswith('.png') or f.endswith('.jpg')
            ]
            self.transform = transform

    unsorted_ds = UnsortedSEMDataset(
        img_dir=os.path.join(NOTEBOOK_DATA, "images"),
        label_dir=os.path.join(NOTEBOOK_DATA, "labels"),
        transform=None,
    )
    _, unsorted_val = random_split(unsorted_ds, [train_size, val_size])
    unsorted_val_loader = DataLoader(unsorted_val, batch_size=1, shuffle=False)

    unsorted_val_filenames = [os.path.basename(unsorted_ds.image_list[i]) for i in unsorted_val.indices]
    print(f"  Val filenames (unsorted listdir): {sorted(unsorted_val_filenames)}")
    print(f"  Same as Test 1? {sorted(unsorted_val_filenames) == sorted(val_filenames)}")

    sk_metrics6 = eval_sklearn_per_image(nb_model, unsorted_val_loader, device)
    gl_metrics6 = eval_global(nb_model, unsorted_val_loader, device)
    print(f"\n  sklearn per-image:  mIoU={sk_metrics6['sklearn_img_miou']:.4f}  F1={sk_metrics6['sklearn_img_f1']:.4f}  Acc={sk_metrics6['sklearn_img_acc']:.4f}")
    print(f"  global pooled:      mIoU={gl_metrics6['miou']:.4f}  F1={gl_metrics6['f1_macro']:.4f}  Acc={gl_metrics6['pixel_accuracy']:.4f}")

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------
    print("\n\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'Test':<55} {'sk_mIoU':>8} {'sk_F1':>8} {'gl_mIoU':>8} {'gl_F1':>8}")
    print("-" * 100)

    all_results = [
        ("1: NB ckpt + NB data + no seed + raw", sk_metrics, gl_metrics),
        ("2: NB ckpt + NB data + seed=42 + raw", sk_metrics2, gl_metrics2),
        ("3: NB ckpt + ALL 83 images + raw", sk_metrics3, gl_metrics3),
        ("4: Pipeline ckpt + seed=42 + raw [0-255]", sk_metrics4, gl_metrics4),
        ("5: Pipeline ckpt + seed=42 + normalized", sk_metrics5, gl_metrics5),
        ("6: NB ckpt + unsorted listdir + no seed + raw", sk_metrics6, gl_metrics6),
    ]

    for label, sk, gl in all_results:
        print(f"{label:<55} {sk['sklearn_img_miou']:>8.4f} {sk['sklearn_img_f1']:>8.4f} {gl['miou']:>8.4f} {gl['f1_macro']:>8.4f}")

    print("=" * 100)
    print("\nNotebook reported:                                      0.9496   0.9732")
    print("Pipeline reported (unet-d83, global):                                       0.6911   0.8064")


if __name__ == "__main__":
    main()
