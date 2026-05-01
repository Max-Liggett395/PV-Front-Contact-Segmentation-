"""Final comparison: notebook checkpoint vs pipeline raw checkpoint on same data."""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from sklearn.metrics import jaccard_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.unet import UNet as PipelineUNet
from src.evaluation.metrics import compute_metrics


from scripts.reproduce_notebook import NotebookUNet


class RawDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_list = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.endswith('.png') or f.endswith('.jpg')
        ])

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
        image = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(np.array(label), dtype=torch.long)
        return image, label


def evaluate(model, loader, device, label=""):
    model.eval()
    all_preds, all_targets = [], []
    sk_ious, sk_f1s = [], []

    with torch.no_grad():
        for img, lbl in loader:
            img = img.to(device)
            out = model(img)
            pred = out.argmax(dim=1).cpu()
            all_preds.append(pred)
            all_targets.append(lbl)

            pred_np = pred.squeeze(0).numpy().astype(np.uint8)
            lbl_np = lbl.squeeze(0).numpy().astype(np.uint8)
            sk_ious.append(jaccard_score(lbl_np.flatten(), pred_np.flatten(), average="macro"))
            sk_f1s.append(f1_score(lbl_np.flatten(), pred_np.flatten(), average="macro"))

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    gl = compute_metrics(all_preds, all_targets, 6)

    print(f"\n  {label}")
    print(f"    Global:    mIoU={gl['miou']:.4f}  F1={gl['f1_macro']:.4f}  Acc={gl['pixel_accuracy']:.4f}")
    print(f"    Per-image: mIoU={gl['img_miou']:.4f}  F1={gl['img_f1_macro']:.4f}  Acc={gl['img_pixel_accuracy']:.4f}")
    print(f"    sklearn:   mIoU={np.mean(sk_ious):.4f}  F1={np.mean(sk_f1s):.4f}")
    return gl


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    DATA_IMG = "/home/mis60/sem/data/images"
    DATA_MASK = "/home/mis60/sem/data/labels"

    ds = RawDataset(DATA_IMG, DATA_MASK)
    train_size = int(0.85 * len(ds))
    val_size = len(ds) - train_size

    # ============================================================
    # SPLIT ANALYSIS
    # ============================================================
    print("\n" + "=" * 80)
    print("SPLIT ANALYSIS")
    print("=" * 80)

    # Notebook split (no seed) — indices from reproduction test 1
    nb_val_indices = [11, 16, 24, 30, 33, 35, 36, 47, 48, 59, 62, 78, 82]
    nb_train_indices = [i for i in range(83) if i not in nb_val_indices]

    # Pipeline split (seed=42)
    _, pipe_val = random_split(ds, [train_size, val_size],
                               generator=torch.Generator().manual_seed(42))
    pipe_val_indices = pipe_val.indices

    # Check cell-level overlap
    def get_cell(filename):
        """Extract solar cell ID from filename."""
        parts = os.path.splitext(os.path.basename(filename))[0].split('.')
        if len(parts) >= 2:
            return '.'.join(parts[:2])
        return parts[0]

    nb_val_files = [os.path.basename(ds.image_list[i]) for i in nb_val_indices]
    pipe_val_files = [os.path.basename(ds.image_list[i]) for i in pipe_val_indices]
    all_files = [os.path.basename(ds.image_list[i]) for i in range(83)]

    nb_val_cells = set(get_cell(f) for f in nb_val_files)
    nb_train_cells = set(get_cell(all_files[i]) for i in nb_train_indices)
    pipe_val_cells = set(get_cell(f) for f in pipe_val_files)
    pipe_train_cells = set(get_cell(all_files[i]) for i in range(83) if i not in pipe_val_indices)

    # How many val cells have a match in training?
    nb_val_cells_in_train = nb_val_cells & nb_train_cells
    pipe_val_cells_in_train = pipe_val_cells & pipe_train_cells

    print(f"\nNotebook split:")
    print(f"  Val cells: {sorted(nb_val_cells)}")
    print(f"  Val cells with train cell-mates: {len(nb_val_cells_in_train)}/{len(nb_val_cells)}")
    print(f"  Matched cells: {sorted(nb_val_cells_in_train)}")

    print(f"\nPipeline split (seed=42):")
    print(f"  Val cells: {sorted(pipe_val_cells)}")
    print(f"  Val cells with train cell-mates: {len(pipe_val_cells_in_train)}/{len(pipe_val_cells)}")
    print(f"  Matched cells: {sorted(pipe_val_cells_in_train)}")

    # Unique cells in val (no cell-mate in training)
    nb_unique = nb_val_cells - nb_train_cells
    pipe_unique = pipe_val_cells - pipe_train_cells
    print(f"\n  Notebook val cells with NO training cell-mate: {sorted(nb_unique) if nb_unique else 'NONE'}")
    print(f"  Pipeline val cells with NO training cell-mate: {sorted(pipe_unique) if pipe_unique else 'NONE'}")

    # ============================================================
    # EVALUATE: Notebook checkpoint on different splits
    # ============================================================
    print("\n" + "=" * 80)
    print("NOTEBOOK CHECKPOINT EVALUATION")
    print("=" * 80)

    nb_model = NotebookUNet().to(device)
    with torch.no_grad():
        dummy = torch.randn(1, 1, 768, 1024).to(device)
        _ = nb_model(dummy)
    nb_ckpt = "/home/mis60/CSE_MSE_RXF131/cradle-members/mds3/mis60/mds3-advman-2/25-mds3-data-segmentation/models/sem/unet-83/checkpoints/best_model.pth"
    nb_model.load_state_dict(torch.load(nb_ckpt, map_location=device, weights_only=False))

    # On notebook val split
    _, nb_val = random_split(ds, [train_size, val_size])
    nb_val_loader = DataLoader(nb_val, batch_size=1, shuffle=False)
    evaluate(nb_model, nb_val_loader, device, "NB ckpt → NB val (no seed)")

    # On pipeline val split
    pipe_val_loader = DataLoader(pipe_val, batch_size=1, shuffle=False)
    evaluate(nb_model, pipe_val_loader, device, "NB ckpt → Pipeline val (seed=42)")

    # On ALL data
    all_loader = DataLoader(ds, batch_size=1, shuffle=False)
    evaluate(nb_model, all_loader, device, "NB ckpt → ALL 83 images")

    # On ONLY images the notebook model NEVER trained on
    # (images in pipeline val that were also in notebook val — only 4 images)
    nb_val_set = set(nb_val_indices)
    pipe_val_set = set(pipe_val_indices)
    never_trained = list(nb_val_set & pipe_val_set)  # in both val sets = never in NB training
    print(f"\n  Images NB model never trained on (in both val sets): {len(never_trained)}")
    print(f"  Indices: {never_trained}")
    print(f"  Files: {[os.path.basename(ds.image_list[i]) for i in never_trained]}")

    if never_trained:
        class SubsetByIndices(Dataset):
            def __init__(self, base_ds, indices):
                self.base = base_ds
                self.indices = indices
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, idx):
                return self.base[self.indices[idx]]

        never_trained_ds = SubsetByIndices(ds, never_trained)
        never_trained_loader = DataLoader(never_trained_ds, batch_size=1, shuffle=False)
        evaluate(nb_model, never_trained_loader, device,
                 f"NB ckpt → {len(never_trained)} truly unseen images")

    # On images ONLY in notebook's training set (not in pipeline val)
    nb_only_train = list(set(nb_train_indices) - pipe_val_set)
    print(f"\n  Images in NB train AND pipeline train: {len(nb_only_train)}")

    # ============================================================
    # EVALUATE: Pipeline raw checkpoint
    # ============================================================
    print("\n" + "=" * 80)
    print("PIPELINE RAW CHECKPOINT EVALUATION")
    print("=" * 80)

    raw_ckpt_path = "/home/mis60/sem/logs/runs/unet-d83-raw/checkpoints/best.pt"
    if os.path.exists(raw_ckpt_path):
        pipe_raw_model = PipelineUNet(in_channels=1, num_classes=6, dropout=0.3).to(device)
        raw_ckpt = torch.load(raw_ckpt_path, map_location=device, weights_only=False)
        pipe_raw_model.load_state_dict(raw_ckpt["model_state_dict"])
        print(f"  Raw checkpoint best epoch: {raw_ckpt.get('epoch', '?')}")

        evaluate(pipe_raw_model, pipe_val_loader, device,
                 "Pipeline raw ckpt → Pipeline val (seed=42)")
        evaluate(pipe_raw_model, all_loader, device,
                 "Pipeline raw ckpt → ALL 83 images")
    else:
        print(f"  No raw checkpoint found at {raw_ckpt_path}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 80)
    print("TRAINING CURVE COMPARISON")
    print("=" * 80)
    print(f"  Notebook best val_loss:        0.0529  (epoch 569/1000)")
    print(f"  Pipeline raw best val_loss:     0.5662  (epoch 57/~150)")
    print(f"  Pipeline norm+aug best val_loss: 0.4650  (epoch 632/1000)")
    print(f"")
    print(f"  The notebook's val_loss is 10x lower than any pipeline run.")
    print(f"  This means the notebook's val images were 10x more predictable")
    print(f"  from the notebook's training images than our seed=42 val images")
    print(f"  are from our seed=42 training images.")


if __name__ == "__main__":
    main()
