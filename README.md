# PV Front Contact Segmentation

Semantic segmentation of SEM (Scanning Electron Microscope) images of photovoltaic front contacts. Classifies each pixel into one of six material/defect classes using deep learning.

## Dataset

- **130 SEM images** at 1024x768 (grayscale)
- **6 classes:** background, silver, glass, silicon, void, interfacial_void
- **Split:** 85% train / 15% val (seed=42)

## Results

**Best model: DeepLabV3+ ResNet101 with COCO-pretrained backbone + cosine annealing**

| Rank | Method | val mIoU | vs Baseline |
|------|--------|----------|-------------|
| **1** | **Pretrained + Cosine Annealing** | **0.786** | **+0.005** |
| 2 | Pretrained COCO backbone | 0.777 | -0.004 |
| 3 | Cosine annealing (lr=1e-4) | 0.772 | -0.009 |
| 4 | Dice+CE loss (0.5/0.5) | 0.772 | -0.009 |
| 5 | Batch size 4 + AMP | 0.769 | -0.012 |
| 6 | AdamW (wd=0.01) | 0.768 | -0.013 |
| 7 | SMP UNet++ EffNet-B4 | 0.763 | -0.018 |
| 8 | SMP UNet ResNet50 | 0.751 | -0.030 |
| 9 | SMP DeepLabV3+ EffNet-B4 | 0.748 | -0.033 |
| 10 | Focal loss (gamma=2.0) | 0.699 | -0.082 |
| 11 | Higher LR (3e-4) + cosine | 0.658 | -0.124 |
| 12 | Strong augmentations | 0.647 | -0.134 |
| 13 | OneCycleLR (max_lr=1e-3) | 0.564 | -0.217 |
| — | **Baseline** | **0.781** | — |

Baseline: DeepLabV3+ ResNet101 (random init), Adam lr=5e-5, CrossEntropyLoss, 1000 epochs.

Full experiment details in [docs/autoresearch-report.md](docs/autoresearch-report.md).

## Key Findings

- **Pretrained weights matter more than architecture.** COCO-pretrained DeepLabV3+ beat all ImageNet-pretrained SMP variants.
- **Conservative optimization is key.** Low learning rate (1e-4), cosine decay, no aggressive scheduling.
- **Less augmentation is more.** Light augmentation outperformed strong augmentation by 13 mIoU points on this small dataset.
- **Architecture swaps didn't help.** UNet++, UNet, and alternative backbones all underperformed the baseline architecture.

## Quick Start

```bash
pip install -r requirements.txt

# Train baseline
python train.py --config configs/experiment/autoresearch/ar_exp07_pretrained.yaml

# Train best model (pretrained + cosine)
python train.py --config configs/experiment/autoresearch/ar_exp15_pretrained_cosine.yaml

# Evaluate
python evaluate.py --checkpoint logs/runs/<run_name>/checkpoints/best.pt
```

## Project Structure

```
├── configs/
│   ├── data/                # Dataset configs
│   ├── model/               # Model architecture configs
│   └── experiment/
│       └── autoresearch/    # All experiment configs (exp01-19)
├── src/
│   ├── data/                # Dataset, transforms, loading
│   ├── models/              # Model factory, losses
│   ├── training/            # Trainer, schedulers
│   ├── evaluation/          # Metrics
│   └── utils/               # Config, logging
├── autoresearch/            # Experiment runner and evaluation scripts
├── docs/
│   └── autoresearch-report.md  # Full experiment report
├── train.py                 # Training entrypoint
├── evaluate.py              # Evaluation entrypoint
└── requirements.txt
```
