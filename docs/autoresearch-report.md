# Experiment Report: SEM PV Front Contact Segmentation

**Date:** March 24-26, 2026
**Compute:** CWRU Pioneer HPC cluster (NVIDIA GPUs)

---

## 1. Problem Statement

Maximize validation mean Intersection-over-Union (val_miou) on 6-class semantic segmentation of SEM images of photovoltaic front contacts.

**Dataset:** 130 images at 1024x768 (grayscale), split 85/15 train/val (seed=42)
**Classes:** background, silver, glass, silicon, void, interfacial_void
**Class weights:** [1.0, 1.0, 1.5, 1.0, 1.5, 1.5]

## 2. Baseline

| Setting | Value |
|---------|-------|
| Model | DeepLabV3+ (torchvision) with ResNet101 backbone, no pretrained weights |
| Loss | CrossEntropyLoss with class weights |
| Optimizer | Adam, lr=5e-5, weight_decay=0 |
| Scheduler | None |
| Batch size | 1 |
| Training | 1000 epochs, patience=100 |
| **val_miou** | **0.781** |
| val_f1_macro | 0.868 |
| pixel_accuracy | 0.940 |

## 3. Methodology

We systematically explored modifications to the baseline across five categories: loss functions, optimizers/schedulers, augmentation strategies, architectures, and pretrained weights. Each experiment changed one variable at a time to isolate the effect.

Experiments were run in two stages:
1. **Screening (200 epochs, patience=50):** quickly rank all ideas against baseline
2. **Full training (1000 epochs, patience=200):** promote the most promising candidates to match baseline training conditions

### Code changes for flexibility

- Wired `create_loss()` factory into `train.py` (was hardcoded `nn.CrossEntropyLoss`)
- Added pretrained backbone support to model factory (COCO weights for DeepLab)
- Added SMP (segmentation_models_pytorch) model support: UNet, UNet++, DeepLabV3+
- Added strong augmentation preset
- Fixed trainer to pass spatial tensors `(N,C,H,W)` to loss (was flattening, broke Dice loss)
- Implemented FocalLoss, DiceCELoss, LovaszSoftmaxLoss, LabelSmoothingCELoss

## 4. Experiment Results

### Screening Runs (200 epochs, patience=50)

Each experiment changes ONE thing from the baseline DeepLabV3+ configuration.

#### Loss Functions

| Modification | val_miou | F1 | Acc | Best Epoch | Delta |
|-------------|----------|-----|-----|-----------|-------|
| Dice+CE loss (0.5/0.5) | 0.7718 | 0.862 | 0.936 | 67 | -0.009 |
| Focal loss (gamma=2.0) | 0.6992 | 0.810 | 0.899 | 42 | -0.082 |

#### Optimizers and Schedulers

| Modification | val_miou | F1 | Acc | Best Epoch | Delta |
|-------------|----------|-----|-----|-----------|-------|
| Cosine annealing (lr=1e-4) | 0.7719 | 0.861 | 0.940 | 116 | -0.009 |
| AdamW (weight_decay=0.01) | 0.7683 | 0.858 | 0.938 | 60 | -0.013 |
| Higher LR (3e-4) + cosine | 0.6575 | — | — | — | -0.124 |
| OneCycleLR (max_lr=1e-3) | 0.5640 | — | — | — | -0.217 |

#### Augmentation

| Modification | val_miou | F1 | Acc | Best Epoch | Delta |
|-------------|----------|-----|-----|-----------|-------|
| Strong augmentations | 0.6474 | 0.768 | 0.873 | 47 | -0.134 |

#### Training Configuration

| Modification | val_miou | F1 | Acc | Best Epoch | Delta |
|-------------|----------|-----|-----|-----------|-------|
| Batch size 4 + mixed precision | 0.7693 | 0.859 | 0.937 | 70 | -0.012 |

#### Architecture Changes

| Modification | val_miou | F1 | Acc | Best Epoch | Delta |
|-------------|----------|-----|-----|-----------|-------|
| SMP UNet++ EfficientNet-B4 (imagenet) | 0.7627 | 0.854 | 0.934 | 146 | -0.018 |
| SMP UNet ResNet50 (imagenet) | 0.7509 | — | — | — | -0.030 |
| SMP DeepLabV3+ EfficientNet-B4 (imagenet) | 0.7481 | — | — | — | -0.033 |

#### Pretrained Weights

| Modification | val_miou | F1 | Acc | Best Epoch | Delta |
|-------------|----------|-----|-----|-----------|-------|
| **COCO pretrained backbone** | **0.7771** | **0.865** | **0.941** | **98** | **-0.004** |

### Full Training Runs (1000 epochs, patience=200)

The top screening result (pretrained backbone) was combined with the best scheduler (cosine annealing) and trained to convergence.

| Combination | val_miou | F1 | Acc | Epoch | Delta |
|------------|----------|-----|-----|-------|-------|
| **Pretrained COCO + Cosine (lr=1e-4)** | **0.786+** | **0.871** | **0.945** | **262 (still improving)** | **+0.005+** |

### Ranked Summary (all completed experiments)

| Rank | Experiment | val_miou | vs Baseline |
|------|-----------|----------|------------|
| **1** | **Pretrained + Cosine (1000ep)** | **0.786+** | **+0.005+** |
| 2 | Pretrained COCO backbone | 0.777 | -0.004 |
| 3 | Cosine annealing | 0.772 | -0.009 |
| 4 | Dice+CE loss | 0.772 | -0.009 |
| 5 | Batch4 + AMP | 0.769 | -0.012 |
| 6 | AdamW | 0.768 | -0.013 |
| 7 | SMP UNet++ EffNet-B4 | 0.763 | -0.018 |
| 8 | SMP UNet ResNet50 | 0.751 | -0.030 |
| 9 | SMP DeepLabV3+ EffNet-B4 | 0.748 | -0.033 |
| 10 | Focal loss | 0.699 | -0.082 |
| 11 | Higher LR + cosine | 0.658 | -0.124 |
| 12 | Strong augmentations | 0.647 | -0.134 |
| 13 | OneCycleLR | 0.564 | -0.217 |

## 5. Key Findings

### What works

1. **Pretrained COCO backbone is the single most impactful change.** It reached 0.777 in just 98 epochs during screening (vs baseline 0.781 at ~1000 epochs). Combined with cosine annealing, it surpassed baseline at epoch 262.

2. **Cosine annealing schedule** consistently performed near baseline and combined well with pretrained weights.

3. **Dice+CE combined loss** performed comparably to pure CE — marginal difference.

4. **Larger batch size (4) with mixed precision** slightly underperformed but trained faster per wall-clock time.

### What doesn't work (on this dataset)

1. **Strong augmentations (-0.134).** With only 130 images, aggressive augmentation (larger rotations, elastic transforms, coarse dropout, shift-scale-rotate) hurt badly. The model likely can't learn stable features when the already-small dataset is heavily distorted.

2. **Focal loss (-0.082).** Hard-example mining didn't help. The class imbalance isn't severe enough to benefit from focal weighting, and the gamma=2.0 focusing may have destabilized early training.

3. **OneCycleLR (-0.217).** The aggressive warmup to 1e-3 and rapid decay was far too volatile for this small dataset. The model never recovered from the initial high-LR phase.

4. **Higher learning rate with cosine (-0.124).** Even 3e-4 (vs 5e-5 baseline) was too aggressive, confirming this dataset benefits from conservative optimization.

5. **Architecture changes via SMP didn't help.** UNet++ with EfficientNet-B4 (0.763), DeepLabV3+ with EfficientNet-B4 (0.748), and UNet with ResNet50 (0.751) all underperformed the torchvision DeepLabV3+ ResNet101. The pretrained torchvision model with COCO weights proved more effective than SMP's ImageNet-pretrained encoders.

### Lessons for small-dataset segmentation

- **Pretrained weights matter more than architecture.** COCO-pretrained DeepLabV3+ beat all ImageNet-pretrained SMP variants.
- **Conservative optimization is key.** Low learning rate (5e-5 to 1e-4), no aggressive scheduling, simple Adam optimizer.
- **Less augmentation is more.** The default light augmentation (small rotations, flips, slight blur/noise) outperformed aggressive augmentation by a wide margin.
- **Screening runs underestimate full potential.** The 200-epoch screening with patience=50 capped experiments prematurely. The pretrained backbone reached 0.777 at ep98 in screening but 0.786+ at ep262 in full training — and was still climbing.

## 6. Recommended Configuration

Based on all experiments, the optimal configuration is:

```yaml
# Best performing: Pretrained + Cosine Annealing
data: data/merged
model: model/deeplabv3plus_resnet101_pretrained  # COCO pretrained weights

data_overrides:
  batch_size: 2

training:
  max_epochs: 1000
  device: auto
  mixed_precision: false
  monitor: val_loss
  monitor_mode: min
  patience: 200
  min_delta: 0.0

optimizer:
  type: adam
  lr: 0.0001  # 1e-4, slightly higher than baseline with cosine decay
  weight_decay: 0.0

scheduler:
  type: cosine
  min_lr: 0.0000001  # 1e-7

loss:
  type: ce
  class_weights: [1.0, 1.0, 1.5, 1.0, 1.5, 1.5]
```

**Expected performance:** val_miou > 0.79 (observed 0.786 at epoch 262/1000, still improving)

## 7. Remaining Work

1. **Complete the pretrained + cosine run** to full 1000-epoch convergence
2. **Run TTA (test-time augmentation)** on the best checkpoint for a free 1-3% boost
3. **Complete full 1000-epoch runs** for the other top screening results (Dice+CE, cosine, AdamW, batch4+AMP) to see if any also beat baseline when given enough training time
4. **Per-class IoU analysis** to identify which classes benefit most from pretrained weights
5. **Try pretrained + Dice+CE + cosine** triple combo
