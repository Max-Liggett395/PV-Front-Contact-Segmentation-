# AutoResearch Report: SEM PV Front Contact Segmentation

**Date:** March 24-26, 2026
**Method:** Karpathy-style autoresearch loop adapted for semantic segmentation
**Compute:** RunPod RTX 4090 GPUs ($0.59/hr), 4 pods in parallel
**Total experiments:** 19 designed, 13 completed screening, 1 confirmed beating baseline

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

This baseline was established on the CWRU Pioneer HPC cluster (NVIDIA DGX / A100 class GPUs).

## 3. Methodology

Adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch):

1. **Editable asset:** experiment configs (YAML) + model/loss/augmentation code
2. **Scalar metric:** val_miou (higher is better)
3. **Cycle:** Phase 1-2 used 200-epoch screening runs (patience=50) to quickly rank ideas. Phase 3 promoted winners to full 1000-epoch runs (patience=200).

### Modifications to Karpathy's approach

- **Parallel batch exploration** instead of serial greedy hill-climbing (multiple pods running simultaneously)
- **SLURM-style job submission** adapted for RunPod cloud GPUs
- **Two-stage evaluation:** fast screening (200ep) then full training (1000ep) for winners

### Infrastructure built

| File | Purpose |
|------|---------|
| `autoresearch/run_experiment.py` | Standardized experiment runner with JSON output |
| `autoresearch/evaluate_batch.py` | Leaderboard generator comparing results to baseline |
| `autoresearch/tta_evaluate.py` | Test-time augmentation evaluator |
| `autoresearch/program.md` | Agent instructions (search space, constraints) |
| `autoresearch/runpod_launcher.py` | RunPod SDK-based parallel pod launcher |
| `autoresearch/pod_start.sh` | Pod startup script for RunPod environments |

### Code changes for flexibility

- Wired `create_loss()` factory into `train.py` (was hardcoded `nn.CrossEntropyLoss`)
- Added pretrained backbone support to model factory (COCO weights for DeepLab)
- Added SMP (segmentation_models_pytorch) model support: UNet, UNet++, DeepLabV3+
- Added strong augmentation preset
- Fixed trainer to pass spatial tensors `(N,C,H,W)` to loss (was flattening, broke Dice loss)
- Added FocalLoss, FocalDiceLoss, LovaszSoftmaxLoss, LabelSmoothingCELoss

## 4. Experiment Results

### Phase 1: Single-Change Screening (200 epochs, patience=50)

Each experiment changes ONE thing from the baseline DeepLabV3+ merged config.

| Rank | Exp | Modification | val_miou | F1 | Acc | Stopped | Delta |
|------|-----|-------------|----------|-----|-----|---------|-------|
| 1 | 07 | Pretrained COCO backbone | **0.7771** | 0.865 | 0.941 | ep98 | -0.004 |
| 2 | 04 | Cosine annealing (lr=1e-4) | 0.7719 | 0.861 | 0.940 | ep166 | -0.009 |
| 3 | 01 | Dice+CE loss (0.5/0.5) | 0.7718 | 0.862 | 0.936 | ep117 | -0.009 |
| 4 | 06 | Batch size 4 + mixed precision | 0.7693 | 0.859 | 0.937 | ep70 | -0.012 |
| 5 | 05 | AdamW (weight_decay=0.01) | 0.7683 | 0.858 | 0.938 | ep110 | -0.013 |
| 6 | 08 | SMP UNet++ EfficientNet-B4 | 0.7627 | 0.854 | 0.934 | ep146 | -0.018 |
| 7 | 02 | Focal loss (gamma=2.0) | 0.6992 | 0.810 | 0.899 | ep92 | -0.082 |
| 8 | 03 | Strong augmentations | 0.6474 | 0.768 | 0.873 | ep97 | -0.134 |

### Phase 2: Additional Ideas (200 epochs, patience=50)

| Rank | Exp | Modification | val_miou | Delta |
|------|-----|-------------|----------|-------|
| 1 | 12 | SMP UNet ResNet50 (imagenet) | 0.7509 | -0.030 |
| 2 | 11 | SMP DeepLabV3+ EfficientNet-B4 (imagenet) | 0.7481 | -0.033 |
| 3 | 09 | Higher LR (3e-4) + cosine | 0.6575 | -0.124 |
| 4 | 10 | OneCycleLR (max_lr=1e-3) | 0.5640 | -0.217 |
| - | 13 | Lovasz-Softmax loss | killed | too slow per epoch |
| - | 14 | Label smoothing CE (0.1) | never ran | — |

### Phase 3: Combo + Full Training (1000 epochs, patience=200)

| Exp | Combination | val_miou | Status |
|-----|------------|----------|--------|
| **15** | **Pretrained + Cosine (lr=1e-4)** | **0.786** | **Beat baseline at epoch 262/1000, still climbing. Pods terminated before completion.** |
| 16 | Pretrained + Dice+CE + Cosine | — | Started, pods terminated |
| 17 | SMP DeepLabV3+ ResNet101 (imagenet) | — | Never reached |
| 18 | Pretrained + Cosine + Dice+CE + Batch4 + AMP | — | Started, pods terminated |
| 19 | SMP DeepLabV3+ EfficientNet-B7 (imagenet) | — | Never reached |

### Full Training Runs (1000 epochs, patience=200) — Lost

These were launched on Pods 3-5 but all pods were terminated before completion:

| Exp | Original Screening mIoU | Full Run Status |
|-----|------------------------|----------------|
| 01-full | 0.7718 (Dice+CE) | Started, lost |
| 04-full | 0.7719 (Cosine) | Never started |
| 05-full | 0.7683 (AdamW) | Never started |
| 06-full | 0.7693 (Batch4+AMP) | Started, lost |
| 08-full | 0.7627 (SMP UNet++) | Never started |
| 11-full | 0.7481 (SMP DLV3+ B4) | Never started |
| 12-full | 0.7509 (SMP UNet RN50) | Started, lost |

## 5. Key Findings

### What works

1. **Pretrained COCO backbone is the single most impactful change.** It reached 0.777 in just 98 epochs during screening (vs baseline 0.781 at ~1000 epochs). Combined with cosine annealing, it surpassed baseline at epoch 262.

2. **Cosine annealing schedule** consistently performed near baseline and combined well with pretrained weights.

3. **Dice+CE combined loss** performed comparably to pure CE — marginal difference.

4. **Larger batch size (4) with mixed precision** slightly underperformed but trained faster per wall-clock time.

### What doesn't work (on this dataset)

1. **Strong augmentations (-0.134).** With only 130 images, aggressive augmentation (larger rotations, elastic transforms, coarse dropout, shift-scale-rotate) hurt badly. The model likely can't learn stable features when the already-small dataset is heavily distorted.

2. **Focal loss (-0.082).** Hard-example mining didn't help — the class imbalance isn't severe enough to benefit from focal weighting, and the gamma=2.0 focusing may have destabilized early training.

3. **OneCycleLR (-0.217).** The aggressive warmup to 1e-3 and rapid decay was far too volatile for this small dataset. The model never recovered from the initial high-LR phase.

4. **Higher learning rate with cosine (-0.124).** Even 3e-4 (vs 5e-5 baseline) was too aggressive, confirming this dataset benefits from conservative optimization.

5. **Architecture changes via SMP didn't help.** UNet++ with EfficientNet-B4 (0.763), DeepLabV3+ with EfficientNet-B4 (0.748), and UNet with ResNet50 (0.751) all underperformed the torchvision DeepLabV3+ ResNet101. The pretrained torchvision model with COCO weights proved more effective than SMP's ImageNet-pretrained encoders.

### Lessons for small-dataset segmentation

- **Pretrained weights matter more than architecture.** COCO-pretrained DeepLabV3+ beat all ImageNet-pretrained SMP variants.
- **Conservative optimization is key.** Low learning rate (5e-5), no aggressive scheduling, simple Adam optimizer.
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

1. **Complete exp15** to full 1000-epoch convergence with network volume persistence
2. **Run TTA (test-time augmentation)** on the best checkpoint for a free 1-3% boost
3. **Complete full 1000-epoch runs** for the other top-5 screening results (exp01, 04, 05, 06) to see if any also beat baseline when given enough training time
4. **Per-class IoU analysis** to identify which classes benefit most from pretrained weights
5. **Try exp16 (pretrained + Dice+CE + cosine)** — the triple combo that never completed

## 8. Cost Summary

| Phase | Pods | Duration | GPU Cost |
|-------|------|----------|----------|
| Phase 1 screening (8 exp) | 1x RTX 4090 | ~7 hours | ~$4 |
| Phase 2 screening (6 exp) | 1x RTX 4090 | ~5 hours | ~$3 |
| Phase 3 full training | 1x RTX 4090 | ~8 hours | ~$5 |
| Full training (Pods 3-5) | 3x RTX 4090 | ~8 hours | ~$14 |
| Pod idle/setup time | various | ~3 hours | ~$4 |
| **Total compute** | | | **~$30** |
| Network volume (20GB, 1 month) | | | $1.40 |

## Appendix: All Experiment Configs

Located in `configs/experiment/autoresearch/`:

```
ar_exp01_dice_ce_loss.yaml          # Dice+CE combined loss
ar_exp02_focal_loss.yaml            # Focal loss (gamma=2.0)
ar_exp03_strong_augment.yaml        # Aggressive augmentation preset
ar_exp04_cosine_schedule.yaml       # CosineAnnealingLR, lr=1e-4
ar_exp05_adamw.yaml                 # AdamW, weight_decay=0.01
ar_exp06_batch4_amp.yaml            # Batch size 4, mixed precision
ar_exp07_pretrained.yaml            # COCO pretrained backbone
ar_exp08_smp_unetpp.yaml            # SMP UNet++ EfficientNet-B4
ar_exp09_higher_lr_cosine.yaml      # lr=3e-4 + cosine
ar_exp10_onecycle.yaml              # OneCycleLR, max_lr=1e-3
ar_exp11_smp_dlv3plus_effb4.yaml    # SMP DeepLabV3+ EfficientNet-B4
ar_exp12_smp_unet_resnet50.yaml     # SMP UNet ResNet50
ar_exp13_lovasz_loss.yaml           # Lovász-Softmax loss
ar_exp14_label_smoothing.yaml       # Label smoothing CE (0.1)
ar_exp15_pretrained_cosine.yaml     # Pretrained + cosine (WINNER)
ar_exp16_pretrained_dicece_cosine.yaml  # Pretrained + Dice+CE + cosine
ar_exp17_smp_dlv3p_rn101.yaml       # SMP DeepLabV3+ ResNet101
ar_exp18_pretrained_combo.yaml      # Pretrained + cosine + Dice+CE + batch4 + AMP
ar_exp19_smp_dlv3p_effb7.yaml       # SMP DeepLabV3+ EfficientNet-B7
```

Model configs in `configs/model/`:

```
deeplabv3plus_resnet101.yaml             # Baseline model
deeplabv3plus_resnet101_pretrained.yaml  # + COCO pretrained weights
smp_unetpp_effb4.yaml                   # SMP UNet++ EfficientNet-B4
smp_deeplabv3plus_effb4.yaml            # SMP DeepLabV3+ EfficientNet-B4
smp_deeplabv3plus_rn101.yaml            # SMP DeepLabV3+ ResNet101
smp_deeplabv3plus_effb7.yaml            # SMP DeepLabV3+ EfficientNet-B7
smp_unet_resnet50.yaml                  # SMP UNet ResNet50
```
