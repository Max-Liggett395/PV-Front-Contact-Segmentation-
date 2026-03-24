# AutoResearch: SEM PV Front Contact Segmentation

## Objective

Maximize **val_miou** (validation mean Intersection-over-Union) on 6-class semantic
segmentation of SEM images of photovoltaic front contacts.

## Current Baseline

| Metric | Value |
|--------|-------|
| val_miou | 0.781 |
| val_f1_macro | 0.868 |
| pixel_accuracy | 0.940 |
| Model | DeepLabV3+ ResNet101 |
| Dataset | merged (130 images) |
| Loss | CrossEntropyLoss with class_weights [1.0, 1.0, 1.5, 1.0, 1.5, 1.5] |
| Optimizer | Adam, lr=5e-5, weight_decay=0 |
| Scheduler | None |
| Batch size | 1 |
| Epochs | 1000, patience=100 |

## Classes

0: background, 1: silver, 2: glass, 3: silicon, 4: void, 5: interfacial_void

## What You May Change

- Model architecture (backbone, decoder, attention modules, new models from SMP)
- Loss function (Dice, Focal, Lovász, combinations)
- Optimizer (AdamW, SGD+momentum, etc.) and learning rate
- LR scheduler (cosine annealing, warmup, OneCycleLR, etc.)
- Data augmentation pipeline (new transforms, stronger/weaker augmentation)
- Batch size and gradient accumulation
- Dropout rate, weight decay, and other regularization
- Mixed precision training (AMP)
- Pretrained weights (ImageNet, COCO)
- Class weights for loss function
- Any training loop modifications

## What You Must NOT Change

- Evaluation metrics implementation (src/evaluation/metrics.py)
- Data split seed (42) and train_split ratio (0.85)
- Image resolution (1024x768)
- Number of classes (6)
- The validation protocol (compute_metrics on full val set)

## Constraints

- Must train on a single GPU with ≤24GB VRAM (RTX 4090)
- Each screening experiment: 200 epochs max, patience=50
- Results must be written to standardized JSON format

## Experiment Protocol

1. Make ONE focused change per experiment (for attribution)
2. Keep all other settings identical to baseline
3. Report: val_miou, val_f1_macro, pixel_accuracy, per_class_iou, val_loss
4. If val_miou improves over baseline, the change is a WINNER
