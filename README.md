# PV Front Contact Segmentation

Semantic segmentation of SEM (Scanning Electron Microscope) images of photovoltaic front contacts. Classifies each pixel into one of six material/defect classes using deep learning.

## Dataset

- **113 SEM images** at 1024×768 (grayscale) — `data/new_113/`
- **6 classes:** background, silver, glass, silicon, void, interfacial_void
- **Split:** 85% train / 15% val (seed=42)

## Results

Five [`segmentation_models.pytorch`](https://github.com/qubvel-org/segmentation_models.pytorch) architectures benchmarked on the 113-image dataset:

| Model | Encoder | Best Epoch | Macro F1 | Pixel Acc | mIoU |
|-------|---------|------------|----------|-----------|------|
| **DeepLabV3+** | ResNet50 | 77 | 0.8516 | 0.9432 | **0.7626** |
| **SegFormer**  | MiT-B2   | 33 | **0.8529** | 0.9385 | 0.7623 |
| DeepLabV3      | ResNet50 | 64 | 0.8286 | 0.9332 | 0.7333 |
| U-Net++        | ResNet50 | 40 | 0.8278 | 0.9299 | 0.7305 |
| U-Net          | ResNet50 | 75 | 0.8243 | 0.9258 | 0.7259 |

Source: [`logs/results_new113.md`](logs/results_new113.md). Training curves: [`logs/runs/new113_5model_curves.png`](logs/runs/new113_5model_curves.png).

_Micro F1 ≡ pixel accuracy by construction for multiclass single-label segmentation, so only one column is shown above._

## Key Findings

- **DeepLabV3+** wins on mIoU and pixel accuracy.
- **SegFormer** wins on macro F1 and converges fastest (peak at epoch 33 vs. 60–80 for the CNN-based models).
- The U-Net family trails by 0.02–0.04 on F1 and mIoU.

## Quick Start

```bash
pip install -r requirements.txt

# Train all five models (SLURM, with chained saliency job per model)
bash scripts/train_new113_5models.sh

# ...or train one
python train.py --config configs/experiment/smp_segformer_new113.yaml

# Evaluate one
python evaluate.py --experiment smp_segformer_new113 \
  --checkpoint logs/runs/smp-segformer-new113/checkpoints/best.pt

# Per-image SmoothGrad saliency
sbatch scripts/saliency_new113.slurm

# After all five trainings finish: assemble results table + comparison plot
bash scripts/post_new113_5models.sh
```

## Project Structure

```
├── configs/
│   ├── data/new_113.yaml
│   ├── model/                       # smp_{unet,unetpp,deeplabv3,deeplabv3plus}_rn50.yaml,
│   │                                # smp_segformer_mitb2.yaml
│   └── experiment/                  # smp_*_new113.yaml (five experiment configs)
├── src/
│   ├── data/                        # Dataset, transforms, loading
│   ├── models/                      # Model factory (registers SMP architectures), losses
│   ├── training/                    # Trainer, schedulers
│   ├── evaluation/                  # Metrics (mIoU, macro/micro F1, pixel acc, per-class IoU)
│   └── utils/                       # Config, logging
├── scripts/
│   ├── train_new113_5models.sh      # Submit all five trainings + chained saliency jobs
│   ├── saliency_new113.slurm        # SLURM saliency job (SmoothGrad)
│   ├── post_new113_5models.sh       # After-training: build table + plot
│   ├── build_results_table.py       # → logs/results_new113.md
│   ├── plot_curves.py               # → logs/runs/new113_5model_curves.png
│   └── error_maps.py                # Per-image error visualization (with class legend)
├── logs/
│   ├── results_new113.md            # Aggregate metrics table
│   └── runs/smp-*-new113/           # Checkpoints + tensorboard per model
├── viz/
│   └── saliency_smp-*-new113/       # SmoothGrad saliency PNGs (5 sample images per model)
├── train.py                         # Training entrypoint
├── evaluate.py                      # Evaluation entrypoint
├── predict.py                       # Inference on new images
└── requirements.txt
```
