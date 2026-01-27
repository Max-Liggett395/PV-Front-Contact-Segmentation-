# SEM Image Segmentation with U-Net

Production-ready training pipeline for semantic segmentation of cross-sectional SEM images of photovoltaic cell metallization contacts.

**Paper**: "Semantic Segmentation for Cross-Sectional Scanning Electron Microscopy Images of Photovoltaic Cell Metallization: A Deep Learning Approach" (2025)

## Overview

This repository implements a U-Net deep learning model for automatically segmenting SEM images into six material classes:

| Class | Material | Color (Visualization) |
|-------|----------|----------------------|
| 0 | Background | Red |
| 1 | Silver (Ag) | Green |
| 2 | Glass | Blue |
| 3 | Silicon (Si) | Yellow |
| 4 | Void | Magenta |
| 5 | Interfacial Void | Cyan |

### Dataset

- **Total**: 116 annotated SEM images with masks
- **Resolution**: 1024×768 pixels (resized from original)
- **Format**: PNG images, JSON label files
- **Cell Types**: Al-BSF, PERC, TOPCon, nPERT
- **Annotations**: VGG Image Annotator (VIA) polygons

## Installation

```bash
# Clone repository
git clone https://github.com/Max-Liggett395/PV-Front-Contact-Segmentation-.git
cd PV-Front-Contact-Segmentation-

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python ≥3.8
- PyTorch ≥2.0.0
- See `requirements.txt` for full list

## Quick Start

### 1. Training

Train a model from scratch:

```bash
python scripts/train.py --config config/train_config.yaml
```

Resume training from checkpoint:

```bash
python scripts/train.py --config config/train_config.yaml --resume checkpoints/latest_checkpoint.pth
```

### 2. Testing

Test the trained model on the test set:

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --config config/train_config.yaml
```

Save metrics to JSON and generate visualizations:

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --config config/train_config.yaml \
    --output results/metrics.json \
    --visualize --num-viz 20
```

### 3. Inference

Run inference on a single image:

```bash
python scripts/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --input data/images/example.png \
    --output results/prediction.png \
    --overlay --alpha 0.5
```

Batch inference on a directory:

```bash
python scripts/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --input-dir data/images/ \
    --output-dir results/predictions/ \
    --overlay
```

## Project Structure

```
PV-Front-Contact-Segmentation-/
├── config/
│   └── train_config.yaml          # Hyperparameters and settings
├── src/
│   ├── models/
│   │   └── unet.py                # U-Net architecture
│   ├── data/
│   │   ├── dataset.py             # SEMDataset class
│   │   └── transforms.py          # Augmentation pipelines
│   ├── training/
│   │   ├── losses.py              # Loss functions
│   │   └── trainer.py             # Training loop
│   ├── evaluation/
│   │   └── metrics.py             # F1, IoU, Dice metrics
│   └── utils/
│       ├── checkpointing.py       # Model save/load
│       └── visualization.py       # Prediction overlays
├── scripts/
│   ├── train.py                   # Main training script
│   ├── evaluate.py                # Test set assessment
│   └── inference.py               # Run predictions
├── tests/
│   ├── test_dataset.py            # Dataset tests
│   ├── test_model.py              # Model architecture tests
│   └── test_training.py           # Training component tests
├── data/
│   ├── images/                    # SEM images (116 files)
│   └── labels/                    # JSON label files
└── requirements.txt
```

## Configuration

Edit `config/train_config.yaml` to customize:

### Dataset
- `images_dir`, `labels_dir`: Paths to data
- `train_split`, `val_split`, `test_split`: 70/15/15 default
- `seed`: Random seed for reproducibility

### Model
- `num_classes`: 6 (fixed for this dataset)
- `dropout`: 0.3 (regularization)
- `in_channels`: 1 (grayscale)

### Training
- `batch_size`: 4 (default per paper)
- `num_epochs`: 1000 (with early stopping)
- `learning_rate`: 1e-4
- `early_stopping.patience`: 50 epochs

### Augmentation
- `gaussian_blur`: Morphological (most effective)
- `gauss_noise`: Morphological
- `horizontal_flip`, `vertical_flip`, `rotate`: Geometric (less effective)

### Loss
- `type`: "cross_entropy" (primary), "bce_dice" (alternative)
- `class_weights`: [1.0, 1.0, 1.5, 1.0, 1.5, 1.5] for imbalance

## Testing

Run the test suite:

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_model.py -v

# Specific test
pytest tests/test_model.py::test_model_parameter_count -v
```

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{liggett2025semantic,
  title={Semantic Segmentation for Cross-Sectional Scanning Electron Microscopy Images of Photovoltaic Cell Metallization: A Deep Learning Approach},
  author={Liggett, et al.},
  journal={...},
  year={2025}
}
```

## Troubleshooting

### Out of Memory

Reduce `batch_size` in config (try 2 or 1):

```yaml
training:
  batch_size: 2
```

### Slow Training

- Enable mixed precision: `device.mixed_precision: true` in config
- Use GPU (CUDA or MPS): training automatically detects
- Reduce augmentation probability values

### Poor Performance

- Increase training epochs: `num_epochs: 1500`
- Adjust class weights in loss function
- Try alternative loss: `loss.type: "bce_dice"`
- Check for data quality issues (corrupted images/labels)

## License

See LICENSE file.

## Contact

For questions about the dataset or implementation, refer to the paper or repository issues.
