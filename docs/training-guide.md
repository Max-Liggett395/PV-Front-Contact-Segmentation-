# Training, Evaluation, and Inference Guide

## Quick Start

```bash
# Train the best model (pretrained DeepLabV3+ with cosine annealing)
python train.py --experiment configs/experiment/autoresearch/ar_exp15_pretrained_cosine.yaml --run-name my-run

# Evaluate a trained model
python evaluate.py --checkpoint logs/runs/my-run/checkpoints/best.pt --data-config configs/data/merged.yaml

# Run inference on new images (see Section 5)
python predict.py --checkpoint logs/runs/my-run/checkpoints/best.pt --input images/ --output predictions/
```

---

## 1. Experiment Config Format

Everything is controlled by a single YAML experiment config. It references a data config and model config by path, then specifies training, optimizer, scheduler, and loss settings.

```yaml
# configs/experiment/my_experiment.yaml

data: data/merged                                    # path under configs/, no .yaml extension
model: model/deeplabv3plus_resnet101_pretrained       # path under configs/, no .yaml extension

data_overrides:       # optional — override anything in the data config
  batch_size: 2

training:
  max_epochs: 1000
  device: auto        # auto | cuda | cpu | mps
  mixed_precision: false
  monitor: val_loss
  monitor_mode: min   # min | max
  patience: 200       # early stopping patience (epochs without improvement)
  min_delta: 0.0      # minimum change to count as improvement

optimizer:
  type: adam           # adam | adamw | sgd
  lr: 0.0001
  weight_decay: 0.0

scheduler:             # optional — omit for no scheduler
  type: cosine         # cosine | reduce_on_plateau | cosine_warm_restarts | one_cycle
  T_max: 1000
  min_lr: 0.0000001

loss:
  type: ce             # ce | dice | dice_ce | focal | focal_dice | lovasz | label_smoothing_ce
  class_weights: [1.0, 1.0, 1.5, 1.0, 1.5, 1.5]
```

### Running training

```bash
python train.py --experiment configs/experiment/my_experiment.yaml --run-name my-run
```

Output goes to `logs/runs/my-run/`:
```
logs/runs/my-run/
├── checkpoints/
│   ├── best.pt       # best validation checkpoint
│   └── latest.pt     # most recent epoch
└── tensorboard/      # training curves
```

Monitor with TensorBoard:
```bash
tensorboard --logdir logs/runs/my-run/tensorboard
```

---

## 2. All Available Options

### Models

Set via the `model:` field in the experiment config, pointing to a model config file.

| Model Config | Architecture | Backbone | Pretrained | Params |
|-------------|-------------|----------|-----------|--------|
| `model/deeplabv3plus_resnet101_pretrained` | DeepLabV3+ | ResNet101 | COCO | 58.6M |
| `model/deeplabv3plus_resnet101` | DeepLabV3+ | ResNet101 | No | 58.6M |
| `model/deeplabv3_resnet50` | DeepLabV3 | ResNet50 | No | 39.6M |
| `model/unet_resnet34` | U-Net | Custom | No | ~31M |
| `model/smp_unetpp_effb4` | UNet++ (SMP) | EfficientNet-B4 | ImageNet | ~18M |
| `model/smp_deeplabv3plus_effb4` | DeepLabV3+ (SMP) | EfficientNet-B4 | ImageNet | ~14M |
| `model/smp_deeplabv3plus_rn101` | DeepLabV3+ (SMP) | ResNet101 | ImageNet | ~58M |
| `model/smp_deeplabv3plus_effb7` | DeepLabV3+ (SMP) | EfficientNet-B7 | ImageNet | ~63M |
| `model/smp_unet_resnet50` | UNet (SMP) | ResNet50 | ImageNet | ~32M |

**To create a new model config**, add a YAML file to `configs/model/`:

```yaml
# configs/model/my_model.yaml
architecture: smp_deeplabv3plus   # architecture name (see table above)
encoder_name: resnet50            # SMP encoder (only for smp_* architectures)
encoder_weights: imagenet         # imagenet | null (only for smp_* architectures)
pretrained: true                  # true | false (only for torchvision architectures)
in_channels: 1                    # 1 for grayscale, 3 for RGB
num_classes: 6
```

### Datasets

Set via the `data:` field. Available configs in `configs/data/`:

| Data Config | Images | Description |
|------------|--------|-------------|
| `data/merged` | 130 | Combined dataset (recommended) |
| `data/dataset_116` | 116 | 116-image subset |
| `data/dataset_83` | 83 | Original 83-image dataset |

All use an 85/15 train/val split with seed=42.

### Optimizers

Set in the `optimizer:` block of the experiment config.

**Adam** (default):
```yaml
optimizer:
  type: adam
  lr: 0.00005       # default 5e-5
  weight_decay: 0.0
```

**AdamW** (Adam with decoupled weight decay):
```yaml
optimizer:
  type: adamw
  lr: 0.00005
  weight_decay: 0.01  # default 1e-2
```

**SGD** (with momentum):
```yaml
optimizer:
  type: sgd
  lr: 0.01           # default 1e-2
  momentum: 0.9
  weight_decay: 0.0001
```

### Schedulers

Set in the `scheduler:` block. Omit entirely for a fixed learning rate.

**Cosine Annealing** (recommended):
```yaml
scheduler:
  type: cosine            # or cosine_annealing
  T_max: 1000             # defaults to max_epochs
  min_lr: 0.0000001       # minimum LR at end of cycle
```

**Reduce on Plateau** (reduce LR when validation stalls):
```yaml
scheduler:
  type: reduce_on_plateau
  factor: 0.5             # multiply LR by this factor
  patience: 10            # epochs to wait before reducing
  min_lr: 0.0000001
```

**Cosine Warm Restarts** (cosine with periodic resets):
```yaml
scheduler:
  type: cosine_warm_restarts
  T_0: 50                 # epochs in first cycle
  T_mult: 2               # multiply cycle length after each restart
  min_lr: 0.0000001
```

**OneCycleLR** (warmup then decay — aggressive):
```yaml
scheduler:
  type: one_cycle
  # max_lr is auto-set to 10x the optimizer lr
```

### Loss Functions

Set in the `loss:` block.

**Cross-Entropy** (default):
```yaml
loss:
  type: ce
  class_weights: [1.0, 1.0, 1.5, 1.0, 1.5, 1.5]  # per-class weights
  ignore_index: -1                                   # optional
```

**Dice + Cross-Entropy** (combined):
```yaml
loss:
  type: dice_ce
  dice_weight: 0.5
  ce_weight: 0.5
  class_weights: [1.0, 1.0, 1.5, 1.0, 1.5, 1.5]
```

**Focal Loss** (emphasize hard examples):
```yaml
loss:
  type: focal
  gamma: 2.0
  alpha: 0.25              # or per-class list
  class_weights: [1.0, 1.0, 1.5, 1.0, 1.5, 1.5]
```

**Focal + Dice** (combined):
```yaml
loss:
  type: focal_dice
  gamma: 2.0
  focal_weight: 0.5
  dice_weight: 0.5
  class_weights: [1.0, 1.0, 1.5, 1.0, 1.5, 1.5]
```

**Lovasz-Softmax** (directly optimizes IoU):
```yaml
loss:
  type: lovasz
  classes: present         # present | all
```

**Label Smoothing Cross-Entropy**:
```yaml
loss:
  type: label_smoothing_ce
  smoothing: 0.1
  class_weights: [1.0, 1.0, 1.5, 1.0, 1.5, 1.5]
```

**Dice Loss** (standalone):
```yaml
loss:
  type: dice
  smooth: 1.0
  ignore_index: -1
```

### Augmentation

Controlled via the `augmentation:` field in the data config. Two presets are available:

- **default** — light augmentation: flips, 5-degree rotation, slight blur/noise, elastic transforms
- **strong** — adds brightness/contrast, CLAHE, coarse dropout, shift-scale-rotate, 15-degree rotation

Override in your experiment config:
```yaml
data_overrides:
  augmentation: strong   # default | strong
```

### Training Parameters

```yaml
training:
  max_epochs: 1000       # maximum training epochs
  patience: 200          # stop after N epochs without improvement
  min_delta: 0.0         # minimum metric change to count as improvement
  device: auto           # auto (GPU if available) | cuda | cpu | mps
  mixed_precision: false # enable AMP (requires CUDA)
  monitor: val_loss      # metric to monitor for early stopping / checkpointing
  monitor_mode: min      # min (lower is better) | max (higher is better)
```

---

## 3. Example Experiment Configs

### Baseline (no pretrained, no scheduler)
```yaml
data: data/merged
model: model/deeplabv3plus_resnet101

training:
  max_epochs: 1000
  patience: 100

optimizer:
  type: adam
  lr: 0.00005

loss:
  type: ce
  class_weights: [1.0, 1.0, 1.5, 1.0, 1.5, 1.5]
```

### Best Model (pretrained + cosine)
```yaml
data: data/merged
model: model/deeplabv3plus_resnet101_pretrained

data_overrides:
  batch_size: 2

training:
  max_epochs: 1000
  patience: 200

optimizer:
  type: adam
  lr: 0.0001

scheduler:
  type: cosine
  min_lr: 0.0000001

loss:
  type: ce
  class_weights: [1.0, 1.0, 1.5, 1.0, 1.5, 1.5]
```

### SMP UNet++ with Dice+CE and AdamW
```yaml
data: data/merged
model: model/smp_unetpp_effb4

data_overrides:
  batch_size: 4

training:
  max_epochs: 500
  patience: 100
  mixed_precision: true

optimizer:
  type: adamw
  lr: 0.0001
  weight_decay: 0.01

scheduler:
  type: cosine
  min_lr: 0.0000001

loss:
  type: dice_ce
  dice_weight: 0.5
  ce_weight: 0.5
  class_weights: [1.0, 1.0, 1.5, 1.0, 1.5, 1.5]
```

---

## 4. Evaluation

Evaluate a trained model on the validation set:

```bash
python evaluate.py \
    --checkpoint logs/runs/my-run/checkpoints/best.pt \
    --data-config configs/data/merged.yaml \
    --output logs/runs/my-run/eval_results.json
```

Output:
```
mIoU: 0.7860
F1 (macro): 0.8710
Pixel accuracy: 0.9450
```

The JSON output includes per-class IoU breakdown:
```json
{
  "miou": 0.786,
  "f1_macro": 0.871,
  "pixel_accuracy": 0.945,
  "per_class_iou": [0.95, 0.82, 0.71, 0.88, 0.65, 0.61],
  "per_class": {
    "background": 0.95,
    "silver": 0.82,
    "glass": 0.71,
    "silicon": 0.88,
    "void": 0.65,
    "interfacial_void": 0.61
  }
}
```

---

## 5. Inference on New Images

There is no built-in inference script yet. Use this snippet to run predictions on new images:

```python
import torch
import numpy as np
from PIL import Image
from src.models import create_model
from src.data.dataset import SEMDataset

# Load checkpoint
checkpoint = torch.load("logs/runs/my-run/checkpoints/best.pt", map_location="cpu", weights_only=False)
model_cfg = checkpoint["config"]["model"]
model = create_model(model_cfg)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load and preprocess image
image = np.array(Image.open("my_image.tif").convert("L"))  # grayscale
image = image.astype(np.float32) / 255.0
image = (image - 0.5) / 0.5  # normalize same as training
tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

# Predict
with torch.no_grad():
    output = model(tensor)
    if isinstance(output, dict):
        output = output["out"]
    pred = output.argmax(dim=1).squeeze().cpu().numpy()  # (H, W) class indices

# Class mapping
CLASSES = ["background", "silver", "glass", "silicon", "void", "interfacial_void"]
COLORS = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]

# Save colored prediction
color_mask = np.zeros((*pred.shape, 3), dtype=np.uint8)
for i, color in enumerate(COLORS):
    color_mask[pred == i] = color
Image.fromarray(color_mask).save("prediction.png")
```

---

## 6. SLURM (Pioneer HPC)

Example SLURM job script for training on the CWRU Pioneer cluster:

```bash
#!/bin/bash
#SBATCH --job-name=sem-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%j.out

module load python/3.11 cuda/12.1

cd /home/mis60/sem
source venv/bin/activate

python train.py \
    --experiment configs/experiment/autoresearch/ar_exp15_pretrained_cosine.yaml \
    --run-name pretrained-cosine-$SLURM_JOB_ID
```

Submit with:
```bash
sbatch train.slurm
```

---

## 7. Classes

The 6 segmentation classes and their recommended class weights:

| Index | Class | Weight | Description |
|-------|-------|--------|-------------|
| 0 | background | 1.0 | Background regions |
| 1 | silver | 1.0 | Silver contact fingers |
| 2 | glass | 1.5 | Glass frit layer |
| 3 | silicon | 1.0 | Silicon wafer |
| 4 | void | 1.5 | Voids / pores |
| 5 | interfacial_void | 1.5 | Interfacial voids at boundaries |

Higher weights (1.5) on glass, void, and interfacial_void compensate for their smaller area in the dataset.
