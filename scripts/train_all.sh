#!/bin/bash
# Submit all 9 models (3 models x 3 datasets) as parallel SLURM jobs.
# Each gets its own GPU and runs independently.
#
# Usage: bash scripts/train_all.sh

set -e

mkdir -p logs/slurm

echo "Submitting 9 training jobs (3 models x 3 datasets)..."
echo ""

# --- Dataset 83 (83 original NPY masks) ---

JOB1=$(sbatch --job-name=sem-unet-d83 \
    --parsable \
    scripts/train_model.slurm \
    configs/experiment/unet_dataset83.yaml \
    unet-d83-1000ep)

JOB2=$(sbatch --job-name=sem-dlv3-d83 \
    --parsable \
    scripts/train_model.slurm \
    configs/experiment/deeplabv3_dataset83.yaml \
    deeplabv3-d83-1000ep)

JOB3=$(sbatch --job-name=sem-dlv3p-d83 \
    --parsable \
    scripts/train_model.slurm \
    configs/experiment/deeplabv3plus_dataset83.yaml \
    deeplabv3plus-d83-1000ep)

# --- Dataset 116 (116 JSON-annotated masks) ---

JOB4=$(sbatch --job-name=sem-unet-d116 \
    --parsable \
    scripts/train_model.slurm \
    configs/experiment/unet_dataset116.yaml \
    unet-d116-1000ep)

JOB5=$(sbatch --job-name=sem-dlv3-d116 \
    --parsable \
    scripts/train_model.slurm \
    configs/experiment/deeplabv3_dataset116.yaml \
    deeplabv3-d116-1000ep)

JOB6=$(sbatch --job-name=sem-dlv3p-d116 \
    --parsable \
    scripts/train_model.slurm \
    configs/experiment/deeplabv3plus_dataset116.yaml \
    deeplabv3plus-d116-1000ep)

# --- Merged dataset (130 images, all masks) ---

JOB7=$(sbatch --job-name=sem-unet-merged \
    --parsable \
    scripts/train_model.slurm \
    configs/experiment/baseline.yaml \
    unet-merged-1000ep)

JOB8=$(sbatch --job-name=sem-dlv3-merged \
    --parsable \
    scripts/train_model.slurm \
    configs/experiment/deeplabv3.yaml \
    deeplabv3-merged-1000ep)

JOB9=$(sbatch --job-name=sem-dlv3p-merged \
    --parsable \
    scripts/train_model.slurm \
    configs/experiment/deeplabv3plus.yaml \
    deeplabv3plus-merged-1000ep)

echo "Submitted jobs:"
echo ""
echo "  Dataset 83:"
echo "    U-Net:        $JOB1"
echo "    DeepLabV3:    $JOB2"
echo "    DeepLabV3+:   $JOB3"
echo ""
echo "  Dataset 116:"
echo "    U-Net:        $JOB4"
echo "    DeepLabV3:    $JOB5"
echo "    DeepLabV3+:   $JOB6"
echo ""
echo "  Merged (130):"
echo "    U-Net:        $JOB7"
echo "    DeepLabV3:    $JOB8"
echo "    DeepLabV3+:   $JOB9"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs at:      logs/slurm/"
