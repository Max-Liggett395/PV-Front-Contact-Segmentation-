#!/bin/bash
# Submit all 3 models as parallel SLURM jobs.
# Each gets its own GPU and runs independently.
#
# Usage: bash scripts/train_all.sh

set -e

mkdir -p logs/slurm

echo "Submitting 3 training jobs in parallel..."

JOB1=$(sbatch --job-name=sem-unet \
    --parsable \
    scripts/train_model.slurm \
    configs/experiment/baseline.yaml \
    unet-merged-100ep)

JOB2=$(sbatch --job-name=sem-deeplabv3 \
    --parsable \
    scripts/train_model.slurm \
    configs/experiment/deeplabv3.yaml \
    deeplabv3-merged-100ep)

JOB3=$(sbatch --job-name=sem-deeplabv3plus \
    --parsable \
    scripts/train_model.slurm \
    configs/experiment/deeplabv3plus.yaml \
    deeplabv3plus-merged-100ep)

echo ""
echo "Submitted jobs:"
echo "  U-Net:        $JOB1"
echo "  DeepLabV3:    $JOB2"
echo "  DeepLabV3+:   $JOB3"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs at:      logs/slurm/"
echo "Checkpoints:  logs/runs/{unet,deeplabv3,deeplabv3plus}-merged-100ep/checkpoints/"
