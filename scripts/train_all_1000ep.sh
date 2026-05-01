#!/bin/bash
# Submit all 9 training jobs (3 models x 3 datasets) in parallel.
# Each gets its own GPU and runs independently for up to 1000 epochs.
#
# Usage: bash scripts/train_all_1000ep.sh

set -e

mkdir -p logs/slurm

echo "Submitting 9 training jobs (3 models x 3 datasets, 1000 epochs)..."
echo ""

# U-Net (baseline)
JOB1=$(sbatch --job-name=sem-unet-merged \
    --parsable \
    scripts/train_model.slurm \
    configs/experiment/baseline_merged.yaml \
    unet-merged-1000ep)

JOB2=$(sbatch --job-name=sem-unet-old83 \
    --parsable \
    scripts/train_model.slurm \
    configs/experiment/baseline_old83.yaml \
    unet-old83-1000ep)

JOB3=$(sbatch --job-name=sem-unet-new116 \
    --parsable \
    scripts/train_model.slurm \
    configs/experiment/baseline_new116.yaml \
    unet-new116-1000ep)

# DeepLabV3
JOB4=$(sbatch --job-name=sem-dlv3-merged \
    --parsable \
    scripts/train_model.slurm \
    configs/experiment/deeplabv3_merged.yaml \
    deeplabv3-merged-1000ep)

JOB5=$(sbatch --job-name=sem-dlv3-old83 \
    --parsable \
    scripts/train_model.slurm \
    configs/experiment/deeplabv3_old83.yaml \
    deeplabv3-old83-1000ep)

JOB6=$(sbatch --job-name=sem-dlv3-new116 \
    --parsable \
    scripts/train_model.slurm \
    configs/experiment/deeplabv3_new116.yaml \
    deeplabv3-new116-1000ep)

# DeepLabV3+
JOB7=$(sbatch --job-name=sem-dlv3p-merged \
    --parsable \
    scripts/train_model.slurm \
    configs/experiment/deeplabv3plus_merged.yaml \
    deeplabv3plus-merged-1000ep)

JOB8=$(sbatch --job-name=sem-dlv3p-old83 \
    --parsable \
    scripts/train_model.slurm \
    configs/experiment/deeplabv3plus_old83.yaml \
    deeplabv3plus-old83-1000ep)

JOB9=$(sbatch --job-name=sem-dlv3p-new116 \
    --parsable \
    scripts/train_model.slurm \
    configs/experiment/deeplabv3plus_new116.yaml \
    deeplabv3plus-new116-1000ep)

echo "Submitted jobs:"
echo "  U-Net    merged:  $JOB1"
echo "  U-Net    old_83:  $JOB2"
echo "  U-Net    new_116: $JOB3"
echo "  DLV3     merged:  $JOB4"
echo "  DLV3     old_83:  $JOB5"
echo "  DLV3     new_116: $JOB6"
echo "  DLV3+    merged:  $JOB7"
echo "  DLV3+    old_83:  $JOB8"
echo "  DLV3+    new_116: $JOB9"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs at:      logs/slurm/"
echo "History CSVs: logs/runs/*/history.csv"
echo "Checkpoints:  logs/runs/*/checkpoints/{best,latest}.pt"
