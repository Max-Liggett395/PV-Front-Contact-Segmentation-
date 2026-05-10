#!/bin/bash
# Submit 5 parallel training jobs (DLv3, DLv3+, U-Net, U-Net++, Segformer) on
# the new_113 dataset, each up to 250 epochs with early stopping.
# Each training job has a chained saliency job (--dependency=afterok) that
# runs SmoothGrad on 5 fixed val stems once training finishes.
#
# Usage: bash scripts/train_new113_5models.sh

set -e

mkdir -p logs/slurm

declare -a NAMES=(
    smp-dlv3-new113
    smp-dlv3p-new113
    smp-unet-new113
    smp-unetpp-new113
    smp-segformer-new113
)
declare -a EXPS=(
    configs/experiment/smp_deeplabv3_new113.yaml
    configs/experiment/smp_deeplabv3plus_new113.yaml
    configs/experiment/smp_unet_new113.yaml
    configs/experiment/smp_unetpp_new113.yaml
    configs/experiment/smp_segformer_new113.yaml
)

echo "Submitting 5 training jobs (250 epochs, early stop patience=25 on val_loss)"
echo "and 5 chained saliency jobs (afterok dependency)."
echo ""

for i in "${!NAMES[@]}"; do
    NAME="${NAMES[$i]}"
    EXP="${EXPS[$i]}"

    TRAIN_JOB=$(sbatch --job-name="sem-${NAME}" --parsable \
        scripts/train_model.slurm "$EXP" "$NAME")

    SAL_JOB=$(sbatch --job-name="sem-sal-${NAME}" --parsable \
        --dependency=afterok:"$TRAIN_JOB" \
        scripts/saliency_new113.slurm "$EXP" "$NAME")

    printf "  %-22s  train=%s  saliency=%s (afterok:%s)\n" \
        "$NAME" "$TRAIN_JOB" "$SAL_JOB" "$TRAIN_JOB"
done

echo ""
echo "Monitor with:    squeue -u \$USER"
echo "Training logs:   logs/slurm/sem-smp-*.out"
echo "Run dirs:        logs/runs/<run_name>/"
echo "After training, build plots and table with:"
echo "  bash scripts/post_new113_5models.sh"
