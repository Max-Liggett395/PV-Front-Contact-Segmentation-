#!/bin/bash
#SBATCH --job-name=sem-saliency
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=rxf131
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G

# Usage:
#   sbatch scripts/saliency_new113.slurm <experiment_config> <run_name>
# Runs SmoothGrad on a fixed list of 5 representative new_113 val stems
# using the run's best.pt checkpoint.

EXPERIMENT=$1
RUN_NAME=$2

if [ -z "$EXPERIMENT" ] || [ -z "$RUN_NAME" ]; then
    echo "Usage: sbatch scripts/saliency_new113.slurm <experiment_config> <run_name>"
    exit 1
fi

CHECKPOINT=logs/runs/${RUN_NAME}/checkpoints/best.pt
OUTPUT=viz/saliency_${RUN_NAME}

echo "=========================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "GPU:        $CUDA_VISIBLE_DEVICES"
echo "Experiment: $EXPERIMENT"
echo "Run:        $RUN_NAME"
echo "Checkpoint: $CHECKPOINT"
echo "Output:     $OUTPUT"
echo "=========================================="

module load GCCcore/12.3.0
module load Python/3.11.3-GCCcore-12.3.0

source $HOME/sem/.venv/bin/activate
cd $HOME/sem

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint $CHECKPOINT not found"
    exit 2
fi

# 5 fixed val-split stems (new_113 val set under seed=42, train_split=0.85).
# Picked to span distinct sample families: CG, NPERT, SLTE, VCAD, gLASS.
python scripts/smoothgrad.py \
    --checkpoint "$CHECKPOINT" \
    --experiment "$EXPERIMENT" \
    --data configs/data/new_113.yaml \
    --stems \
        4-10-24_CG002_BJ3_5 \
        NPERT.CG006.BL.c4.1 \
        SLTE.C17BB11.w3.2 \
        VCAD.C33BB21.c3.1 \
        gLASSgLASS.GG1.w1.1 \
    --n-samples 16 \
    --sigma-frac 0.15 \
    --output "$OUTPUT"

echo "Saliency complete: $OUTPUT"
