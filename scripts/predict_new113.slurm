#!/bin/bash
#SBATCH --job-name=sem-predict
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=rxf131
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G

# Generate predicted-mask PNGs for the full new113 dataset using one model.
# Output: viz/<short_name>/masks/
#
# Usage:
#   sbatch scripts/predict_new113.slurm <experiment_config> <run_name>
# Example:
#   sbatch scripts/predict_new113.slurm \
#       configs/experiment/smp_segformer_new113.yaml smp-segformer-new113

EXPERIMENT=$1
RUN_NAME=$2

if [ -z "$EXPERIMENT" ] || [ -z "$RUN_NAME" ]; then
    echo "Usage: sbatch scripts/predict_new113.slurm <experiment_config> <run_name>"
    exit 1
fi

CHECKPOINT=logs/runs/${RUN_NAME}/checkpoints/best.pt
# Derive short name (smp-dlv3p-new113 -> dlv3p) for per-model viz layout.
SHORT_NAME="${RUN_NAME#smp-}"
SHORT_NAME="${SHORT_NAME%-new113}"
OUTPUT=viz/${SHORT_NAME}/masks

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

mkdir -p "$OUTPUT"

python predict.py \
    --checkpoint "$CHECKPOINT" \
    --experiment "$EXPERIMENT" \
    --input data/new_113/images \
    --output "$OUTPUT" \
    --overlay
