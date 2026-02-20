#!/bin/bash
#SBATCH --job-name=sem-train
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=sdle       # SDLE lab dedicated GPU node (also: gpu, cgpu)
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G

# Usage: sbatch --job-name=sem-unet scripts/train_model.slurm configs/experiment/baseline.yaml unet

EXPERIMENT=$1
RUN_NAME=$2

if [ -z "$EXPERIMENT" ] || [ -z "$RUN_NAME" ]; then
    echo "Usage: sbatch scripts/train_model.slurm <experiment_config> <run_name>"
    exit 1
fi

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Experiment: $EXPERIMENT"
echo "Run Name: $RUN_NAME"
echo "=========================================="

# Load modules (adjust versions to what's available: `module avail`)
# module load cuda/12.1
# module load python/3.10

# Activate environment
source .venv/bin/activate

# Create log directories
mkdir -p logs/slurm

# Print GPU info
nvidia-smi

# Train
python train.py --experiment "$EXPERIMENT" --run-name "$RUN_NAME"

echo "Training complete for $RUN_NAME"
