#!/bin/bash
#SBATCH --job-name=sem-smoothgrad
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

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

module load GCCcore/12.3.0
module load Python/3.11.3-GCCcore-12.3.0

source $HOME/sem/.venv/bin/activate

cd $HOME/sem

python scripts/smoothgrad.py \
    --checkpoint logs/runs/deeplabv3plus-d83/checkpoints/best.pt \
    --experiment configs/experiment/deeplabv3plus_dataset83.yaml \
    --data configs/data/merged.yaml \
    --stems-from predictions/error_maps_dlv3p_d83_on_merged/ranked.csv \
    --top-k 5 \
    --n-samples 16 \
    --sigma-frac 0.15 \
    --output predictions/smoothgrad_dlv3p_d83_on_merged
