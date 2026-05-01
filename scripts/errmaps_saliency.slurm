#!/bin/bash
#SBATCH --job-name=sem-errsal
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=rxf131
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
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

CKPT=logs/runs/deeplabv3plus-d83/checkpoints/best.pt
EXP=configs/experiment/deeplabv3plus_dataset83.yaml
DATA=configs/data/merged.yaml
ERR_OUT=predictions/error_maps_dlv3p_d83_on_merged
SAL_OUT=predictions/saliency_dlv3p_d83_on_merged

echo ">>> Step 1: error maps on full merged dataset"
python scripts/error_maps.py \
    --checkpoint $CKPT \
    --experiment $EXP \
    --data $DATA \
    --all-images \
    --output $ERR_OUT

echo ">>> Step 2: Grad-CAM on top-5 worst"
python scripts/saliency_maps.py \
    --checkpoint $CKPT \
    --experiment $EXP \
    --data $DATA \
    --stems-from $ERR_OUT/ranked.csv \
    --top-k 5 \
    --output $SAL_OUT
