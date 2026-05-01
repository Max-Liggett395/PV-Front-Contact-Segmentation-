#!/bin/bash
#SBATCH --job-name=ar-smp
#SBATCH --output=autoresearch/experiments/track_smp_models/logs/slurm/ar-smp-%j.out
#SBATCH --error=autoresearch/experiments/track_smp_models/logs/slurm/ar-smp-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=rxf131
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu2v100|gpu4v100|gpu2080|gpul40s|gpu4090|gpu2h100
#SBATCH --time=08:00:00
#SBATCH --mem=32G

set -e

REPO_ROOT="$HOME/sem"
cd "$REPO_ROOT"

mkdir -p autoresearch/experiments/track_smp_models/logs/slurm

echo "============================================"
echo "SEM Autoresearch — track_smp_models — $(date)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "============================================"

module load GCCcore/12.3.0
module load Python/3.11.3-GCCcore-12.3.0
source "$REPO_ROOT/.venv/bin/activate"

export TRACK_NAME="track_smp_models"

echo ">> Running prepare.py ..."
python autoresearch/prepare.py
echo ""

echo ">> Running track_smp_models/train.py ..."
python autoresearch/experiments/track_smp_models/train.py

echo ""
echo "============================================"
echo "Done — $(date)"
echo "============================================"
