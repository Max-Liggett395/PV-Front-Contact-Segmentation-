#!/bin/bash
#SBATCH --job-name=ar-arch
#SBATCH --output=autoresearch/experiments/track_architecture/logs/slurm/ar-arch-%j.out
#SBATCH --error=autoresearch/experiments/track_architecture/logs/slurm/ar-arch-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=rxf131
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu2080|gpu4v100|gpu2v100|gpul40s|gpu4090|gpu2h100|dgx
#SBATCH --time=08:00:00
#SBATCH --mem=32G

set -e

REPO_ROOT="$HOME/sem"
cd "$REPO_ROOT"

mkdir -p autoresearch/experiments/track_architecture/logs/slurm

echo "============================================"
echo "SEM Autoresearch — track_architecture — $(date)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "============================================"

module load GCCcore/12.3.0
module load Python/3.11.3-GCCcore-12.3.0
source "$REPO_ROOT/.venv/bin/activate"

export TRACK_NAME="track_architecture"

echo ">> Running prepare.py ..."
python autoresearch/prepare.py
echo ""

echo ">> Running track_architecture/train.py ..."
python autoresearch/experiments/track_architecture/train.py

echo ""
echo "============================================"
echo "Done — $(date)"
echo "============================================"
