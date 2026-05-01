#!/bin/bash
#SBATCH --job-name=autoresearch
#SBATCH --output=logs/slurm/autoresearch-%j.out
#SBATCH --error=logs/slurm/autoresearch-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=rxf131
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G

# ============================================================
# SEM Segmentation Autoresearch — Runner
#
# Usage (SLURM — run from repo root):
#   sbatch autoresearch/run.sh
# ============================================================

set -e

# Always work from the repo root
REPO_ROOT="$HOME/sem"
cd "$REPO_ROOT"

mkdir -p logs/slurm
mkdir -p autoresearch/logs

echo "============================================"
echo "SEM Autoresearch — $(date)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "============================================"

# Load modules (match cluster Python version)
module load GCCcore/12.3.0
module load Python/3.11.3-GCCcore-12.3.0

# Activate environment
source "$REPO_ROOT/.venv/bin/activate"

# Step 1: Verify data
echo ""
echo ">> Running prepare.py ..."
python autoresearch/prepare.py
echo ""

# Step 2: Run training
echo ">> Running train.py ..."
python autoresearch/train.py

echo ""
echo "============================================"
echo "Done — $(date)"
echo "Results logged to autoresearch/logs/results.jsonl"
echo "============================================"
