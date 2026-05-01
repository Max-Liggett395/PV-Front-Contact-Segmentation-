#!/bin/bash
#SBATCH --job-name=sem-overlap
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --account=rxf131
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --mem=8G

module load GCCcore/12.3.0
module load Python/3.11.3-GCCcore-12.3.0
source $HOME/sem/.venv/bin/activate
cd $HOME/sem

python scripts/investigate_overlap.py
