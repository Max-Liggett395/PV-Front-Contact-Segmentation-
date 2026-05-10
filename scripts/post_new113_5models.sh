#!/bin/bash
# Post-training: assemble the comparison plot and final results table for the
# 5-model new_113 sweep. Run AFTER all 5 training jobs have finished
# (squeue -u $USER shows none of them pending/running).
#
# Usage: bash scripts/post_new113_5models.sh

set -e

module load GCCcore/12.3.0 2>/dev/null || true
module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || true
module load libffi/3.4.4-GCCcore-12.3.0 2>/dev/null || true
module load bzip2/1.0.8-GCCcore-12.3.0 2>/dev/null || true
source $HOME/sem/.venv/bin/activate
cd $HOME/sem

# Find the most recent slurm log for each run name.
declare -a NAMES=(
    smp-dlv3-new113
    smp-dlv3p-new113
    smp-unet-new113
    smp-unetpp-new113
    smp-segformer-new113
)
declare -a SHORT=(dlv3 dlv3p unet unetpp segformer)

PLOT_ARGS=()
TABLE_ARGS=()
for i in "${!NAMES[@]}"; do
    NAME="${NAMES[$i]}"
    LABEL="${SHORT[$i]}"
    LOG=$(ls -t logs/slurm/sem-${NAME}-*.out 2>/dev/null | head -1)
    if [ -z "$LOG" ]; then
        echo "  WARN: no slurm log for $NAME"
    else
        PLOT_ARGS+=("${LABEL}=${LOG}")
    fi
    TABLE_ARGS+=("${LABEL}=logs/runs/${NAME}")
done

echo "Plotting curves -> logs/runs/new113_5model_curves.png + viz/training_curves/<model>.png"
python scripts/plot_curves.py "${PLOT_ARGS[@]}" \
    --out logs/runs/new113_5model_curves.png \
    --per-model-dir viz/training_curves

echo ""
echo "Building results table -> logs/results_new113.md"
python scripts/build_results_table.py "${TABLE_ARGS[@]}" --out logs/results_new113.md
