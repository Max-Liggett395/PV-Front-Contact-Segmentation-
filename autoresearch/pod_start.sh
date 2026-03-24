#!/bin/bash
# AutoResearch pod startup script.
# Runs inside a RunPod pod. Expects environment variables:
#   EXPERIMENT_CONFIG - path to experiment YAML (relative to repo root)
#   RUN_NAME          - unique name for this run
#   MAX_EPOCHS        - max training epochs (default: 200)
#   PATIENCE          - early stopping patience (default: 50)
#   REPO_URL          - git repo URL to clone
#   REPO_BRANCH       - git branch to use (default: main)
#
# Data expected at: /runpod-volume/data/
# Results written to: /runpod-volume/results/

set -e

echo "=========================================="
echo "AutoResearch Pod Starting"
echo "  Run Name:   ${RUN_NAME}"
echo "  Experiment: ${EXPERIMENT_CONFIG}"
echo "  Max Epochs: ${MAX_EPOCHS:-200}"
echo "  Patience:   ${PATIENCE:-50}"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================="

WORK_DIR="/workspace/sem"
RESULTS_DIR="/runpod-volume/results"
DATA_DIR="/runpod-volume/data"

# Clone repo
if [ -n "$REPO_URL" ]; then
    echo "Cloning repo..."
    git clone --depth 1 --branch "${REPO_BRANCH:-main}" "$REPO_URL" "$WORK_DIR"
else
    # Fallback: copy from volume
    echo "Copying project from volume..."
    cp -r /runpod-volume/project "$WORK_DIR"
fi

cd "$WORK_DIR"

# Symlink data from volume into expected locations
echo "Linking data..."
rm -rf data
ln -s "$DATA_DIR" data

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Create results directory
mkdir -p "$RESULTS_DIR"

# Run experiment
echo "Starting training..."
python -m autoresearch.run_experiment \
    --experiment "${EXPERIMENT_CONFIG}" \
    --run-name "${RUN_NAME}" \
    --max-epochs "${MAX_EPOCHS:-200}" \
    --patience "${PATIENCE:-50}" \
    --output-dir "$RESULTS_DIR"

echo "Training complete. Results at ${RESULTS_DIR}/${RUN_NAME}.json"

# Also copy best checkpoint to results
CKPT_DIR="logs/runs/${RUN_NAME}/checkpoints"
if [ -f "${CKPT_DIR}/best.pt" ]; then
    mkdir -p "${RESULTS_DIR}/checkpoints/${RUN_NAME}"
    cp "${CKPT_DIR}/best.pt" "${RESULTS_DIR}/checkpoints/${RUN_NAME}/best.pt"
    echo "Checkpoint saved to ${RESULTS_DIR}/checkpoints/${RUN_NAME}/best.pt"
fi

echo "Pod work complete. Stopping..."

# Self-terminate via RunPod API if available
if [ -n "$RUNPOD_API_KEY" ] && [ -n "$RUNPOD_POD_ID" ]; then
    curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"mutation { podStop(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) { id } }\"}" \
        > /dev/null 2>&1 || true
    echo "Pod stop signal sent."
fi
