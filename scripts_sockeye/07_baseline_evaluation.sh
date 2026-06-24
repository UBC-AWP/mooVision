#!/bin/bash
#SBATCH --job-name=baseline-eval
#SBATCH --account=st-nina-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/baseline_eval_%A_%a.out
#SBATCH --error=logs/baseline_eval_%A_%a.err

# ==============================================================================
# ENVIRONMENT
# ==============================================================================
cd /scratch/st-nina-1/mooVision
source .env_sockeye

DATA_SPLITS=(
    "random"
    "day_based"
    "pen_based/pen_2"
    "pen_based/pen_3"
    "pen_based/pen_5"
    "period_based/POSTWEAN"
    "period_based/PREWEAN"
    "period_based/WEAN"
)

SPLIT="${DATA_SPLITS[$SLURM_ARRAY_TASK_ID]}"
SPLIT_LABEL="${SPLIT//\//_}"

echo "======================================================"
echo "Task ${SLURM_ARRAY_TASK_ID}: split=${SPLIT}"
echo "======================================================"

# ==============================================================================
# PATHS
# ==============================================================================
PREDICTIONS="${ROOT_DIR}/results/metadata/${SPLIT}/baseline/"
GROUND_TRUTH="${ROOT_DIR}/data/processed/processed_clips_index.csv"
OUTPUT_DIR="${ROOT_DIR}/results/evaluation"
OUTPUT_FILE="${OUTPUT_DIR}/evaluation_report_${SPLIT_LABEL}_baseline.json"
LABELLED_CLIPS_DIR="cross_sucking_labelled"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -d "${PREDICTIONS}" ]]; then
    echo "ERROR: predictions directory not found: ${PREDICTIONS}" >&2
    exit 1
fi

# ==============================================================================
# RUN
# ==============================================================================
uv run python scripts/evaluation/evaluation.py \
    --predictions "${PREDICTIONS}" \
    --ground_truth "${GROUND_TRUTH}" \
    --output "${OUTPUT_FILE}" \
    --labelled_clips_dir "${LABELLED_CLIPS_DIR}"

echo "Done → ${OUTPUT_FILE}"