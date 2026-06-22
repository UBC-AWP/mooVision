#!/bin/bash
#SBATCH --job-name=model-eval
#SBATCH --account=st-nina-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-15                      # 8 splits × 2 pipelines = 16 tasks
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err

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

PIPELINES=("yolo" "seq-nms")

# ==============================================================================
# MAP TASK ID → (SPLIT, PIPELINE)
# Task layout: tasks 0-7 = yolo splits 0-7, tasks 8-15 = seq-nms splits 0-7
# ==============================================================================
N_SPLITS=${#DATA_SPLITS[@]}                          # 8
PIPELINE_IDX=$(( SLURM_ARRAY_TASK_ID / N_SPLITS ))  # 0 or 1
SPLIT_IDX=$(( SLURM_ARRAY_TASK_ID % N_SPLITS ))     # 0–7

SPLIT="${DATA_SPLITS[$SPLIT_IDX]}"
PIPELINE="${PIPELINES[$PIPELINE_IDX]}"

# Flatten split path for use in output filename  (e.g. pen_based/pen_2 → pen_based_pen_2)
SPLIT_LABEL="${SPLIT//\//_}"

echo "======================================================"
echo "Task ${SLURM_ARRAY_TASK_ID}: split=${SPLIT}  pipeline=${PIPELINE}"
echo "======================================================"

# ==============================================================================
# PATHS  (all relative to ROOT_DIR, which .env_sockeye sets to USER_SCRATCH)
# ==============================================================================
PREDICTIONS="${ROOT_DIR}/results/metadata/${SPLIT}/${PIPELINE}/"
GROUND_TRUTH="${ROOT_DIR}/data/processed/processed_clips_index.csv"
OUTPUT_DIR="${ROOT_DIR}/results/evaluation"
OUTPUT_FILE="${OUTPUT_DIR}/evaluation_report_${SPLIT_LABEL}_${PIPELINE//-/_}.json"
LABELLED_CLIPS_DIR="cross_sucking_labelled"

mkdir -p "${OUTPUT_DIR}"

# Sanity-check that predictions dir exists before launching
if [[ ! -d "${PREDICTIONS}" ]]; then
    echo "ERROR: predictions directory not found: ${PREDICTIONS}" >&2
    exit 1
fi

# ==============================================================================
# RUN
# ==============================================================================
uv run python scripts/evaluation.py \
    --predictions "${PREDICTIONS}" \
    --ground_truth "${GROUND_TRUTH}" \
    --output "${OUTPUT_FILE}" \
    --labelled_clips_dir "${LABELLED_CLIPS_DIR}"

echo "Done → ${OUTPUT_FILE}"