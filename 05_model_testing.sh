#!/bin/bash
#SBATCH --account=st-nina-1-gpu
#SBATCH --job-name=model_inference
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --array=0-79
#SBATCH --output=logs/infer_%A_%a.out
#SBATCH --error=logs/infer_%A_%a.err

# ── Map flat index → (model+test pair, chunk) ────────────────────────────────
PAIR_IDX=$(( SLURM_ARRAY_TASK_ID / 10 ))
CHUNK_IDX=$(( SLURM_ARRAY_TASK_ID % 10 ))

# ── Set overwrite behaviour ───────────────────────────────────────────────────
OVERWRITE=${OVERWRITE:-false}

# ── Model+test pairs (split_1 = random, split_2 = day_based, etc.) ───────────
MODEL_NAMES=(
    "split_2_model"
    "split_3_model"
    "split_4_model"
    "split_5_model"
    "split_6_model"
    "split_7_model"
    "split_8_model"
    "split_9_model"
)

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

MODEL_PATH="/scratch/st-nina-1/moovision/yolo_training_runs/${MODEL_NAMES[$PAIR_IDX]}/weights/best.pt"
DATA_PATH="data/processed/${DATA_SPLITS[$PAIR_IDX]}"/test.csv

echo "========================================================"
echo "TASK ${SLURM_ARRAY_TASK_ID}: pair=${PAIR_IDX} chunk=${CHUNK_IDX} chunk_pct=0.10"
echo "  MODEL: ${MODEL_PATH}"
echo "  DATA:  ${DATA_PATH}"
echo "========================================================"

cd /scratch/st-nina-1/mooVision
mkdir -p logs
source .env_sockeye

# --- Fix Read-Only & Permission Warnings (Job ID + Array Safe) ---
# Format will look like: .../.config/ultralytics_job_123456_task_1
export YOLO_CONFIG_DIR="${USER_SCRATCH}/.config/ultralytics_job_${SLURM_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"
export YOLOV8_CONFIG_DIR="${USER_SCRATCH}/.config/ultralytics_job_${SLURM_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"

# Isolate Matplotlib and Font caches perfectly as well
export MPLCONFIGDIR="${USER_SCRATCH}/.config/matplotlib_job_${SLURM_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"
export FONTCONFIG_PATH="${USER_SCRATCH}/.config/font_job_${SLURM_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"

# Dynamically generate the unique structure before Python boots up
mkdir -p "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR" "$FONTCONFIG_PATH"

OVERWRITE_FLAG=""
if [ "$OVERWRITE" = "true" ]; then
    OVERWRITE_FLAG="--overwrite"
fi

uv run scripts/run-testing-2.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --chunk "$CHUNK_IDX" \
    --chunk_pct 0.10 \
    $OVERWRITE_FLAG

echo "========================================================"
echo "TASK ${SLURM_ARRAY_TASK_ID} COMPLETE."
echo "========================================================"