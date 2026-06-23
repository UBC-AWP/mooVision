#!/bin/bash
#SBATCH --account=st-nina-1-gpu
#SBATCH --job-name=baseline_inference
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --array=0-7
#SBATCH --output=logs/baseline_%A_%a.out
#SBATCH --error=logs/baseline_%A_%a.err

# ── Map array index → split ───────────────────────────────────────────────────
SPLIT_IDX=${SLURM_ARRAY_TASK_ID}

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

CSV_PATH="data/processed/${DATA_SPLITS[$SPLIT_IDX]}/test.csv"

echo "========================================================"
echo "TASK ${SLURM_ARRAY_TASK_ID}: split=${DATA_SPLITS[$SPLIT_IDX]}"
echo "  CSV: ${CSV_PATH}"
echo "========================================================"

cd /scratch/st-nina-1/mooVision
mkdir -p logs
source .env_sockeye

# ── Per-task isolated config/cache dirs ──────────────────────────────────────
export YOLO_CONFIG_DIR="${USER_SCRATCH}/.config/ultralytics_job_${SLURM_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"
export MPLCONFIGDIR="${USER_SCRATCH}/.config/matplotlib_job_${SLURM_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"
export FONTCONFIG_PATH="${USER_SCRATCH}/.config/font_job_${SLURM_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"
export XDG_CACHE_HOME="${USER_SCRATCH}/.cache/job_${SLURM_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR" "$FONTCONFIG_PATH" "$XDG_CACHE_HOME/fontconfig"

uv run scripts/models/baseline/baseline.py \
    --csv "$CSV_PATH" \
    --frame_skip 10

echo "========================================================"
echo "TASK ${SLURM_ARRAY_TASK_ID} COMPLETE."
echo "========================================================"