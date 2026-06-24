#!/bin/bash
#SBATCH --account=st-nina-1-gpu
#SBATCH --job-name=yolo_training
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=32:00:00
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --array=0-8

export PYTHONUNBUFFERED=1
cd /scratch/st-nina-1/mooVision
source .env_sockeye

# Identify the unique dataset YAML config for this specific thread
TAR_PATH="${YOLO_TAR_FILES[$SLURM_ARRAY_TASK_ID]}"

# Assign a completely isolated checkpoints/weights folder for this split model
SPLIT_NUM=$((SLURM_ARRAY_TASK_ID + 1))
UNIQUE_RUN_NAME="split_${SPLIT_NUM}_model"

TARGET_PROJECT_DIR="${YOLO_SCRATCH_OUTPUT}"
CHECKPOINT_PATH="${TARGET_PROJECT_DIR}/${UNIQUE_RUN_NAME}/weights/last.pt"

echo "========================================================"
echo "Sockeye GPU Array Engine Active"
echo "Executing Split Task : $SLURM_ARRAY_TASK_ID (Split #$SPLIT_NUM)"
echo "Target Dataset YAML  : $TAR_PATH"
echo "Isolated Run Name    : $UNIQUE_RUN_NAME"
echo "Compute Node Assigned: $SLURM_NODENAME"
echo "========================================================"

# --- Fix Read-Only & Permission Warnings (Job ID + Array Safe) ---
# Format will look like: .../.config/ultralytics_job_123456_task_1
export YOLO_CONFIG_DIR="${USER_SCRATCH}/.config/ultralytics_job_${SLURM_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"
export YOLOV8_CONFIG_DIR="${USER_SCRATCH}/.config/ultralytics_job_${SLURM_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"

# Isolate Matplotlib and Font caches perfectly as well
export MPLCONFIGDIR="${USER_SCRATCH}/.config/matplotlib_job_${SLURM_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"
export FONTCONFIG_PATH="${USER_SCRATCH}/.config/font_job_${SLURM_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}"

# Dynamically generate the unique structure before Python boots up
mkdir -p "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR" "$FONTCONFIG_PATH"
mkdir -p "$TARGET_PROJECT_DIR"

echo "Starting fresh training initialization..."
# Ensure your training_yolo.py script passes this project/name tag 
# down to the Ultralytics model.train() function!
uv run --frozen --offline python scripts/training/training_yolo.py \
    --dataset="$TAR_PATH" \
    --project="$TARGET_PROJECT_DIR" \
    --name="$UNIQUE_RUN_NAME" \
    --device="0" \
    --workers=8 \
    --weights_dir=$WEIGHTS_DIR \
    --model=26 \
    --model_size=m \
    --batch=64 \
    --epochs=100 \
    --patience=40 

# WIP: Add resume training logic

# 3. Handle Resume Checkpoint Logic vs Fresh Start
# if [ -f "$CHECKPOINT_PATH" ]; then
#     echo "FOUND RECOVERY CHECKPOINT. Resuming training..."
#     uv run scripts/training/training_yolo.py \
#         --resume_path="$CHECKPOINT_PATH" \
#         --device="0"
# else
#     echo "Starting fresh training initialization..."
#     # Ensure your training_yolo.py script passes this project/name tag 
#     # down to the Ultralytics model.train() function!
#     uv run --frozen --offline python scripts/training/training_yolo.py \
#         --yaml_path="$YAML_PATH" \
#         --project="$TARGET_PROJECT_DIR" \
#         --name="$UNIQUE_RUN_NAME" \
#         --device="[0,1]" \
#         --workers=8 \
#         --weights_dir=$WEIGHTS_DIR \
#         --model=26 \
#         --model_size=m \
#         --batch=64 \
#
# fi

echo "========================================================"
echo "SUCCESS: Split #$SPLIT_NUM Model Training Complete."
echo "========================================================"