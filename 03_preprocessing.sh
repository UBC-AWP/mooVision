#!/bin/bash

#SBATCH --account=st-nina-1
#SBATCH --job-name=preprocess_splits
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
#SBATCH --output=logs/preprocess_%A_%a.out
#SBATCH --error=logs/preprocess_%A_%a.err

# ─── JOB ARRAY SPECIFICATION ──────────────────────────────────────────
# Spawns 8 independent worker tasks simultaneously (Task IDs 0 to 7)
#SBATCH --array=0-7

cd /scratch/st-nina-1/mooVision
source .env_sockeye

# Extract the specific paths for THIS parallel task instance
# (We pull from the arrays we defined in .env_sockeye)
INPUT_PATH="${PREPROCESS_YOLO_INPUT_PATHS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="${PREPROCESS_YOLO_OUTPUT_DIRS[$SLURM_ARRAY_TASK_ID]}"

echo "========================================================"
echo "Sockeye Array Engine Active" : EXECUTING Split Task $SLURM_ARRAY_TASK_ID
echo "Reading From CSV   : $INPUT_PATH"
echo "Writing Out To     : $OUTPUT_DIR"
echo "Compute Node       : $SLURM_NODENAME"
echo "========================================================"

# 3. PIPELINE EXECUTION VIA UV
# uv automatically synchronization virtual environment settings and steps down
uv run scripts/preprocessing/preprocessing_yolo.py \
    --input_path="$INPUT_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --skip=${SKIP} \
    --FORCE

echo "========================================================"
echo "SUCCESS: Task $SLURM_ARRAY_TASK_ID finished cleanly."
echo "========================================================"