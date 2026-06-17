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

# ── Model+test pairs (split_1 = random, split_2 = day_based, etc.) ───────────
MODEL_NAMES=(
    "split_1_model"
    "split_2_model"
    "split_3_model"
    "split_4_model"
    "split_5_model"
    "split_6_model"
    "split_7_model"
    "split_8_model"
)

DATA_SPLITS=(
    "data/processed/random/test.csv"
    "data/processed/day_based/test.csv"
    "data/processed/pen_based/pen_2/test.csv"
    "data/processed/pen_based/pen_3/test.csv"
    "data/processed/pen_based/pen_5/test.csv"
    "data/processed/period_based/POSTWEAN/test.csv"
    "data/processed/period_based/PREWEAN/test.csv"
    "data/processed/period_based/WEAN/test.csv"
)

MODEL_PATH="/scratch/st-nina-1/moovision/yolo_training_runs/${MODEL_NAMES[$PAIR_IDX]}/weights/best.pt"
DATA_PATH="${DATA_SPLITS[$PAIR_IDX]}"

echo "========================================================"
echo "TASK ${SLURM_ARRAY_TASK_ID}: pair=${PAIR_IDX} chunk=${CHUNK_IDX} chunk_pct=0.10"
echo "  MODEL: ${MODEL_PATH}"
echo "  DATA:  ${DATA_PATH}"
echo "========================================================"

cd /scratch/st-nina-1/mooVision
mkdir -p logs
source .env_sockeye

uv run scripts/run-testing-2.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --chunk "$CHUNK_IDX" \
    --chunk_pct 0.10

echo "========================================================"
echo "TASK ${SLURM_ARRAY_TASK_ID} COMPLETE."
echo "========================================================"