#!/bin/bash

# 1. Submit the setup job and capture its SLURM Job ID number
#    (sbatch outputs text like "Submitted batch job 123456")
echo "Configuring Sockeye Repository"
cd /scratch/st-nina-1/mooVision/scripts_sockeye
bash 01_setup.sh

# 2. Submit the read and split data job and capture its SLURM Job ID number
READ_DATA_MSG=$(sbatch 02_read_and_split_data.sh)
READ_DATA_JOB_ID=$(echo "$READ_DATA_MSG" | awk '{print $4}')

echo "Dispatched Read and Split Data Script: Job ID is $READ_DATA_JOB_ID"

# 3. Submit the Job Array, but tell it to WAIT for the Setup Job ID to finish perfectly
PREPROCESS_MSG=$(sbatch --dependency=afterok:$READ_DATA_JOB_ID 03_preprocessing.sh)
PREPROCESS_JOB_ID=$(echo "$PREPROCESS_MSG" | awk '{print $4}')

echo "Dispatched Processing Job Array: Job ID is $PREPROCESS_JOB_ID (Waiting on $READ_DATA_JOB_ID)"

# 4. Parallel Training GPU Array (8 tasks, waits for corresponding preprocessing tasks)
TRAIN_MSG=$(sbatch --dependency=afterok:$PREPROCESS_JOB_ID 04_train_yolo.sh)
TRAIN_JOB_ID=$(echo "$TRAIN_MSG" | awk '{print $4}')

echo "Dispatched Parallel GPU Training Array: Job ID is $TRAIN_JOB_ID (Waiting on $PREPROCESS_JOB_ID)"

# 5. Baseline inference (8 tasks, waits for preprocessing — does not need training)
BASELINE_INFER_MSG=$(sbatch --dependency=afterok:$TRAIN_JOB_ID 05_baseline_testing.sh)
BASELINE_INFER_JOB_ID=$(echo "$BASELINE_INFER_MSG" | awk '{print $4}')
 
echo "Dispatched Baseline Inference Array: Job ID is $BASELINE_INFER_JOB_ID (Waiting on $TRAIN_JOB_ID)"
 
# 6. YOLO model inference on test sets (80 tasks, waits for training)
YOLO_INFER_MSG=$(sbatch --dependency=afterok:$TRAIN_JOB_ID 06_yolo_testing.sh)
YOLO_INFER_JOB_ID=$(echo "$YOLO_INFER_MSG" | awk '{print $4}')
 
echo "Dispatched YOLO Inference Array: Job ID is $YOLO_INFER_JOB_ID (Waiting on $TRAIN_JOB_ID)"
 
# 7. Baseline evaluation (8 tasks, waits for baseline inference)
BASELINE_EVAL_MSG=$(sbatch --dependency=afterok:$BASELINE_INFER_JOB_ID 07_baseline_evaluation.sh)
BASELINE_EVAL_JOB_ID=$(echo "$BASELINE_EVAL_MSG" | awk '{print $4}')
 
echo "Dispatched Baseline Evaluation Array: Job ID is $BASELINE_EVAL_JOB_ID (Waiting on $BASELINE_INFER_JOB_ID)"
 
# 8. YOLO evaluation (16 tasks: 8 splits × 2 pipelines, waits for YOLO inference)
YOLO_EVAL_MSG=$(sbatch --dependency=afterok:$YOLO_INFER_JOB_ID 08_yolo_evaluation.sh)
YOLO_EVAL_JOB_ID=$(echo "$YOLO_EVAL_MSG" | awk '{print $4}')
 
echo "Dispatched YOLO Evaluation Array: Job ID is $YOLO_EVAL_JOB_ID (Waiting on $YOLO_INFER_JOB_ID)"
 
echo "--------------------------------------------------------"
echo "Full Pipeline Successfully Queued!"
echo ""
echo "Job dependency chain:"
echo "  $READ_DATA_JOB_ID (read+split)"
echo "    └── $PREPROCESS_JOB_ID (preprocess)"
echo "          ├── $TRAIN_JOB_ID (train YOLO)"
echo "          │     └── $YOLO_INFER_JOB_ID (YOLO inference)"
echo "          │           └── $YOLO_EVAL_JOB_ID (YOLO evaluation)"
echo "          └── $BASELINE_INFER_JOB_ID (baseline inference)"
echo "                └── $BASELINE_EVAL_JOB_ID (baseline evaluation)"
 