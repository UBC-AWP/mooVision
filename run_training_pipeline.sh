#!/bin/bash

# 1. Submit the setup job and capture its SLURM Job ID number
#    (sbatch outputs text like "Submitted batch job 123456")
echo "Configuring Sockeye Repository"
cd /scratch/st-nina-1/mooVision
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

# 5. Parallel model application on test sets Array (80 tasks, waits for corresponding preprocessing tasks)
TEST_MSG=$(sbatch --dependency=afterok:$TRAIN_JOB_ID 05_model_testing.sh)
TEST_JOB_ID=$(echo "$TEST_MSG" | awk '{print $4}')

echo "Dispatched Parallel GPU Training Array: Job ID is $TEST_JOB_ID (Waiting on $TRAIN_JOB_ID)"

echo "--------------------------------------------------------"
echo "Full 8-Split Processing and Training Matrix Successfully Queued!"