#!/bin/bash

# 1. Submit the setup job and capture its SLURM Job ID number
#    (sbatch outputs text like "Submitted batch job 123456")
SETUP_MSG=$(sbatch 01_setup.sh)
SETUP_JOB_ID=$(echo "$SETUP_MSG" | awk '{print $4}')

echo "Dispatched Setup Script: Job ID is $SETUP_JOB_ID"

# 2. Submit the read and split data job and capture its SLURM Job ID number
READ_DATA_MSG=$(sbatch --dependency=afterok:$SETUP_JOB_ID 02_read_and_split_data.sh)
READ_DATA_JOB_ID=$(echo "$READ_DATA_MSG" | awk '{print $4}')

echo "Dispatched Read and Split Data Script: Job ID is $READ_DATA_JOB_ID (Waiting on $SETUP_JOB_ID)"

# 2. Submit the Job Array, but tell it to WAIT for the Setup Job ID to finish perfectly
PREPROCESS_MSG=$(sbatch --dependency=afterok:$READ_DATA_JOB_ID 03_preprocessing.sh)
PREPROCESS_MSG_JOB_ID=$(echo "$PREPROCESS_MSG_MSG" | awk '{print $4}')

echo "Dispatched Processing Job Array: Job ID is $PREPROCESS_MSG_JOB_ID (Waiting on $READ_DATA_JOB_ID)"
echo "--------------------------------------------------------"
echo "Pipeline successfully queued!"