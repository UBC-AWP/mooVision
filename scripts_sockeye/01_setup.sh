#!/bin/bash

# LOCAL REPO SETUP
set -e

# Move into MooVision Project
cd /scratch/st-nina-1/mooVision

echo "=== Running One-Time Global Setup ==="

# PULL LATEST CHANGES
git pull origin dev

# ENVIRONMENT VALIDATION & LOAD
if [ ! -f .env_sockeye ]; then
    echo "ERROR: .env_sockeye file not found in current directory!"
    echo "Please copy and configure the .env_sockeye template before submitting."
    exit 1
fi

source .env_sockeye

if [ ! -f .env ]; then
    ln -s .env_sockeye .env
fi

# Sync environment
uv python pin 3.12
uv sync

# DOWNLOAD YOLO MODEL WEIGHTS 
URL="https://github.com/ultralytics/assets/releases/download/v8.4.0/${YOLO_MODEL}"

echo "Checking environment..."

# Safety Check: Prevent running this on an offline compute node
if [ -n "$PBS_JOBID" ] || [ -n "$SLURM_JOB_ID" ]; then
    echo "ERROR: You are running this on an offline compute node!"
    echo "Please run this script from a Sockeye LOGIN node where internet is available."
    exit 1
fi

# Create weights directory if it doesn't exist
mkdir -p "$WEIGHTS_DIR"

# Check if the model already exists to save bandwidth
if [ -f "$WEIGHTS_DIR/$YOLO_MODEL" ]; then
    echo "Model weights ($YOLO_MODEL) already exist in $WEIGHTS_DIR. Skipping download."
else
    echo "Downloading $YOLO_MODEL into $WEIGHTS_DIR..."
    wget -P "$WEIGHTS_DIR" "$URL"
    echo "Download complete!"
fi

# Run any standalone pre-checks or global asset extractions here
echo "Setup complete."