#!/bin/bash
#SBATCH --account=st-nina-1
#SBATCH --job-name=read_and_split_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=01:00:00
#SBATCH --output=logs/read_%j.out

export PYTHONUNBUFFERED=1
cd /scratch/st-nina-1/mooVision
source .env_sockeye

echo "========================================================"
echo "READING DATA FILE AND SPLITTING TO TRAIN/TEST SETS"
echo "========================================================"

uv run scripts/read_data/read_all_clips_index.py --force
uv run scripts/split_data/split_data.py --force

echo "========================================================"
echo "SUCCESS: DATA SPLITS CREATED."
echo "========================================================"