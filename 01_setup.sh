#!/bin/bash
#SBATCH --account=st-nina-1
#SBATCH --job-name=setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4gb
#SBATCH --time=00:10:00
#SBATCH --output=logs/setup_%j.out

# module load git

cd /scratch/st-nina-1/mooVision

echo "=== Running One-Time Global Setup ==="
# git pull origin arc-setup-dev

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

# Run any standalone pre-checks or global asset extractions here
echo "Setup complete."