#!/bin/bash
#!/bin/bash
#SBATCH --account=st-nina-1-gpu          # Your specific project allocation account
#SBATCH --job-name=model_inference
#SBATCH --partition=gpu             # Target Sockeye's hardware-accelerated nodes
#SBATCH --time=06:00:00             # Time limit (HH:MM:SS) - adjust based on data size
#SBATCH --nodes=1                   # Keep everything on 1 physical node
#SBATCH --ntasks=1                  # 1 master application task
#SBATCH --cpus-per-task=4           # 4 CPU cores are plenty just to stream video files to the GPU
#SBATCH --mem=32G                   # 32GB of system RAM to handle video streams safely
#SBATCH --gpus=1                    # Request exactly 1 GPU (e.g., NVIDIA V100 or A100)
#SBATCH --output=logs/infer_%A_%a.out
#SBATCH --error=logs/infer_%A_%a.err

cd /scratch/st-nina-1/mooVision
source .env_sockeye


# Ensure your orchestrator is calling the updated GPU batch version of your script!
uv run scripts/run-testing-3.py \
    --model_type yolo \
    --model_path /scratch/st-nina-1/moovision/yolo_training_runs/split_1_model/weights/best.pt \
    --data_path /scratch/st-nina-1/moovision/data/processed/random/test.csv