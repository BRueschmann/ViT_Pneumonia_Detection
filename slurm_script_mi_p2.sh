#!/bin/bash
#SBATCH --job-name=python_train_script
#SBATCH --output=logs/my_job_%j.out   # Output log
#SBATCH --error=logs/my_job_%j.err    # Error log
#SBATCH --partition=gpu               # Use GPU partition
#SBATCH --gres=gpu:1                  # Request one GPU
#SBATCH --time=12:00:00               # Maximum run time of 12 hours
#SBATCH --mem=48G                     # Memory allocation of 48GB
#SBATCH --cpus-per-task=10            # 21 CPU cores

# Load and activate Conda environment
source ~/.bashrc
conda activate myenv

# Navigate to your project directory
cd /home/br61/medical_imaging/window_150_200

# Execute training script
CUDA_VISIBLE_DEVICES=0 python code_mi_p2.py \
    --model_name "google/vit-base-patch16-224-in21k" \
    --windowing \
    --lower "150" \
    --upper "200" \
    --data_root "/sharedscratch/br61/medical_imaging/chest_xray" \
    --output_dir "/home/br61/medical_imaging/window150_200" \
    --batch_size 16 \
    --num_workers 4
# Confirmation message
echo "Job complete"

