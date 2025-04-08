#!/bin/bash
#SBATCH --job-name=study2
#SBATCH -t 119:59:59
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=niche_squad
#SBATCH --output=logs/study2-ft_%A_%a.out
#SBATCH --error=logs/study2-ft_%A_%a.err


# Load necessary modules (if required)
source activate pyniche

# Define models, configs, and sample sizes
MODELS=("yolo12x" "yolo11x" "rtdetr-x" "rtdetr-l" "yolo12m" "yolo11m" "yolo12n" "yolo11n")

for MODEL in "${MODELS[@]}"; do
    python study2-finetune.py \
        --model $MODEL
done
