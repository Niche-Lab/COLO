#!/bin/bash
#SBATCH --job-name=study1
#SBATCH -t 119:59:59
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=niche_squad
#SBATCH --array=0-7 # Job array
#SBATCH --output=logs/study1_%A_%a.out
#SBATCH --error=logs/study1_%A_%a.err

# Load necessary modules (if required)
source activate pyniche

# Set iteration and thread variables
THREAD=$SLURM_ARRAY_TASK_ID

# Define models, configs, and sample sizes
MODELS=("yolo12x" "yolo11x" "rtdetr-x" "rtdetr-l" "yolo12m" "yolo11m" "yolo12n" "yolo11n")
CONFIGS=("0_all" "a1_t2s" "a2_s2t" "b_light")
N_SAMPLES=(500 256 128 64 32)

# Run the script with different configurations
for ITER in {0..10}; do
    for MODEL in "${MODELS[@]}"; do
        for CONFIG in "${CONFIGS[@]}"; do
            for N_SAMPLE in "${N_SAMPLES[@]}"; do
                python study1.py \
                    --iter $ITER \
                    --thread $THREAD \
                    --config $CONFIG \
                    --n_sample $N_SAMPLE \
                    --model $MODEL
            done
        done
    done
done
