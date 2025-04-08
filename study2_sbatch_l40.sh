#!/bin/bash
#SBATCH --job-name=study2
#SBATCH -t 143:59:59
#SBATCH --partition=l40s_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=niche_squad
#SBATCH --array=5-9 # Job array
#SBATCH --output=logs/study2_%A_%a.out
#SBATCH --error=logs/study2_%A_%a.err

# Load necessary modules (if required)
source activate pyniche

# Set iteration and thread variables
THREAD=${SLURM_ARRAY_TASK_ID}

# Define models, configs, and sample sizessq
MODELS=("rtdetr-x" "rtdetr-l" "yolo12n" "yolo11n")
CONFIGS=("e_IMU" "e_VT")
# N_SAMPLES=(200 128 64 32)
N_SAMPLES=(16)

# Run the script with different configurations
for ITER in {1..100}; do
    for MODEL in "${MODELS[@]}"; do
        for CONFIG in "${CONFIGS[@]}"; do
            for N_SAMPLE in "${N_SAMPLES[@]}"; do
                python study2_l40.py \
                    --thread $THREAD \
                    --iter $ITER \
                    --model $MODEL \
                    --config $CONFIG \
                    --n_sample $N_SAMPLE \
                    --is_pretrained 

                python study2_l40.py \
                    --thread $THREAD \
                    --iter $ITER \
                    --model $MODEL \
                    --config $CONFIG \
                    --n_sample $N_SAMPLE
            done
        done
    done
done



# python study2_l40.py \
#     --thread 5 \
#     --iter 1 \
#     --model rtdetr-x \
#     --config e_IMU \
#     --n_sample 16 \
#     --is_pretrained 