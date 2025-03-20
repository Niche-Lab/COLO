# Set iteration and thread variables
THREAD="white"

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
