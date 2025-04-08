THREAD="s2"

# Define models, configs, and sample sizes
MODELS=("yolo12x" "yolo11x" "rtdetr-x" "rtdetr-l" "yolo12m" "yolo11m" "yolo12n" "yolo11n")
CONFIGS=("e_IMU" "e_VT")
N_SAMPLES=(200 128 64 32)

# Run the script with different configurations
for ITER in {0..10}; do
    for MODEL in "${MODELS[@]}"; do
        for CONFIG in "${CONFIGS[@]}"; do
            for N_SAMPLE in "${N_SAMPLES[@]}"; do
                    python study2.py \
                        --iter $ITER \
                        --thread $THREAD \
                        --config $CONFIG \
                        --n_sample $N_SAMPLE \
                        --model $MODEL \
                        --is_pretrained
                    python study2.py \
                        --iter $ITER \
                        --thread $THREAD \
                        --config $CONFIG \
                        --n_sample $N_SAMPLE \
                        --model $MODEL
                done
            done
        done
    done
done
