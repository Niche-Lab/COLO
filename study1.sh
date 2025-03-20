#
# python study1.py\
#     --iter 0\
#     --thread 0\
#     --config "0_all"\
#     --n_sample 500\
#     --model "yolo12x"
#

# models: ["rtdetr-x", "rtdetr-l", "yolo12x", "yolo11x", "yolo12m", "yolo11m", "yolo12n", "yolo11n"]
# config: ["0_all", "a1_t2s", "a2_s2t", "b_light"]
# n_sample: 32, 64, 128, 256, 500
for i in {0..10}
do
    for model in "yolo12x" "yolo11x" "rtdetr-x" "rtdetr-l" "yolo12m" "yolo11m" "yolo12n" "yolo11n"
    do
        for config in "0_all" "a1_t2s" "a2_s2t" "b_light"
        do
            for n_sample in 32 64 128 256 500
            do
                python study1.py\
                    --iter $i\
                    --thread $1\
                    --config $config\
                    --n_sample $n_sample\
                    --model $model
            done                
        done
    done
done