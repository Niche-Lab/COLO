import os
import argparse
import sys
from ultralytics import YOLO

# local imports
from path import PathFinder
from evaluate import eval_metrics
PATHS = PathFinder()
sys.path.insert(0, PATHS["LIB_PYNICHE"].as_posix())
from pyniche.data.yolo.API import YOLO_API

LS_MODELS = [
    "yolo11n", "yolov11m", "yolo11x", #  39.5, 51.5, 54.7
    "yolo12n", "yolov12m", "yolo12x", # 40.6, 52.5, 55.2
    "rtdetr-l", "rtdetr-x"] # 53.0 mAP, 54.8 mAP
LS_PARMS = [
    2.6, 20.1, 56.9,
    2.6, 20.2, 59.1,
    45, 86]
LS_SIZES = [32, 64, 128, 256, 500]
LS_DATA = ["0_all", "a1_t2s", "a2_s2t", "b_light"]


def main(args):
    iters = args.iter
    thread = str(args.thread)
    DIR_OUT = PATHS["DIR_SRC"] / "out" / f"thread_{thread}"
    FILE_OUT = DIR_OUT / "results.csv"

    for d in LS_DATA:
        DIR_DATA = PATHS["DIR_DATA"] / d
        
        for n in LS_SIZES:

            data = YOLO_API(DIR_DATA)
            data.shuffle_train_val(split_src="train", n=n, suffix=thread)
            data.make_test(split_src="test", suffix=thread)
            path_yaml = data.save_yaml(classes=["cow"], suffix=thread)

            for m, np in zip(LS_MODELS, LS_PARMS):
                project = DIR_OUT / f"{d}_{m}_{n}"
                
                model = YOLO(m)
                model.train(data=path_yaml, 
                            epochs=2,
                            patience=50,
                            cos_lr=True,
                            project=project,
                            name=f"iter_{iters}",)
                out = model.val(data=path_yaml,
                                split="test",
                                project=str(project) + "-eval",
                                name=f"iter_{iters}",)
                
                # evaluation
                metrics = eval_metrics(out)
                str_profile = f"{d},{m},{np},{n},{thread},{iters},"
                str_metrics = ",".join([str(value) for value in metrics.values()])

                if os.path.exists(FILE_OUT):
                    with open(FILE_OUT, "a") as file:
                        file.write(str_profile + str_metrics + "\n")
                else:
                    with open(FILE_OUT, "w") as file:
                        file.write("data,model,params,n,thread,iter,map5095,map50,precision,recall,f1,n_all,n_fn,n_fp\n")
                        file.write(str_profile + str_metrics + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iter", type=int, default=0)
    parser.add_argument("-t", "--thread", type=int, default=0)
    args = parser.parse_args()
    main(args)


