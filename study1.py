import os
import argparse
import sys
from ultralytics import YOLO, RTDETR
import random
import hashlib

# local imports
from path import PathFinder
from evaluate import eval_metrics
PATHS = PathFinder()
sys.path.insert(0, PATHS["LIB_PYNICHE"].as_posix())
from pyniche.data.yolo.API import YOLO_API

DICT_PARAMS = dict({
    "rtdetr-l": 45,
    "rtdetr-x": 86,
    "yolo12n": 2.6,
    "yolo12m": 20.2,
    "yolo12x": 59.1,
    "yolo11n": 2.6,
    "yolo11m": 20.1,
    "yolo11x": 56.9,
})

# LS_DATA = ["0_all", "a1_t2s", "a2_s2t", "b_light"]


def main(args):
    iters = args.iter
    thread = args.thread
    config = args.config
    n_sample = args.n_sample
    modelname = args.model
    n_params = DICT_PARAMS[modelname]
    
    DIR_DATA = PATHS["DIR_DATA"] / config
    DIR_OUT = PATHS["DIR_SRC"] / "out" / f"thread_{thread}"
    FILE_OUT = DIR_OUT / "results.csv"

    # data ------------------------
    seed = string_to_seed(f"{iters}_{thread}")
    random.seed(seed)
    
    data = YOLO_API(DIR_DATA)
    data.shuffle_train_val(split_src="train", n=n_sample, suffix=thread)
    data.make_test(split_src="test", suffix=thread)
    path_yaml = data.save_yaml(classes=["cow"], suffix=thread)

    # model ------------------------
    if "detr" in modelname:
        model = RTDETR(modelname)
    else:
        model = YOLO(modelname)

    # training ------------------------
    project = DIR_OUT / f"{config}_{modelname}_{n_sample}"
    model.train(data=path_yaml, 
                epochs=300,
                patience=50,
                project=project,
                name=f"iter_{iters}",)
    out = model.val(data=path_yaml,
                    split="test",
                    project=str(project) + "-eval",
                    name=f"iter_{iters}",)
    
    # evaluation ------------------------
    metrics = eval_metrics(out)
    str_profile = f"{config},{modelname},{n_params]},{n_sample},{thread},{iters},"
    str_metrics = ",".join([str(value) for value in metrics.values()])

    if os.path.exists(FILE_OUT):
        with open(FILE_OUT, "a") as file:
            file.write(str_profile + str_metrics + "\n")
    else:
        with open(FILE_OUT, "w") as file:
            file.write("data,model,params,n,thread,iter,map5095,map50,precision,recall,f1,n_all,n_fn,n_fp\n")
            file.write(str_profile + str_metrics + "\n")


def string_to_seed(s):
    # Use hashlib to get a consistent integer from a string
    hash_object = hashlib.md5(s.encode())  # can also use sha256
    seed_int = int(hash_object.hexdigest(), 16) % (2**32)
    return seed_int


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iter", type=str, default="0")
    parser.add_argument("-t", "--thread", type=str, default="0")
    parser.add_argument("-c", "--config", type=str, default="0_all")
    parser.add_argument("-n", "--n_sample", type=str, default="32")
    parser.add_argument("-m", "--model", type=str, default="yol1o12n")
    
    args = parser.parse_args()
    main(args)


