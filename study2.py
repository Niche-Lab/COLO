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


def main(args):
    iters = args.iter
    thread = args.thread
    config = args.config
    n_sample = args.n_sample
    modelname = args.model
    is_pretrain = args.is_pretrained
    n_params = DICT_PARAMS[modelname]
    # skip yolo11m, yolo12m
    if modelname in ["yolo11m", "yolo12m"]:
        return None
    # skip n_sample = 32 and 128
    if int(n_sample) in [32, 128]:
        return None
    
    DIR_DATA = PATHS["DIR_SRC"] / "data" / f"thread_{thread}" / config
    PATH_MODEL = PATHS["DIR_SRC"] / "out" / "study2_finetune" / modelname / "weights" / "best.pt"
    FILE_OUT = PATHS["DIR_SRC"] / "out" / "study2" / f"results_{thread}.csv"
    DIR_PROJECT = PATHS["DIR_SRC"] / "out" / "study2" / f"thread_{thread}" / f"{config}_{modelname}_{n_sample}_pt-{is_pretrain}"

    # data ------------------------
    seed = string_to_seed(f"{iters}_{thread}")
    random.seed(seed)
    
    data = YOLO_API(DIR_DATA, write_txt=True)
    data.shuffle_train_val(split_src="train", n=int(n_sample), suffix=thread)
    data.make_test(split_src="test", suffix=thread)
    path_yaml = data.save_yaml(classes=["cow"], suffix=thread)

    # model ------------------------
    if "detr" in modelname:
        if is_pretrain:
            model = RTDETR(modelname)
        else:
            model = RTDETR(PATH_MODEL)
    else:
        if is_pretrain:
            model = YOLO(modelname)
        else:
            model = YOLO(PATH_MODEL)

    # training ------------------------
    is_largemodel = True if modelname in ["yolo12x", "yolo11x"] else False
    batch = 16 if is_largemodel else 32 
    epochs, patience = get_config(batch, int(n_sample))

    model.train(data=path_yaml,
                epochs=epochs,
                patience=patience,
                workers=4,
                batch=batch,
                project=DIR_PROJECT,
                name=f"iter_{iters}",)
    out = model.val(data=path_yaml,
                    split="test",
                    project=DIR_PROJECT.as_posix() + "-eval",
                    name=f"iter_{iters}",)
    
    # evaluation ------------------------
    metrics = eval_metrics(out)
    str_profile = f"{config},{is_pretrain},{modelname},{n_params},{n_sample},{thread},{iters},"
    str_metrics = ",".join([str(value) for value in metrics.values()])

    if os.path.exists(FILE_OUT):
        with open(FILE_OUT, "a") as file:
            file.write(str_profile + str_metrics + "\n")
    else:
        with open(FILE_OUT, "w") as file:
            file.write("data,pretrain,model,params,n,thread,iter,map5095,map50,precision,recall,f1,n_all,n_fn,n_fp\n")
            file.write(str_profile + str_metrics + "\n")

    # remove the project folder
    # if thread != "0":
    import shutil
    dir_model = DIR_PROJECT / f"iter_{iters}" / "weights"
    shutil.rmtree(dir_model)


def string_to_seed(s):
    # Use hashlib to get a consistent integer from a string
    hash_object = hashlib.md5(s.encode())  # can also use sha256
    seed_int = int(hash_object.hexdigest(), 16) % (2**32)
    return seed_int


def get_config(batch, n, total_steps=8192):
    steps_per_epoch = n // batch
    epochs = total_steps // steps_per_epoch
    patience = epochs // 4
    return epochs, patience
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iter", type=str, default="0")
    parser.add_argument("-t", "--thread", type=str, default="0")
    parser.add_argument("-c", "--config", type=str, default="0_all")
    parser.add_argument("-n", "--n_sample", type=str, default="32")
    parser.add_argument("-m", "--model", type=str, default="yol1o12n")
    parser.add_argument("-p", "--is_pretrained", action="store_true", default=False)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        # write error to file
        # concate all args
        taskid = f"{args.thread}_{args.iter}_{args.model}_{args.config}_{args.n_sample}_{args.is_pretrained}"
        error_message = str(e)
        error_file = PATHS["DIR_SRC"] / "logs" / "study2" / f"error_{taskid}.txt"
        with open(error_file, "a") as file:
            file.write(f"Error in task {taskid}: {error_message}\n")
        import time
        time.sleep(300)
        raise e


