"""
pretrain the model
"""
import argparse
import sys
from ultralytics import YOLO, RTDETR

# local imports
from path import PathFinder
PATHS = PathFinder()
sys.path.insert(0, PATHS["LIB_PYNICHE"].as_posix())
from pyniche.data.yolo.API import YOLO_API


def main(args):
    thread = "finetune"
    modelname = args.model
        
    DIR_DATA = PATHS["DIR_DATA"] / "0_all"
    DIR_OUT = PATHS["DIR_SRC"] / "out" / "study2_finetune" / modelname

    # data ------------------------

    data = YOLO_API(DIR_DATA)
    data.make_train(split_src="train", suffix=thread)
    data.make_val(split_src="test", suffix=thread)
    data.make_test(split_src="test", suffix=thread)
    path_yaml = data.save_yaml(classes=["cow"], suffix=thread)

    # model ------------------------
    if "detr" in modelname:
        model = RTDETR(modelname)
    else:
        model = YOLO(modelname)


    # training ------------------------
    model.train(data=path_yaml,
                epochs=300,
                patience=20,
                batch=16, 
                project=DIR_OUT,
                name=".",)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo12x")
    
    args = parser.parse_args()
    main(args)