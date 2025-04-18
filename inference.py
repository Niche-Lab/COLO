import torch
from ultralytics import YOLO
import time
import pandas as pd
from path import PathFinder
import numpy as np
import tqdm

GPU = "L40S"
LS_MODEL = ["yolo11x.pt", "yolo12x.pt", "yolo11n.pt", "yolo12n.pt", "rtdetr-x.pt", "rtdetr-l.pt"]
N_IMGS = 64
paths = PathFinder()
path_out = paths["DIR_SRC"] / "out" / "infer.csv"

for m in LS_MODEL:
    model = YOLO(m)

for m in tqdm.tqdm(LS_MODEL, desc="model"):
    tensor = torch.randn(N_IMGS, 3, 640, 640)
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    tensor = tensor.float().cuda()
    
    model = torch.load(m)["model"]
    model = model.float().cuda()
    
    ls_time = []
    for i in tqdm.tqdm(range(30), desc="iteration"):
        start = time.time()
        _ = model(tensor)
        end = time.time()
        ls_time.append(end - start)

    print(f"Max: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

    ls_FPS = np.round(N_IMGS / np.array(ls_time), 2)
    pd.DataFrame({"model": m[:-3], "fps": ls_FPS[1:], "gpu": GPU}).\
        to_csv(path_out, index=False, mode='a', header=False)

# yolo11x:  8954.13MB
# yolo12x:  8062.41MB
# yolov11n: 1777.25MB
# yolov12n:  1777.16MB
# rtdetr-x:  6586.85MB
# rtdetr-l:  5057.37MP