import torch
from ultralytics import YOLO
import time
import pandas as pd
from path import PathFinder
import numpy as np
import tqdm


# batch = 64
# yolo11x:  8954.13MB
# yolo12x:  8962.41MB
# yolov11n: 1777.25MB
# yolov12n:  1777.16MB
# rtdetr-x:  6586.85MB
# rtdetr-l:  5057.37MP
# batch = 16
# yolo11x = 3125.37 MB
# yolo12x = 3133.62 mBpoy
# yolo11n =  650.00 mB
# yolo12n =  648.75 mB
# rtdetr-x =  1948.73 mB
# rtdetr-l = 1469.92 mB


GPU = "A100"
LS_MODEL = ["yolo11x.pt", "yolo12x.pt", "yolo11n.pt", "yolo12n.pt", "rtdetr-x.pt", "rtdetr-l.pt"]
N_IMGS = 16
paths = PathFinder()
path_out = paths["DIR_SRC"] / "out" / "infer.csv"

for m in LS_MODEL:
    model = YOLO(m)


for m in tqdm.tqdm(LS_MODEL, desc="model"):
    try:
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
    except Exception as e:
        print(f"Error processing {m}: {e}")
        ls_FPS = [0] * 30
    pd.DataFrame({"model": m[:-3], "fps": ls_FPS[1:], "gpu": GPU}).\
        to_csv(path_out, index=False, mode='a', header=False)




# m = LS_MODEL[5]
# tensor = torch.randn(N_IMGS, 3, 640, 640)
# tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
# tensor = tensor.float().cuda()

# model = torch.load(m)["model"]
# model = model.float().cuda()

# ls_time = []
# for i in tqdm.tqdm(range(30), desc="iteration"):
#     start = time.time()
#     _ = model(tensor)
#     end = time.time()
#     ls_time.append(end - start)

# print(f"Max: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
