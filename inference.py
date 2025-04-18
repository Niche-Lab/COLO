import torch
from ultralytics import YOLO
import time
import pandas as pd
from path import PathFinder
import numpy as np

GPU = "L40S"
LS_MODEL = ["yolo11x.pt", "yolo12x.pt", "yolo11n.pt", "yolo12n.pt", "rtdetr-x.pt", "rtdetr-l.pt"]
N_IMGS = 64

paths = PathFinder()
path_out = paths["DIR_SRC"] / "out" / "infer.csv"

m = LS_MODEL[0]
model = YOLO(m)

tensor = torch.randn(N_IMGS, 3, 640, 640)
tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

ls_time = []
ls_mem = []
for i in range(31):
    start = time.time()
    _ = model(tensor)
    end = time.time()
    ls_time.append(end - start)
    ls_mem.append(torch.cuda.memory_allocated() / 1024**2)

print(f"Max: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

ls_FPS = np.round(N_IMGS / np.array(ls_time), 2)
pd.DataFrame({"model": m[:-3], "fps": ls_FPS[1:]}).to_csv(path_out, index=False, append=True)

# yolo11x: 
# yolov11n: 1777.25MB