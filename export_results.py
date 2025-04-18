import pandas as pd
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

from path import PathFinder
PATHS = PathFinder()
DIR_OUT = PATHS["DIR_SRC"] / "out" 
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

def main():
    ls_csv = [f for f in (DIR_OUT / "study1").glob("*.csv")] + \
             [f for f in (DIR_OUT / "study2").glob("*.csv")]

    # study 1
    df_s1 = pd.concat([pd.read_csv(f) for f in ls_csv if "study1" in str(f)], ignore_index=True).drop_duplicates()
    df_s1 = df_s1.query("n in [32, 128, 500]")
    df_s1 = add_family(df_s1)
    df_s1 = add_size(df_s1)
    df_s1 = df_s1.query("iter <= 2")

    # study 2
    df_s2 = pd.concat([pd.read_csv(f) for f in ls_csv if "study2" in str(f)], ignore_index=True).drop_duplicates()
    df_s2 = add_family(df_s2)
    df_s2 = add_size(df_s2)
    
    # study time
    tensorboard_files = list(DIR_OUT.rglob("events.out.tfevents.*"))
    df_time = get_med_time_from_tensorboard(tensorboard_files)
    df_time.head()
    df_time = add_family(df_time)
    df_time = add_size(df_time)
    df_time = add_gpu(df_time)
    df_time = add_params(df_time)
    
    # export
    df_s1.to_csv(DIR_OUT / "study1.csv", index=False)
    df_s2.to_csv(DIR_OUT / "study2.csv", index=False)
    df_time.to_csv(DIR_OUT / "train_time.csv", index=False)
    
def add_family(df):
    df.loc[:, ["family"]] = "DETR"
    df.loc[df["model"].str.contains("yolo12"), "family"] = "YOLOv12"
    df.loc[df["model"].str.contains("yolo11"), "family"] = "YOLOv11"
    return df

def add_size(df):
    df.loc[:, ["size"]] = "small"
    df.loc[df["model"].str.contains("yolo11x"), "size"] = "large"
    df.loc[df["model"].str.contains("yolo12x"), "size"] = "large"
    df.loc[df["model"].str.contains("rtdetr-x"), "size"] = "large"
    return df

def add_gpu(df):
    df.loc[:, "gpu"] = ["a100 (80GB)" if t < 5 else "l40s (48GB)" for t in df["thread"]]
    return df

def add_params(df):
    df.loc[:, "params"] = df["model"].map(DICT_PARAMS)
    return df

def get_med_time_from_tensorboard(tensorboard_files):
    ls_select = []
    ls_models = []
    ls_thread = []
    ls_mu = []
    for m in ["yolo12x", "yolo11x", "rtdetr-x", "rtdetr-l", "yolo12n", "yolo11n"]:
        for t in range(10):
            for i in range(2):
                filename = f"study1/thread_{t}/0_all_{m}_128/iter_{i}/events.out"
                match_file = [f for f in tensorboard_files if filename in str(f)]
                if len(match_file) == 0:
                    continue
                path_file = PATHS["DIR_SRC"] / "out" / match_file[0]
                mu = get_med_time(path_file)
                ls_mu.append(mu)
                ls_models.append(m)
                ls_thread.append(t)
                ls_select.append(filename)
    data = pd.DataFrame({
        "model": ls_models,
        "thread": ls_thread,
        "median": ls_mu,
        "filename": ls_select,
    })
    return data

def get_med_time(tfile):
    ea = event_accumulator.EventAccumulator(str(tfile))
    ea.Reload()
    events = ea.Scalars("train/cls_loss")
    step_times = []
    for i in range(1, len(events)):
        delta_time = events[i].wall_time - events[i - 1].wall_time
        delta_step = events[i].step - events[i - 1].step
        if delta_step > 0:
            step_times.append(delta_time / delta_step)
    med = np.median(step_times)
    return med


if __name__ == "__main__":
    main()
    
    
    
# import pandas as pd

# from export_results import add_family, add_params, add_size

# data = pd.read_csv("mem.csv")
# ls_models = ["yolo12x", "yolo11x", "rtdetr-x", "rtdetr-l", "yolo12n", "yolo11n"]
# ls_size = [32, 128, 500]
# data.loc[:, "model"] = [m for m in ls_models for _ in range(12)]
# data.loc[:, "n"] = [s for _ in ls_models for i in range(4) for s in ls_size ]
# data

# import seaborn as sns
# import matplotlib.pyplot as plt

# # y = mem, x = n, color = model
# sns.set(style="whitegrid")
# plt.figure(figsize=(10, 6))
# sns.lineplot(data=data, x="n", y="mem", hue="model", marker="o")


# data = add_family(data)
# data = add_size(data)
# data = add_params(data)

# data.to_csv("train_mem.csv", index=False)