from path import PathFinder
import pandas as pd

PATHS = PathFinder()
DIR_OUT = PATHS["DIR_SRC"] / "out" 

def main():
    ls_csv = [f for f in (DIR_OUT / "study1").glob("*.csv")] + \
            [f for f in (DIR_OUT / "study2").glob("*.csv")]

    df_s1 = pd.concat([pd.read_csv(f) for f in ls_csv if "study1" in str(f)], ignore_index=True).drop_duplicates()
    df_s1 = df_s1.query("n in [32, 128, 500]")
    df_s1 = add_family(df_s1)
    df_s1 = add_size(df_s1)
    df_s1 = df_s1.query("iter <= 2")

    df_s2 = pd.concat([pd.read_csv(f) for f in ls_csv if "study2" in str(f)], ignore_index=True).drop_duplicates()
    df_s2 = add_family(df_s2)
    df_s2 = add_size(df_s2)
    
    # export
    df_s1.to_csv(DIR_OUT / "study1.csv", index=False)
    df_s2.to_csv(DIR_OUT / "study2.csv", index=False)
    
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


if __name__ == "__main__":
    main()