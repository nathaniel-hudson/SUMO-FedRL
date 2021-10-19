import os
import pandas as pd
from glob import glob
from os.path import join
from pandas import DataFrame

PIVOT_KEY = "policy"
PIVOT_KEY_VALUE = "TEST-EVAL_POLICY"

def clean(df) -> DataFrame:
    if PIVOT_KEY in df.columns:
        df = df.query(f"policy == '{PIVOT_KEY_VALUE}'")
        df.reset_index(inplace=True)
        df.drop(PIVOT_KEY, inplace=True, axis=1)
    return df

if __name__ == "__main__":
    trainer_dirs = ["FedRL", "MARL", "SARL"]
    intersection_dirs = ["double", "grid-3x3", "grid-5x5"]
    print(">> Cleaning data files...")
    for trainer in trainer_dirs:
        for intersection in intersection_dirs:
            for p in glob(join("out", "data", trainer, intersection, "*.csv")):
                filename = p.split(os.sep)[-1].split(".")[0]
                out_path = ("out", "data", "Final", trainer, intersection)
                if not os.path.isdir(join(*out_path)):
                    os.makedirs(join(*out_path))
                df = pd.read_csv(p)
                df = clean(df)

                print(f">> Cleaning '{p}'.")
                csv_filename = join(*out_path, f"{filename}.csv")
                df.to_csv(csv_filename)
                
                print(f"\t+ '{csv_filename}' saved!")
                json_filename = join(*out_path, f"{filename}.json")
                df.to_json(json_filename)
                print(f"\t+ '{json_filename}' saved!")
    print(f">> Done!")