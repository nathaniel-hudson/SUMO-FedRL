import os
import pandas as pd
from glob import glob
from os.path import join
from pandas import DataFrame
from seal.logging import *

PIVOT_KEY = "policy"
PIVOT_KEY_VALUE = "TEST-EVAL_POLICY"

# ICCPS 2022 Subsmission Settings
# OUT_PATH_PREFIX = ("out", "data", "Final")
TRAINER_DIRS = ["FedRL", "MARL", "SARL"]
INTERSECTION_DIRS = ["double", "grid-3x3", "grid-5x5"]

# SMARTCOMP 2022 Submission Settings
OUT_PATH_PREFIX = ("out", "data", "SMARTCOMP")
TRAINER_DIRS = ["FedRL", "MARL", "SARL"]
INTERSECTION_DIRS = ["grid-3x3", "grid-5x5", "grid-7x7"]


def clean(data: DataFrame) -> DataFrame:
    if PIVOT_KEY in data.columns:
        data = data.query(f"policy == '{PIVOT_KEY_VALUE}'")
        data.reset_index(inplace=True)
        data.drop(PIVOT_KEY, inplace=True, axis=1)
    return data


if __name__ == "__main__":
    trainer_dirs = ["FedRL", "MARL", "SARL"]
    logging.info("Starting to clean data files.")
    for trainer in TRAINER_DIRS:
        for intersection in INTERSECTION_DIRS:
            for p in glob(join("out", "SMARTCOMP", "data", trainer, intersection, "*.csv")):
                filename = p.split(os.sep)[-1].split(".")[0]
                out_path = (*OUT_PATH_PREFIX, trainer, intersection)
                if not os.path.isdir(join(*out_path)):
                    os.makedirs(join(*out_path))
                df = pd.read_csv(p)
                df = clean(df)

                logging.info(f"Cleaning file '{p}'.")
                csv_filename = join(*out_path, f"{filename}.csv")
                df.to_csv(csv_filename)

                logging.info(f"File '{csv_filename}' saved.")
                json_filename = join(*out_path, f"{filename}.json")
                df.to_json(json_filename)
                logging.info(f"\t+ '{json_filename}' saved.")
    logging.info(f"Finished cleaning data.")
