import ast
import pandas as pd
import sys

from collections import defaultdict
from pandas import DataFrame

sys.path.append("..")
from seal.trainer.communication import *

class PreprocessedCommData(object):

    # data: DataFrame

    # def __init__(self, obj) -> None:
    #     if isinstance(obj, DataFrame):
    #         self.data = self.preprocess(obj)
    #     elif isinstance(obj, str):
    #         self.data = self.preprocess(pd.read_csv(obj))
    #     else:
    #         raise ValueError("Invalid type for argument `obj`.")

    @staticmethod
    def preprocess(data: DataFrame) -> DataFrame:
        pre = defaultdict(list)
        for index, row in data.iterrows():
            hist_stats = ast.literal_eval(row["hist_stats"])
            # print(hist_stats)
            for comm_type in COMM_TYPES:
                pre["round"].append(row["round"])
                pre["timesteps_total"].append(row["timesteps_total"])
                pre["trainer"].append(row["trainer"])
                pre["rank"].append(row["rank"])
                pre["episode_reward_mean"].append(row["episode_reward_mean"])
                pre["comm_type"].append(comm_type)
                pre["comm_cost"].append(sum(
                    sum(hist_stats[key]) if isinstance(hist_stats[key], list) else hist_stats[key]
                    for key in hist_stats
                    if comm_type in key
                ))
        preprocessed_data = DataFrame.from_dict(pre)
        return preprocessed_data