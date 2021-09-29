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
    def preprocess(data: DataFrame, ranked: bool=None) -> DataFrame:
        pre = defaultdict(list)
        for index, row in data.iterrows():
            hist_stats = ast.literal_eval(row["hist_stats"])
            # print(hist_stats)
            episodes_this_iter = row["episodes_this_iter"]
            for comm_type in COMM_TYPES:
                for key in hist_stats:
                    if comm_type not in key:
                        continue
                    comm_costs = hist_stats[key][:-episodes_this_iter]
                    for comm_cost in comm_costs:
                        pre["round"].append(row["round"])
                        pre["timesteps_total"].append(row["timesteps_total"])
                        pre["trainer"].append(row["trainer"])
                        if ranked is None:
                            pre["ranked"].append(row["ranked"])
                        else:
                            pre["ranked"].append(ranked)
                        pre["episode_reward_mean"].append(row["episode_reward_mean"])
                        pre["comm_type"].append(comm_type)
                        
                        # Add the communication cost here.
                        if comm_type == TLS2EDGE_POLICY and "FedRL" in row["trainer"]:
                            pre["comm_cost"].append(1)
                        elif comm_type == EDGE2TLS_POLICY and "FedRL" in row["trainer"]:
                            pre["comm_cost"].append(1)
                        else:
                            pre["comm_cost"].append(comm_cost)

                        # TODO: Consider a "total" communication cost feature as well.

        preprocessed_data = DataFrame.from_dict(pre)
        return preprocessed_data