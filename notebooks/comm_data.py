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
    def preprocess(
        path: str=None,
        data: DataFrame=None, 
        ranked: bool=None, 
        is_sarl: bool=False
    ) -> DataFrame:
        assert not all([path, data]), "Can only provide either a DataFrame or a path to a .csv file."
        
        if path is not None:
            data = pd.read_csv(path)
        elif data is None:
            raise ValueError("Must either provide a path to a .csv file or a DataFrame.")
        
        if is_sarl:
            return PreprocessedCommData._sarl_preprocess(data, ranked=ranked)
        else:
            return PreprocessedCommData._marl_preprocess(data, ranked=ranked)

    @staticmethod
    def _sarl_preprocess(data: DataFrame, ranked: bool=None) -> DataFrame:
        
        def get_num_of_trafficlights() -> int:
            trafficlight_ids = set()
            record = next(iter(data["hist_stats"]))
            for key in record:
                if "policy" in key and "comm" in key:
                    idx = key.split("policy")[1].split("comm")[0]
                    idx = idx.replace("_", "")
                    trafficlight_ids.add(idx)
            return len(trafficlight_ids)
        
        
        REWARD_KEY = "policy_sarl-policy_reward"
        pre = defaultdict(list)
        n_trafficlights = get_num_of_trafficlights()
        loop_count = 0

        for index, row in data.iterrows():
            hist_stats = ast.literal_eval(row["hist_stats"])
            episodes_this_iter = row["episodes_this_iter"]
            rewards_this_iter = hist_stats[REWARD_KEY][-episodes_this_iter * n_trafficlights:]
            rewards_index = 0
            total_comm_cost, n_subrows = 0, 0

            for comm_type in COMM_TYPES:
                for comm_key in hist_stats:
                    if comm_type not in comm_key:
                        continue
                    # comm_costs = hist_stats[comm_key][:-episodes_this_iter]
                    comm_costs = hist_stats[comm_key][-episodes_this_iter:]
                    for comm_cost in  comm_costs:
                        pre["round"].append(row["round"])
                        pre["timesteps_total"].append(row["timesteps_total"])
                        pre["trainer"].append(row["trainer"])
                        pre["iteration"].append(row["training_iteration"])
                        if ranked is None:
                            pre["ranked"].append(row["ranked"])
                        else:
                            pre["ranked"].append(ranked)
                        pre["weight_aggr_fn"].append(row["weight_aggr_fn"])
                        pre["episode_reward_mean"].append(row["episode_reward_mean"])
                        pre["policy_reward"].append(rewards_this_iter[rewards_index // len(COMM_TYPES)])
                        rewards_index += 1
                        
                        # Add the communication cost here.
                        pre["comm_type"].append(comm_type)
                        fed_comm_type = (comm_type == TLS2EDGE_POLICY or comm_type == EDGE2TLS_POLICY)
                        if fed_comm_type and "FedRL" in row["trainer"]:
                            pre["comm_cost"].append(1)
                            total_comm_cost += 1
                        else:
                            pre["comm_cost"].append(comm_cost)
                            total_comm_cost += comm_cost

                        # TODO: Consider a "total" communication cost feature as well.
                        loop_count += 1
                        n_subrows += 1

            for _ in range(n_subrows):
                pre["total_comm_cost"].append(total_comm_cost)

        preprocessed_data = DataFrame.from_dict(pre)
        preprocessed_data.fillna("N/A", inplace=True)
        return preprocessed_data


    @staticmethod
    def _marl_preprocess(data: DataFrame, ranked: bool=None) -> DataFrame:
        pre = defaultdict(list)
        loop_count = 0
        for index, row in data.iterrows():
            hist_stats = ast.literal_eval(row["hist_stats"])
            episodes_this_iter = row["episodes_this_iter"]
            total_comm_cost, n_subrows = 0, 0

            for comm_type in COMM_TYPES:
                for comm_key in hist_stats:
                    if comm_type not in comm_key:
                        continue
                    
                    policy = comm_key.split("_")[1]
                    reward_key = f"policy_{policy}_reward"
                    # comm_costs = hist_stats[comm_key][:-episodes_this_iter]
                    # policy_rewards = hist_stats[reward_key][:-episodes_this_iter]
                    comm_costs = hist_stats[comm_key][-episodes_this_iter:]
                    policy_rewards = hist_stats[reward_key][-episodes_this_iter:]
                    
                    for policy_reward, comm_cost in zip(policy_rewards, comm_costs):
                        # pre["intersection"].append(row["intersection"])
                        pre["round"].append(row["round"])
                        pre["timesteps_total"].append(row["timesteps_total"])
                        pre["trainer"].append(row["trainer"])
                        pre["iteration"].append(row["training_iteration"])
                        if ranked is None:
                            pre["ranked"].append(row["ranked"])
                        else:
                            pre["ranked"].append(ranked)
                        pre["weight_aggr_fn"].append(row["weight_aggr_fn"])
                        pre["episode_reward_mean"].append(row["episode_reward_mean"])
                        pre["policy_reward"].append(policy_reward)
                        
                        # Add the communication cost here.
                        pre["comm_type"].append(comm_type)
                        fed_comm_type = (comm_type == TLS2EDGE_POLICY or comm_type == EDGE2TLS_POLICY)
                        if fed_comm_type and "FedRL" in row["trainer"]:
                            pre["comm_cost"].append(1)
                            total_comm_cost += 1
                        else:
                            pre["comm_cost"].append(comm_cost)
                            total_comm_cost += comm_cost

                        # TODO: Consider a "total" communication cost feature as well.
                        loop_count += 1
                        n_subrows += 1

            for _ in range(n_subrows):
                pre["total_comm_cost"].append(total_comm_cost)

        preprocessed_data = DataFrame.from_dict(pre)
        preprocessed_data.fillna("N/A", inplace=True)
        return preprocessed_data