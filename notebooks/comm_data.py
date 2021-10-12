import ast
import pandas as pd
import sys

from collections import defaultdict
from pandas import DataFrame

sys.path.append("..")
from seal.trainer.communication import *

'''
TODO:
    + Currently, the SARL and MARL data were not properly output. They output data in a 
    policy-by-policy manner. This means that there are many redundancies.
    + This is fine for evaluating reward. However, for total communication cost, this is 
      a major liability because many communications will be counted several times over.
    + As such, we need to ignore records in the original data where the TEST-EVAL-POLICY is
      not the policy id.
    + ORRRRR, we could post-process the data and save it as a copy where only the relevant
      rows are stored.
'''

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
        intersection: str=None,
        return_type="brief"
    ) -> DataFrame:
        assert not all([path, data])
        if path is not None:
            data = pd.read_csv(path)

        if return_type == "brief":
            return PreprocessedCommData.__brief(data, intersection, ranked)
        elif return_type == "full":
            return PreprocessedCommData.__full(data, intersection, ranked)
        else:
            raise ValueError("Invalid 'return_type', must be either ['brief', 'full']")


    def __brief(data: DataFrame, intersection: str, ranked: bool) -> DataFrame:
        loop_count = 0
        n_subrows = 0
        standard_features = defaultdict(list)

        last_row = data.iloc[-1]
        # hist_stats = ast.literal_eval(last_row["hist_stats"])

        for i, row in data.iterrows():
            hist_stats = ast.literal_eval(row["hist_stats"])
            ep_this_iter = row["episodes_this_iter"]
            comm_type_costs = defaultdict(int)
            for comm_type in COMM_TYPES:
                # print(f"comm_type = {comm_type}")
                for hist_key in hist_stats:
                    if f"={comm_type}" in hist_key:
                        # print(f"(comm_type={comm_type}, hist_key={hist_key})", end=", ")
                        # a, b = i * ep_this_iter, (i+1) * ep_this_iter
                        if "policy" in comm_type and row["fed_round"]:
                            comm_type_costs[comm_type] += 1
                        else:
                            costs = hist_stats[hist_key][-ep_this_iter:]
                            if sum(costs) > 0 and comm_type == "edge-to-tls-rank-comms" \
                            and row["ranked"] == False:
                                print(f"ERROR: costs = {costs}")
                            comm_type_costs[comm_type] += sum(costs)
                        # a, b = i * episodes_this_iter, (i+1) * episodes_this_iter
                        # comm_type_costs[comm_type] += sum(hist_stats[hist_key][a:b])

            # print(comm_type_costs)
            # exit(0)
            total_comm_cost = sum(comm_type_costs.values())

            for comm_type in COMM_TYPES:
                standard_features["round"].append(row["round"])
                standard_features["intersection"].append(intersection)
                standard_features["timesteps_total"].append(row["timesteps_total"])
                standard_features["trainer"].append(row["trainer"])
                standard_features["iteration"].append(row["training_iteration"])
                standard_features["ranked"].append(row["ranked"] if ranked is None else ranked)
                standard_features["weight_aggr_fn"].append(row["weight_aggr_fn"])
                standard_features["episode_reward_mean"].append(row["episode_reward_mean"])

                standard_features["comm_cost"].append(comm_type_costs[comm_type])
                standard_features["comm_type"].append(comm_type)
                standard_features["total_comm_cost"].append(total_comm_cost)

        df = DataFrame.from_dict(standard_features)
        
        df.query("trainer == 'FedRL' and intersection == 'grid-3x3' and ranked == False and comm_type == 'edge-to-tls-rank-comms'").head()

        return df
            

        # last_row = data.iloc[-1]
        # is_sarl = last_row["trainer"] == "SARL"
        # hist_stats = ast.literal_eval(last_row["hist_stats"])
        # policy_features = defaultdict(list)
        # for comm_type in COMM_TYPES:
        #     for comm_key in hist_stats:
        #         if f"={comm_type}" not in comm_key:
        #             continue

        #         policy = comm_key.split("_")[1]
        #         reward_key = "policy_sarl-policy_reward" if is_sarl else f"policy_{policy}_reward"
        #         comm_costs = hist_stats[comm_key]
        #         policy_rewards = hist_stats[reward_key]

        #         if comm_type == "edge-to-tls-rank-comms" and last_row["ranked"] == False:
        #             print(f"comm_costs:\n{comm_costs}\n")

        #         total_comm_cost = 0
        #         for _round, (cost, reward) in enumerate(zip(comm_costs, policy_rewards)):
        #             policy_features["round"].append(_round)
        #             policy_features["comm_cost"].append(cost)
        #             policy_features["comm_type"].append(comm_type)
        #             policy_features["policy_reward"].append(reward)
        #             total_comm_cost += cost
        #         for _ in range(len(comm_costs)):
        #             policy_features["total_comm_cost"].append(total_comm_cost)

        # standard_df = DataFrame.from_dict(standard_features)
        # policy_df = DataFrame.from_dict(policy_features)
        # return standard_df.merge(policy_df, on="round")


    def __full(data: DataFrame, intersection: str, ranked: bool) -> DataFrame:
        pre = defaultdict(list)
        loop_count = 0
        n_subrows = 0
        for index, row in data.iterrows():
            hist_stats = ast.literal_eval(row["hist_stats"])
            episodes_this_iter = row["episodes_this_iter"]
            is_sarl = row["trainer"] == "SARL"
            total_comm_cost = 0

            for comm_type in COMM_TYPES:
                for comm_key in hist_stats:
                    if comm_type not in comm_key:
                        continue
                    
                    policy = comm_key.split("_")[1]
                    reward_key = "policy_sarl-policy_reward" if is_sarl else f"policy_{policy}_reward"
                    # reward_key = "episode_reward" ## Possible option under SARL.

                    comm_costs = hist_stats[comm_key][-episodes_this_iter:]
                    policy_rewards = hist_stats[reward_key][-episodes_this_iter:]
                    
                    for policy_reward, comm_cost in zip(policy_rewards, comm_costs):
                        pre["round"].append(row["round"])
                        pre["intersection"].append(intersection)
                        pre["timesteps_total"].append(row["timesteps_total"])
                        pre["trainer"].append(row["trainer"])
                        pre["iteration"].append(row["training_iteration"])
                        pre["ranked"].append(row["ranked"] if ranked is None else ranked)
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




















    @staticmethod
    def preprocess_v1(
        path: str=None,
        data: DataFrame=None, 
        ranked: bool=None, 
        intersection: str=None,
        is_sarl: bool=False
    ) -> DataFrame:
        assert not all([path, data]), "Can only provide either a DataFrame or a path to a .csv file."
        
        if path is not None:
            data = pd.read_csv(path)
        elif data is None:
            raise ValueError("Must either provide a path to a .csv file or a DataFrame.")
        
        if is_sarl:
            return PreprocessedCommData._sarl_preprocess(data, ranked=ranked, intersection=intersection)
        else:
            return PreprocessedCommData._marl_preprocess(data, ranked=ranked, intersection=intersection)

    @staticmethod
    def _sarl_preprocess(
        data: DataFrame, 
        ranked: bool=None, 
        intersection: str=None
    ) -> DataFrame:
        
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
                        pre["intersection"].append(intersection)
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
    def _marl_preprocess(
        data: DataFrame, 
        ranked: bool=None, 
        intersection: str=None
    ) -> DataFrame:
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
                        pre["intersection"].append(intersection)
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