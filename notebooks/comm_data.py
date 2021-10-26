import ast
import numpy as np
import pandas as pd
import sys

from collections import defaultdict
from pandas import DataFrame
from typing import Any

sys.path.append("..")
from seal.trainer.communication import *


def read_file(path: str) -> DataFrame:
    if path.endswith("csv"):
        return pd.read_csv(path)
    elif path.endswith("json"):
        return pd.read_json(path)
    elif path.endswith("xls") or path.endswith("xlsx"):
        return pd.read_excel(path)
    else:
        raise ValueError("Unsupported filetype extension for 'path'.")


def preprocess(path: str, intersection: str) -> DataFrame:
    data = read_file(path)
    features = defaultdict(list)
    # is_sarl = next(iter(data["trainer"])) == "SARL"
    ranked = next(iter(data["ranked"]))
    trainer = next(iter(data["trainer"]))
    episode_len_mean = next(iter(data["episode_len_mean"]))
    policies = set([
        key.split("_")[1]
        for key in ast.literal_eval(next(iter(data["hist_stats"])))
        if key.startswith("policy_") and key.endswith("_reward")
    ])
    trafficlights = set([
        key.split("_")[1]
        for key in ast.literal_eval(next(iter(data["hist_stats"])))
        if ("policy_" in key) and ("_comm" in key)
    ])

    def __add_standard_features(row: pd.Series) -> None:
        features["trainer"].append(row["trainer"])
        features["iteration"].append(row["training_iteration"])
        # features["episode"].append(row["training_iteration"])
        features["round"].append(row["round"])
        features["intersection"].append(intersection)
        features["timesteps_total"].append(row["timesteps_total"])
        features["ranked"].append(ranked)
        features["weight_aggr_fn"].append(row["weight_aggr_fn"])
        features["episode_reward_mean"].append(row["episode_reward_mean"])

    def __add_comm_costs(
        hist_stats: dict,
        comm_key: str,
        comm_type: str,
        episodes_this_iter: int,
        episode_index: int
    ) -> float:
        if "policy" in comm_type and trainer == "FedRL":
            if episode_index == episodes_this_iter-1:
                features[comm_type].append(1)
                return 1
            else:
                features[comm_type].append(0)
                return 0
        elif ranked and comm_type == TLS2EDGE_OBS:
            comm_cost = hist_stats[comm_key][-episodes_this_iter:][episode_index]
            features[comm_type].append(comm_cost)
            return comm_cost
        else:
            comm_cost = hist_stats[comm_key][-episodes_this_iter:][episode_index]
            features[comm_type].append(comm_cost)
            return comm_cost

    for _, row in data.iterrows():
        hist_stats = ast.literal_eval(row["hist_stats"])
        episodes_this_iter = row["episodes_this_iter"]
        episodes_total = row["episodes_total"]
        episodes = range(episodes_total-episodes_this_iter, episodes_total)
        for (ep_index, ep) in enumerate(episodes):
            for tls in trafficlights:
                total_comm_cost = 0
                for comm_type in COMM_TYPES:
                    comm_key = f"policy_{tls}_comm={comm_type}"
                    reward_key = f"policy_{tls}_reward"
                    total_comm_cost += __add_comm_costs(
                        hist_stats, comm_key, comm_type, episodes_this_iter, ep_index
                    )
                __add_standard_features(row)
                features["episode"].append(ep)
                features["total_comm_cost"].append(total_comm_cost)
                features["tls"].append(tls)

    try:
        preprocessed_data = DataFrame.from_dict(features)
    except:
        for key in features:
            print(key, len(features[key]))

    preprocessed_data.fillna("N/A", inplace=True)
    preprocessed_data["intersection"].replace({
        "double":   "Double",
        "grid-3x3": "Grid-3x3",
        "grid-5x5": "Grid-5x5"
    }, inplace=True)
    preprocessed_data["trainer"].replace({
        "FedRL": "Federated",
        "MARL":  "Decentralized",
        "SARL":  "Centralized"
    }, inplace=True)
    return preprocessed_data
