"""
For this document, we will setup a basic RL pipeline using our SinglePolicySumoEnv environment.
The RL tool we will incorporate is `stablebaselines`.

Refer to this recent and similar SumoRL tool that has an example for MARL using RlLib:
https://github.com/LucasAlegre/sumo-rl/blob/master/experiments/a3c_4x4grid.py

Ray RlLib agent training example.
https://github.com/ray-project/ray/blob/master/rllib/examples/custom_train_fn.py
"""
import argparse
import json
import os
import pandas as pd
import fluri.trainer.ray.fed_agent as fedrl
import fluri.trainer.ray.multi_agent as marl
import fluri.trainer.ray.single_agent as sarl

from datetime import datetime
from os.path import join

SINGLEAGENT = 1
MULTIAGENT = 2
FEDAGENT = 3


def get_args() -> argparse.ArgumentParser:
    args = argparse.ArgumentParser()
    args.add_argument("-f", "--netfile", default="two_inter", choices=["complex_inter", "example", "two_inter"], type=str)
    args.add_argument("-k", "--kind", default="marl", choices=["fedrl", "marl", "sarl"], 
                      type=str)
    args.add_argument("-n", "--num_rounds", default=50, type=int)
    args.add_argument("-r", "--ranked", action="store_true")
    args.add_argument("-f", "--fed_step", type=int, default=5)
    return args.parse_args()


def get_data_idx(kind: str, ranked: bool) -> int:
    default_json = {
        "fedrl": {"ranked": 0, "unranked": 0},
        "marl":  {"ranked": 0, "unranked": 0},
        "sarl":  {"ranked": 0, "unranked": 0},
    }

    ranked = "ranked" if ranked else "unranked"
    path = join("out", "data", "counter.json")
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(default_json, f)

    with open(path, "r") as f:
        data_json = json.load(f)

    idx = data_json[kind][ranked]
    data_json[kind][ranked] += 1
    with open(path, "w") as f:
        json.dump(data_json, f)

    return idx


def main(args: argparse.ArgumentParser) -> None:
    netfile, kind, num_rounds, ranked, fed_step = \
        args.netfile, args.kind, args.num_rounds, args.ranked, args.fed_step
    del args

    ranked_str = "ranked" if ranked else "unranked"
    idx = get_data_idx(kind, ranked)
    name = f"{ranked_str}_{idx}"
    if kind == "fedrl":
        results = fedrl.train(
            num_rounds, 
            fed_step=fed_step, 
            ranked=ranked, 
            model_name=name
        )
    elif kind == "marl":
        results = marl.train(
            num_rounds, 
            ranked=ranked, 
            model_name=name
        )
    elif kind == "sarl":
        results = sarl.train(
            num_rounds, 
            ranked=ranked, 
            model_name=name
        )
    else:
        raise ValueError("Invalid value for argument `kind`.")

    df = pd.DataFrame.from_dict(results)
    df["trained_on"] = None
    path = join("out", "data", kind, f"{kind}_{name}.csv")
    save_df(df, path)


def date_dir() -> str:
    date = datetime.now()
    return f"{date.year}-{date.month}-{date.day}_{date.hour}.{date.minute}"


def save_df(df: pd.DataFrame, path: str) -> None:
    path_dir = join(*path.split(os.sep)[:-1])
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    df.to_csv(path)


if __name__ == "__main__":
    main(get_args())

    # prompt = "Enter [1] single-agent, [2] multi-agent, or [3] fed-agent: "
    # kind = input(prompt)
    # while kind != str(SINGLEAGENT) and kind != str(MULTIAGENT) and kind != str(FEDAGENT):
    #     kind = input("Try again...", prompt)

    # kind = int(kind)
    # n_rounds = int(input("Enter the number of rounds for training: "))

    # if kind == SINGLEAGENT:
    #     results = sarl.train(n_rounds)
    #     results_df = pd.DataFrame.from_dict(results)
    #     filename = "sarl_data.csv"

    # elif kind == MULTIAGENT:
    #     results = marl.train(n_rounds)
    #     results_df = pd.DataFrame.from_dict(results)
    #     filename = "marl_data.csv"

    # elif kind == FEDAGENT:
    #     fed_step = int(input("Enter how many rounds before FedAvg: "))
    #     results = fedrl.train(n_rounds, fed_step)
    #     results_df = pd.DataFrame.from_dict(results)
    #     filename = "fedrl_data.csv"


    # path = join("out", "models", date_dir(), filename)
    # save_df(results_df, path)
