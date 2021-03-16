"""
For this document, we will setup a basic RL pipeline using our SinglePolicySumoEnv environment.
The RL tool we will incorporate is `stablebaselines`.

Refer to this recent and similar SumoRL tool that has an example for MARL using RlLib:
https://github.com/LucasAlegre/sumo-rl/blob/master/experiments/a3c_4x4grid.py

Ray RlLib agent training example.
https://github.com/ray-project/ray/blob/master/rllib/examples/custom_train_fn.py
"""
import pandas as pd
import fluri.trainer.ray.fed_agent as fedrl
import fluri.trainer.ray.multi_agent as marl
import fluri.trainer.ray.single_agent as sarl


if __name__ == "__main__":
    SINGLEAGENT = 1
    MULTIAGENT = 2
    FEDAGENT = 3

    kind = input(
        "Try again... Enter [1] single-agent, [2] multi-agent, or [3] fed-agent: ")
    while kind != "1" and kind != "2" and kind != "3":
        kind = input(
            "Try again... Enter [1] single-agent, [2] multi-agent, or [3] fed-agent: ")
    kind = int(kind)
    n_rounds = int(input("Enter the number of rounds for training: "))

    if kind == SINGLEAGENT:
        results = sarl.train(n_rounds)
        results_df = pd.DataFrame.from_dict(results)
        results_df.to_csv("sarl_data.csv")

    elif kind == MULTIAGENT:
        results = marl.train(n_rounds)
        results_df = pd.DataFrame.from_dict(results)
        results_df.to_csv("marl_data.csv")

    elif kind == FEDAGENT:
        fed_step = int(input("Enter how many rounds before FedAvg: "))
        results = fedrl.train(n_rounds, fed_step)
        results_df = pd.DataFrame.from_dict(results)
        results_df.to_csv("fedrl_data.csv")
