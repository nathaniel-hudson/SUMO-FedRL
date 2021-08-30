"""
For this document, we will setup a basic RL pipeline using our SinglePolicySumoEnv environment.

Refer to this recent and similar SumoRL tool that has an example for MARL using RlLib:
https://github.com/LucasAlegre/sumo-rl/blob/master/experiments/a3c_4x4grid.py

Ray RlLib agent training example.
https://github.com/ray-project/ray/blob/master/rllib/examples/custom_train_fn.py
"""
import os
from netfiles import *
from seal.trainer.fed_agent import FedPolicyTrainer
from seal.trainer.multi_agent import MultiPolicyTrainer
from seal.trainer.single_agent import SinglePolicyTrainer
from os.path import join

random_routes_config = {

}

if __name__ == "__main__":
    n_episodes = 5 #100
    fed_step =  2 # 5
    # n_episodes = 1
    # fed_step = 1
    net_files = [
        GRID_3x3,
        # GRID_5x5,
        # DOUBLE_LOOP
    ]

    for net_file in net_files:
        for ranked in [True, False]:
            ## NOTE: When I tried running this, it stopped with FedPolicyTrainer and gave
            ## the following warning: "WARNING:root:Nan or Inf found in input tensor".
            ## I'm not sure if this is because of federated averaging or if it's from
            ## the observations.
            intersection = net_file.split(os.sep)[-1]
            print(f">> Training with `FedPolicyTrainer`! (netfile='{intersection}')")
            FedPolicyTrainer(fed_step=fed_step, net_file=net_file, ranked=ranked).\
                train(n_episodes)

            print(">> Training with `MultiPolicyTrainer`!")
            MultiPolicyTrainer(net_file=net_file, ranked=ranked).\
                train(n_episodes)

            print(">> Training with `SinglePolicyTrainer`!")
            SinglePolicyTrainer(net_file=net_file, ranked=ranked).\
                train(n_episodes)
