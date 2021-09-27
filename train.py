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

OUT_PREFIX = "new"
random_routes_config = {}
trainer_kwargs = {
    # Non-Algorithm Trainer Arguments (i.e., not related to PPO).
    "horizon": 360, # 240,
    # "timesteps_per_iteration":  240,
    # "batch_mode": "truncate_episodes",
    # "rollout_fragment_length": 240,
    # "train_batch_size": 240,

    # PPO Trainer Arguments.
    # "sgd_minibatch_size": 30,
}


if __name__ == "__main__":
    n_episodes = 50
    fed_step =  1
    NET_FILES = [
        GRID_3x3,
        GRID_5x5,
        DOUBLE_LOOP
    ]
    RANKED = [
        True, 
        False
    ]

    status = ">>> Training with `{}`! (netfile='{}', ranked={})"
    for net_file in NET_FILES:
        for ranked in RANKED:
            ## NOTE: When I tried running this, it stopped with FedPolicyTrainer and gave
            ## the following warning: "WARNING:root:Nan or Inf found in input tensor".
            ## I'm not sure if this is because of federated averaging or if it's from
            ## the observations.
            intersection = net_file.split(os.sep)[-1]

            print(status.format("FedPolicyTrainer", intersection, ranked))
            FedPolicyTrainer(fed_step=fed_step, net_file=net_file, ranked=ranked, 
                             out_prefix=OUT_PREFIX, trainer_kwargs=trainer_kwargs).\
                train(n_episodes)

            print(status.format("MultiPolicyTrainer", intersection, ranked))
            MultiPolicyTrainer(net_file=net_file, ranked=ranked,  
                               out_prefix=OUT_PREFIX, trainer_kwargs=trainer_kwargs).\
                train(n_episodes)

            print(status.format("SinglePolicyTrainer", intersection, ranked))
            SinglePolicyTrainer(net_file=net_file, ranked=ranked,  
                                out_prefix=OUT_PREFIX, trainer_kwargs=trainer_kwargs).\
                train(n_episodes)
