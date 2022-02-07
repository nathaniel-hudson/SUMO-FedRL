"""
For this document, we will setup a basic RL pipeline using our SinglePolicySumoEnv environment.

Refer to this recent and similar SumoRL tool that has an example for MARL using RlLib:
https://github.com/LucasAlegre/sumo-rl/blob/master/experiments/a3c_4x4grid.py

Ray RlLib agent training example.
https://github.com/ray-project/ray/blob/master/rllib/examples/custom_train_fn.py
"""
import os
from netfiles import *
from seal.logging import *
from seal.trainer.fed_agent import FedPolicyTrainer
from seal.trainer.multi_agent import MultiPolicyTrainer
from seal.trainer.single_agent import SinglePolicyTrainer
from os.path import join

OUT_PREFIX = "v3"
random_routes_config = {}
trainer_kwargs = {
    # =========================================================== #
    # Non-Algorithm Trainer Arguments (i.e., not related to PPO). #
    # =========================================================== #
    "horizon": 360,  # 240, # NOTE: Maybe disable this?
    # "timesteps_per_iteration":  240,
    # "batch_mode": "truncate_episodes",
    # "rollout_fragment_length": 240,
    # "train_batch_size": 240,

    # ====================== #
    # PPO Trainer Arguments. #
    # ====================== #
    # "sgd_minibatch_size": 30,
}


if __name__ == "__main__":
    n_episodes = 50
    fed_step = 1
    NET_FILES = [
        DOUBLE_LOOP,
        GRID_3x3,
        GRID_5x5
    ]
    RANKED = [
        # True,
        False
    ]

    status = "Training with `{}`! (netfile='{}', ranked={})"
    for net_file in NET_FILES:
        for ranked in RANKED:
            intersection = net_file.split(os.sep)[-1]

            # Federated trainer using the 'traffic' aggregation function.
            # logging.info(status.format(
            #     "FedPolicyTrainer (aggr='traffic')", intersection, ranked))
            # traffic_aggr_prefix = f"{OUT_PREFIX}_traffic-aggr"
            # FedPolicyTrainer(fed_step=fed_step, net_file=net_file, ranked=ranked,
            #                  out_prefix=traffic_aggr_prefix,
            #                  trainer_kwargs=trainer_kwargs,
            #                  weight_fn="traffic").\
            #     train(n_episodes)

            # Federated Trainer using the 'negative reward' aggregation function.
            # logging.info(status.format(
            #     "FedPolicyTrainer (aggr='neg_reward')", intersection, ranked))
            # traffic_aggr_prefix = f"{OUT_PREFIX}_neg-reward-aggr"
            # FedPolicyTrainer(fed_step=fed_step, net_file=net_file, ranked=ranked,
            #                  out_prefix=traffic_aggr_prefix,
            #                  trainer_kwargs=trainer_kwargs,
            #                  weight_fn="neg_reward").\
            #     train(n_episodes)

            # Federated Trainer using the 'positive reward' aggregation function.
            logging.info(status.format(
                "FedPolicyTrainer (aggr='pos_reward')", intersection, ranked))
            traffic_aggr_prefix = f"{OUT_PREFIX}_pos-reward-aggr"
            FedPolicyTrainer(fed_step=fed_step, net_file=net_file, ranked=ranked,
                             out_prefix=traffic_aggr_prefix,
                             trainer_kwargs=trainer_kwargs,
                             weight_fn="pos_reward").\
                train(n_episodes)

            # Federated Trainer using the 'naive' weighting aggregation function.
            logging.info(status.format(
                "FedPolicyTrainer (aggr='naive')", intersection, ranked))
            traffic_aggr_prefix = f"{OUT_PREFIX}_naive-aggr"
            FedPolicyTrainer(fed_step=fed_step, net_file=net_file, ranked=ranked,
                             out_prefix=traffic_aggr_prefix,
                             trainer_kwargs=trainer_kwargs,
                             weight_fn="naive").\
                train(n_episodes)

            # MultiPolicy Trainer.
            logging.info(status.format(
                "MultiPolicyTrainer", intersection, ranked))
            MultiPolicyTrainer(net_file=net_file, ranked=ranked,
                               out_prefix=OUT_PREFIX, trainer_kwargs=trainer_kwargs).\
                train(n_episodes)

            # SinglePolicy Trainer.
            logging.info(status.format(
                "SinglePolicyTrainer", intersection, ranked))
            SinglePolicyTrainer(net_file=net_file, ranked=ranked,
                                out_prefix=OUT_PREFIX, trainer_kwargs=trainer_kwargs).\
                train(n_episodes)
