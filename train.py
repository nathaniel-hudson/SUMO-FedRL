"""
For this document, we will setup a basic RL pipeline using our SinglePolicySumoEnv environment.

Refer to this recent and similar SumoRL tool that has an example for MARL using RlLib:
https://github.com/LucasAlegre/sumo-rl/blob/master/experiments/a3c_4x4grid.py

Ray RlLib agent training example.
https://github.com/ray-project/ray/blob/master/rllib/examples/custom_train_fn.py
"""
from fluri.trainer.ray.fed_agent import FedPolicyTrainer
from fluri.trainer.ray.multi_agent import MultiPolicyTrainer
from fluri.trainer.ray.single_agent import SinglePolicyTrainer
from os.path import join


if __name__ == "__main__":
    n_episodes = 100
    net_files = [
        # join("configs", "complex_inter", "complex_inter.net.xml"),
        join("configs", "example", "traffic.net.xml"), # TODO: Change this filename for better folder dirs.
        join("configs", "two_inter", "two_inter.net.xml")
    ]

    for net_file in net_files:
        for ranked in [True, False]:
            FedPolicyTrainer(fed_step=5, net_file=net_file, ranked=ranked).train(n_episodes)
            MultiPolicyTrainer(net_file=net_file, ranked=ranked).train(n_episodes)
            SinglePolicyTrainer(net_file=net_file, ranked=ranked).train(n_episodes)
