"""
For this document, we will setup a basic RL pipeline using our SinglePolicySumoEnv environment.
The RL tool we will incorporate is `stablebaselines`.

Refer to this recent and similar SumoRL tool that has an example for MARL using RlLib:
https://github.com/LucasAlegre/sumo-rl/blob/master/experiments/a3c_4x4grid.py

Ray RlLib agent training example.
https://github.com/ray-project/ray/blob/master/rllib/examples/custom_train_fn.py
"""
from argparse import ArgumentError
from fluri.sumo.multi_agent_env import MultiPolicySumoEnv
import gym
from numpy.lib.function_base import append
import os
import random
import ray
import ray.rllib.agents.ppo as ppo

from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.tune.registry import register_env

from fluri.sumo.utils.random_routes import generate_random_routes


from collections import defaultdict
from fluri.sumo.single_agent_env import SinglePolicySumoEnv
from fluri.sumo.kernel.kernel import SumoKernel
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from os.path import join


# ====================================================================================== #
# .................................. STABLEBASELINES ................................... #
# ====================================================================================== #

def train(config, total_timesteps: int=int(2e6)):
    sim = SumoKernel(config=config)
    env = SinglePolicySumoEnv(sim)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save("simple_model")
    env.close()

def stable_baselines_train(total_timesteps: int=int(2e6)) -> None:
    path = join("configs", "example")
    config = {
        "gui": False,
        "net-file": join(path, "traffic.net.xml"),
        "route-files": join(path, "traffic.rou.xml"),
        "additional-files": join(path, "traffic.det.xml"),
        "tripinfo-output": join(path, "tripinfo.xml")
    }
    train(config, total_timesteps)

# ====================================================================================== #
# ..................................... RAY RlLIB ...................................... #
# ====================================================================================== #

def singleagent_ray_train(n_rounds: int=10) -> None:
    """Single-agent reinforcement learning with Ray's RlLib.

    Parameters
    ----------
    n_rounds : int, optional
        Number of training rounds, by default 10.
    """
    ray.init()
    trainer = PPOTrainer(env=SinglePolicySumoEnv, config={
        "lr": 0.01,
        "num_gpus": 0,
        "num_workers": 0, ## NOTE: For some reason, this *needs* to be 0.
        "framework": "torch",
        "env_config": {
            "gui": False,
            "net-file": join("configs", "two_inter", "two_inter.net.xml"),
            "rand_routes_on_reset": False, ## NOTE: Checking if constant routefile works.
        }
    })
    train_data = {}
    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    out_file =join("out", "models", "simple-ray")

    data = defaultdict(list)

    for n in range(n_rounds):
        result = trainer.train()
        state = trainer.save(out_file)
        print(status.format(
            n + 1, result["episode_reward_min"], result["episode_reward_mean"],
            result["episode_reward_max"], result["episode_len_mean"], 
            out_file.split(os.sep)[-1]
        ))

    state = trainer.save(join("out", "models", "SARL-ray"))
    trainer.stop()
    ray.shutdown()

## TODO: We need to look into exactly HOW to get random route generation to not screw up
##       Ray's RlLib training process. Not sure why they're interfering, but no issue
##       rises when we do not constantly construct random route files.
def multiagent_ray_train(n_rounds: int=10) -> None:
    """Multi-agent reinforcement learning with Ray's RlLib.

    Parameters
    ----------
    n_rounds : int, optional
        Number of training rounds, by default 10.
    """
    dummy_env = MultiPolicySumoEnv(config={
        "gui": False,
        "net-file": join("configs", "two_inter", "two_inter.net.xml")
    })
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space

    ray.init()
    trainer = PPOTrainer(env=MultiPolicySumoEnv, config={
        "multiagent": {
            "policies": {
                "0": (PPOTorchPolicy, obs_space, act_space, {"gamma": 0.95})
            },
            "policy_mapping_fn": lambda _: "0",
        },
        "lr": 0.001,
        "num_gpus": 0,
        "num_workers": 0, ## NOTE: For some reason, this *needs* to be 0.
        "framework": "torch",
        "env_config": {
            "gui": False,
            "net-file": join("configs", "two_inter", "two_inter.net.xml")
        }
    })
    train_data = {}
    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    out_file =join("out", "models", "simple-ray")

    for n in range(n_rounds):
        result = trainer.train()
        state = trainer.save(out_file)
        print(status.format(
            n + 1, result["episode_reward_min"], result["episode_reward_mean"],
            result["episode_reward_max"], result["episode_len_mean"], 
            out_file.split(os.sep)[-1]
        ))

    state = trainer.save(join("out", "models", "MARL-ray"))
    trainer.stop()
    ray.shutdown()

# ====================================================================================== #

if __name__ == "__main__":
    SINGLEAGENT = 1
    MULTIAGENT  = 2

    kind = input("Designate training kind: [1] single-agent or [2] multi-agent: ")
    while kind != "1" and kind != "2":
        kind = input("Try again... Enter [1] single-agent or [2] multi-agent: ")
    kind = int(kind)
    n_rounds = int(input("Enter the number of rounds for training: "))
    
    if kind == SINGLEAGENT:
        singleagent_ray_train(n_rounds=n_rounds)
    elif kind == MULTIAGENT:
        multiagent_ray_train(n_rounds=n_rounds)
    else:
        raise ArgumentError("Illegal argument provided to argparser.")