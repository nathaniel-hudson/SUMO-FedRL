import pandas as pd
import fluri.trainer.ray.fed_agent as fedrl
import fluri.trainer.ray.multi_agent as marl
import fluri.trainer.ray.single_agent as sarl

from collections import defaultdict
from datetime import datetime
from fluri.trainer.const import *
from fluri.sumo.single_agent_env import SinglePolicySumoEnv
from fluri.sumo.multi_agent_env import MultiPolicySumoEnv
from gym import Space
from os.path import join
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from typing import Tuple


def get_policies() -> Tuple[Space, Space]:
    dummy_env = MultiPolicySumoEnv(config={
        "gui": False,
        "net-file": join("configs", "two_inter", "two_inter.net.xml")
    })
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space
    return {
        agent_id: (PPOTorchPolicy, obs_space, act_space, {"gamma": 0.95})
        for agent_id in dummy_env._observe()
    }


def eval(n_episodes: int=1) -> None:
    agent = PPOTrainer(config=config, env=env_class)
    agent.restore(checkpoint_path)

    policies = get_policies()
    agent = PPOTrainer(env=MultiPolicySumoEnv, config={
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": lambda agent_id: agent_id,
        },
        "lr": 0.001,
        "num_gpus": 0,
        "num_workers": 0,  # NOTE: For some reason, this *needs* to be 0.
        "framework": "torch",
        "log_level": "ERROR",
        "env_config": DEFAULT_ENV_CONFIG,
    })

    # STEP 2: Run through the simulation.
    eval_data = defaultdict(list)
    last_reward = defaultdict(float)
    for ep in range(n_episodes):
        env.reset()
        done, step = False, 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

            eval_data["cum_reward"].append(reward + last_reward[ep])
            eval_data["reward"].append(reward)
            eval_data["step"].append(step)
            eval_data["episode"].append(ep)
            last_reward[ep] = eval_data["cum_reward"][-1]

            step += 1
    env.close()
