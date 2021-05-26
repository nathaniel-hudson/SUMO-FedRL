import glob
import os
import pandas as pd
import ray

from collections import defaultdict
from fluri.trainer.ray.base import BaseTrainer
from fluri.sumo.sumo_env import SumoEnv
from fluri.sumo.multi_agent_env import MultiPolicySumoEnv
from fluri.sumo.single_agent_env import SinglePolicySumoEnv
from fluri.trainer.ray.fed_agent import FedPolicyTrainer
from fluri.trainer.ray.multi_agent import MultiPolicyTrainer
from fluri.trainer.ray.single_agent import SinglePolicyTrainer
from os.path import join
from pandas import DataFrame
from typing import Dict, Tuple
from ray.rllib import agents as raygents
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo import PPOTrainer

# TODO: For now, this file only supports PPO agents. Fix this later.


def load_config(netfile: str, kind: str, ranked: bool) -> Dict:
    config = {
        "env_config": {
            "gui": False,
            "net-file": netfile,
            "rand_routes_on_reset": False,
            "ranked": ranked,
        },
        "framework": "torch",
        "log_level": "ERROR",
        "lr": 0.001,
        "num_workers": 0,
    }
    if (kind == "marl") or (kind == "fedrl"):
        dummy_env = MultiPolicySumoEnv(config={
            "gui": False,
            "net-file": netfile,
            "ranked": ranked,
        })
        obs_space = dummy_env.observation_space
        act_space = dummy_env.action_space
        policy = {
            agent_id: (PPOTorchPolicy, obs_space, act_space, {"gamma": 0.95})
            for agent_id in dummy_env._observe()
        }
        config["multiagent"] = {
            "policies": policy,
            "policy_mapping_fn": lambda agent_id: agent_id
        }
    return config


def load_env(netfile: str, kind: str, ranked: bool) -> Tuple[SumoEnv, Dict]:
    if kind == "sarl":
        env_class = SinglePolicySumoEnv
    elif (kind == "marl") or (kind == "fedrl"):
        env_class = MultiPolicySumoEnv
    else:
        raise ValueError("Invalid value for parameter `kind`.")
    # env = env_class({
    #     "gui": False, "net-file": netfile, "rand_routes_on_reset": True,
    # })
    config = load_config(netfile, kind, ranked)
    return env_class, config


def load_agent(
    env: SumoEnv,
    config: Dict,
    checkpoint: str,
    agent_type: str="ppo"
) -> raygents.Trainer:

    if agent_type == "ppo":
        agent_class = PPOTrainer
    else:
        # TODO: Add the others.
        raise ValueError("Invalid agent type.")

    agent = agent_class(env=env, config=config)
    agent.restore(checkpoint)
    return agent


def marl_step(env, obs, agent) -> Tuple:
    action = {agent_id: agent.compute_action(agent_obs, policy_id=agent_id)
              for agent_id, agent_obs in obs.items()}
    next_obs, reward, done, info = env.step(action)
    done = next(iter(done.values()))
    return (next_obs, reward, done, info)


def sarl_step(env, obs, agent) -> Tuple:
    action = agent.compute_action(obs)
    next_obs, reward, done, info = env.step(action)
    return (next_obs, reward, done, info)


def eval(
    netfile: str,
    checkpoint: str,
    kind: str,
    ranked: bool,
    n_episodes: int=1,
    agent_type: str="ppo"
) -> DataFrame:
    ray.init()
    env_class, config = load_env(netfile=netfile, kind=kind, ranked=ranked)
    agent = load_agent(env_class, config, checkpoint, agent_type=agent_type)
    eval_data = defaultdict(list)
    last_reward = defaultdict(float)

    env = env_class(config["env_config"])
    for ep in range(1, n_episodes+1):
        obs = env.reset()
        done, step = False, 0
        last_reward = defaultdict(float)
        while not done:
            if (kind == "marl") or (kind == "fedrl"):
                obs, reward, done, _ = marl_step(env, obs, agent)
                for tls_id, r in reward.items():
                    eval_data["ranked"].append(ranked)
                    eval_data["agent_type"].append(agent_type)
                    eval_data["kind"].append(kind)
                    eval_data["tls_id"].append(tls_id)
                    eval_data["episode"].append(ep)
                    eval_data["cum_reward"].append(r + last_reward[tls_id])
                    eval_data["reward"].append(r)
                    eval_data["step"].append(step)
                    # last_reward[ep, tls_id] = eval_data["cum_reward"][-1]
                    last_reward[tls_id] = eval_data["cum_reward"][-1]
            elif kind == "sarl":
                obs, reward, done, _ = sarl_step(env, obs, agent)
                eval_data["ranked"].append(ranked)
                eval_data["agent_type"].append(agent_type)
                eval_data["kind"].append(kind)
                eval_data["episode"].append(ep)
                eval_data["cum_reward"].append(reward + last_reward["-"])
                eval_data["reward"].append(reward)
                eval_data["step"].append(step)
                # last_reward[ep] = eval_data["cum_reward"][-1]
                last_reward["-"] = eval_data["cum_reward"][-1]
            else:
                raise ValueError("Invalid value for parameter `kind`.")
            step += 1
    env.close()
    ray.shutdown()
    return DataFrame.from_dict(eval_data)


def load_last_checkpoint(nettype: str, kind: str, ranked: bool) -> str:
    assert nettype in ["complex_inter", "single_inter", "two_inter"]
    assert kind in ["fedrl", "marl", "sarl"]
    assert isinstance(ranked, bool)

    if kind == "fedrl":
        kind = "FedRL"
    elif kind == "marl":
        kind = "MARL"
    else:
        kind = "SARL"
    ranked = "ranked" if ranked else "unranked"
    dirs = glob.glob(join("out", "models", kind, nettype, f"{ranked}*"))
    recent_dir = sorted(dirs)[-1]
    dirs = glob.glob(join(recent_dir, "checkpoint*"))
    checkpoint_dir = sorted(dirs)[-1]
    checkpoint_set = set(glob.glob(join(checkpoint_dir, "*"))) - \
        set(glob.glob(join(checkpoint_dir, "*.*")))
    checkpoint = next(iter(checkpoint_set))
    return checkpoint


if __name__ == "__main__":
    net_files = [
        join("configs", "complex_inter", "complex_inter.net.xml"),
        join("configs", "single_inter", "single_inter.net.xml"),
        join("configs", "two_inter", "two_inter.net.xml")
    ]

    dataframes = []
    for netfile in net_files:
        for kind in ["fedrl", "marl", "sarl"]:
            for ranked in [True, False]:
                nettype = netfile.split(os.sep)[1]
                checkpoint = load_last_checkpoint(nettype=nettype, kind=kind,
                                                  ranked=ranked)
                df = eval(netfile, checkpoint, kind, ranked, n_episodes=10)
                dataframes.append(df)

    print(">> EVAL DONE!")
    final_df = pd.concat(dataframes)
    final_df.to_csv("eval_data.csv")
