"""
For this document, we will setup a basic RL pipeline using our SingleSumoEnv environment.
The RL tool we will incorporate is `stablebaselines`.
"""
import gym
from numpy.lib.function_base import append
import ray

from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from fluri.sumo.utils.random_routes import generate_random_routes

"""
Ray RlLib agent training example.
https://github.com/ray-project/ray/blob/master/rllib/examples/custom_train_fn.py
"""


from fluri.sumo.single_agent_env import SingleSumoEnv
from fluri.sumo.kernel.kernel import SumoKernel
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from os.path import join

WORLD_SHAPE = (25, 25)


# ============================================================== #
# ...................... STABLEBASELINES *********************** #
# ============================================================== #

data = None
def init_data() -> None:
    global data
    data = {
        "action": [],
        "step": [],
        "policy": []
    }

def add_record(action, step, policy) -> None:
    global data
    data["action"].append(action[0])
    data["step"].append(step)
    data["policy"].append(policy)

def train(config, total_timesteps: int=int(2e6)):
    init_data()
    sim = SumoKernel(config=config)
    env = SingleSumoEnv(sim, world_dim=WORLD_SHAPE)

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

# ============================================================== #
# ......................... RAY RlLIB .......................... #
# ============================================================== #

import pandas as pd

def append_dict(dict1, dict2):
    for key in dict2:
        if key not in dict1:
            dict1[key] = [dict2[key]]
        elif isinstance(dict1[key], list):
            dict1[key].append(dict2[key])
        else:
            dict1[key] = [dict1[key], dict2[key]]

def ray_train_fn(config, reporter, n_training_steps: int=10): #int(2e6)):
    # Train for 100 iterations with high LR
    agent = PPOTrainer(env="SumoEnv", config=config)
    train_data = {}
    for _ in range(n_training_steps):
        result = agent.train()
        append_dict(train_data, result)
        # result["phase"] = 1
        reporter(**result)
        # phase1_time = result["timesteps_total"]
    state = agent.save(join("out", "models", "simple-ray"))
    agent.stop()
    
    train_data_df = pd.DataFrame.from_dict(train_data)
    train_data_df.to_csv(join("out", "train_data", "simple-ray.csv"))

if __name__ == "__main__":
    import os
    import ray.rllib.agents.ppo as ppo

    ray.init()

    # config = {
    #     "lr": 0.01,
    #     "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    #     "num_workers": 0,
    #     "framework": "torch",
    # }
    # resources = PPOTrainer.default_resource_request(config).to_json()
    # tune.run(ray_train_fn, resources_per_trial=resources, config=config)
    # config = ppo.DEFAULT_CONFIG.copy()
    # config["log_level"] = "WARN"
    
    agent = PPOTrainer(env=SingleSumoEnv, config={
        "lr": 0.01,
        "num_gpus": 0,
        "num_workers": 0,
        "framework": "torch",
        "env_config": {
            "gui": False,
            "net-file": join("configs", "two_inter", "two_inter.net.xml")
        }
    })
    train_data = {}
    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    out_file =join("out", "models", "simple-ray")

    for n in range(2):
        result = agent.train()
        state = agent.save(out_file)
        append_dict(train_data, result)
        print(status.format(
            n + 1, result["episode_reward_min"], result["episode_reward_mean"],
            result["episode_reward_max"], result["episode_len_mean"], {out_file.split(os.sep)[-1]}
        ))

    state = agent.save(join("out", "models", "simple-ray"))
    agent.stop()

    ray.shutdown()