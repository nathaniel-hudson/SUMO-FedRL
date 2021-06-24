import gym
from numpy.lib.function_base import append
import ray

from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from fluri.sumo.utils.random_routes import generate_random_routes
from fluri.sumo.single_agent_env import SinglePolicySumoEnv
from fluri.sumo.kernel.kernel import SumoKernel
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from os.path import join

def train(config, total_timesteps: int=int(2e6)):
    env = SinglePolicySumoEnv(config=config)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save("two_inter_model")
    env.close()

if __name__ == "__main__":
    path = join("configs", "two_inter")
    config = {
        "gui": False,
        "net-file": join(path, "two_inter.net.xml"),
        "tripinfo-output": join(path, "tripinfo.xml")
    }
    train(config, int(2e5))