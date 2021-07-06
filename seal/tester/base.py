from abc import ABC
from seal.sumo.env import SumoEnv
from pandas import DataFrame
from ray.rllib.agents import ppo
from typing import Dict

class BaseTester(ABC):

    def __init__(
        self,
        env_config,
        checkpoint,
        env_class=SumoEnv,
        config=None
    ) -> None:
        self.env_class = env_class
        self.config = self.get_config() if config is None else config
        self.env_config = env_config
        self.agent = ppo.PPOTrainer(config=self.config, env=self.env_class)
        self.agent.restore(checkpoint)


    def test(self) -> DataFrame:
        env = self.env_class(self.env_config)
        episode_reward = 0
        done = False
        obs = env.reset()
        while not done:
            action = self.agent.compute_actions(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        return episode_reward

    def load_model(checkpoint: str=None) -> None:
        pass

    def get_config(self) -> Dict:
        