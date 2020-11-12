import gym

from abc import ABC, abstractmethod
from typing import Any

class SumoEnv(ABC, gym.Env):

    @abstractmethod
    def action_space(self):
        pass

    @abstractmethod
    def observation_space(self):
        pass

    @abstractmethod
    def step(self, actions):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def _do_action(self, actions: Any) -> Any:
        pass

    @abstractmethod
    def _get_observation(self):
        pass

    @abstractmethod
    def _get_reward(self):
        pass