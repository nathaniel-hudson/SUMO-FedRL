import gym
import numpy as np

from gym import spaces
from typing import List, Tuple


class TestbedEnv(gym.Env):
    """
    This is a possible option for constructing the environment to work
    with the testbed. Though, it may be less straightforward and introduce
    more room for error. But, we have this skeleton setup for implementation
    if we choose to go this way.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

    def step(self, action: List[int]) -> Tuple:
        pass

    def _do_action(self, action: List[int]) -> List[int]:
        pass

    def _get_reward(self, obs: np.ndarray) -> float:
        pass

    def _observe(self) -> np.array:
        pass

    @property
    def action_space(self) -> spaces.Space:
        return spaces.Dict({})

    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Dict({})