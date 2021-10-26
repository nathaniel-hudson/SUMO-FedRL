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

    def step(self, action):
        pass

    def _do_action(self, action):
        pass

    def _get_reward(self, obs):
        pass

    def _observe(self):
        pass

    @property
    def action_space(self):
        return spaces.Dict({})

    @property
    def observation_space(self):
        return spaces.Dict({})
