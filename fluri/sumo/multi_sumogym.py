import gym
import numpy as np

from gym import spaces
from typing import Tuple

from .sumosim import SumoSim

class MultiSumoGym(gym.Env):

    def __init__(self, sim: SumoSim, grid_shape: Tuple[int, int]=None):
        self.sim = sim
        self.grid_shape = grid_shape
        self.reset()

    @property
    def action_space(self) -> spaces.MultiDiscrete:
        n_traffic_lights = 10
        return spaces.MultiDiscrete([5
                                     for i in range(n_traffic_lights)])

    @property
    def observation_Space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=10,
            shape=self.get_obs_dims(),
            dtype=np.int8
        )

    def reset(self):
        obs_n = list()
        reward_n = list()
        done_n = list()
        info_n = {"n": []}
        # ...
        return obs_n, reward_n, done_n, info_n