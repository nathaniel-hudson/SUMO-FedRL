import gym
import numpy as np

from gym import spaces
from typing import Tuple

from .sumo_env import SumoEnv
from .sumo_sim import SumoSim

class MultiSumoEnv(SumoEnv):

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
        obs_n = dict()
        reward_n = dict()
        done_n = dict()
        info_n = {"n": []}
        # ...
        return obs_n, reward_n, done_n, info_n

    def step(self, actions: dict):
        # Perform given actions for each agent and then take ONE simulation step in SUMO.
        for agent in self.agent:
            self._do_action(agent, actions[agent])
        self.sim.step()
        self.__update_world()

    def _do_action(self, agent_id, action):
        pass

    def _get_observation(self, agent_id) -> np.ndarray:
        pass

    def _get_reward(self, agent_id) -> float:
        pass

    def __update_world(self) -> None:
        """To (efficiently) get an accurate view of each agents' observation space, this 
           function simply updates the cached view of the entire world's state. This is
           then used to grab the sub-matrices of the world to represent each agents'
           view or observation subspace.
        """   
        pass