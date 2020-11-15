import gym
import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Tuple

from .const import *
from .sumo_sim import SumoSim
from .trafficlights import TrafficLights

class SumoEnv(ABC, gym.Env):

    def __init__(self, sim: SumoSim, world_dim: Tuple[int, int]=None):
        self.sim = sim
        self.world_dim = world_dim
        self.reset()

    def reset(self):
        # Start the simulation and get details surrounding the war.
        self.start()
        self.trafficlights = TrafficLights(self.sim)
        self.mask = (-2 * MIN_DELAY) * np.ones(shape=(self.trafficlights.num))
        self.bounding_box = self.sim.get_bounding_box()

        # These values are with regard to the height and width of the world itself.
        # These values are used to get matrix representations of the world for learning.
        self.sim_h, self.sim_w = self.get_sim_dims()
        self.obs_h, self.obs_w = self.get_obs_dims()
        self.h_scalar = self.obs_h / self.sim_h
        self.w_scalar = self.obs_w / self.sim_w

        self.world = self._get_world()
        return self.world

    def get_sim_dims(self) -> Tuple[int, int]:
        """Provides the original (height, width) dimensions for the simulation for this
           Gym environment.

        Returns
        -------
        Tuple[int, int]
            (width, height) of SingleSumoEnv instance.
        """
        x_min, y_min, x_max, y_max = self.bounding_box
        width = int(x_max - x_min)
        height = int(y_max - y_min)
        return (height, width)

    def get_obs_dims(self) -> Tuple[int, int]:
        """Gets the dimensions of the world observation space. If the `world_dim` param
           is set to None, then the original bounding box's dimensions (provided by TraCI)
           will be used. This, however, is non-optimal and it is recommended that you
           provide a smaller dimensionality to represent the `world_dim` for better
           learning.

        Returns
        -------
        Tuple[int, int]
            (height, width) pair of the observation space.
        """
        if self.world_dim is None:
            return self.get_sim_dims()
        else:
            return self.world_dim

    def close(self) -> None:
        """Close the simulation, thereby ending the the connection to SUMO.
        """
        self.sim.close()

    def start(self) -> None:
        """Start the simulation using the SumoSim interface. This will reload the SUMO
           SUMO simulation if it's been loaded, otherwise it will start SUMO.
        """
        self.sim.start()

    ## ================================================================= ##
    ## ABSTRACT METHODS THAT NEED TO BE IMPLEMENTED BY CHILDREN CLASSES. ##
    ## ----------------------------------------------------------------- ##
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
    def _do_action(self, actions: Any) -> Any:
        pass

    @abstractmethod
    def _get_world(self) -> np.ndarray:
        pass

    @abstractmethod
    def _get_reward(self):
        pass
