import gym
import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from .const import *
from .kernel.kernel import SumoKernel

"""
TODO: The entire environments setup needs to be reformatted.
"""

class SumoEnv(ABC, gym.Env):

    def __init__(self, config: Dict[str, Any], scale_factor: float=0.5):
        assert 0.0 < scale_factor and scale_factor <= 1.0

        pass
        self.config = config
        self.kernel = SumoKernel(self.config, scale_factor)
        self.scale_factor = scale_factor
        self.reset()

    def reset(self):
        # Start the simulation and get details surrounding the war.
        self.start()
        self.action_timer = (-2 * MIN_DELAY) * np.ones(shape=(len(self.kernel.tls_hub)))
        return self.kernel.world.observe()

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
        self.kernel.close()

    def start(self) -> None:
        """Start the simulation using the SumoKernel interface. This will reload the SUMO
           SUMO simulation if it's been loaded, otherwise it will start SUMO.
        """
        self.kernel.start()

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
