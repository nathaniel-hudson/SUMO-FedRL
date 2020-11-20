import gym
import numpy as np
import os

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from .const import *
from .kernel.kernel import SumoKernel
from .utils.random_routes import generate_random_routes

"""
TODO: The entire environments setup needs to be reformatted.
"""

class SumoEnv(ABC, gym.Env):

    def __init__(self, config: Dict[str, Any], scale_factor: float=0.5):
        assert 0.0 < scale_factor and scale_factor <= 1.0

        self.config = config

        self.path = os.path.split(self.config["net-file"])[0] # "foo/bar/car" => "foo/bar"
        self.config["route-files"] = os.path.join(self.path, "traffic.rou.xml")

        self.kernel = SumoKernel(self.config, scale_factor)
        self.scale_factor = scale_factor
        self.rand_routes_on_reset = self.config.get("rand_routes_on_reset", True)
        self.__first_round = True
        self.reset()

    def reset(self):
        # Start the simulation and get details surrounding the war.
        if self.rand_routes_on_reset or self.__first_round:
            self.rand_routes()
            self.__first_round = False
        self.start()
        self.action_timer = (-2 * MIN_DELAY) * np.ones(shape=(len(self.kernel.tls_hub)))
        return self.kernel.world.observe()

    def close(self) -> None:
        """Close the simulation, thereby ending the the connection to SUMO.
        """
        self.kernel.close()

    def start(self) -> None:
        """Start the simulation using the SumoKernel interface. This will reload the SUMO
           SUMO simulation if it's been loaded, otherwise it will start SUMO.
        """
        self.kernel.start()

    def rand_routes(self) -> None:
        net_name = self.config["net-file"]
        rand_args = self.config.get("rand_route_args", dict())
        rand_args["n_routefiles"] = 1 # NOTE: Simplifies the process.
        generate_random_routes(net_name=net_name, path=self.path, **rand_args)


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
    def _get_reward(self):
        pass
