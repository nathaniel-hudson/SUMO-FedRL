import gym
import numpy as np
import os

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from .const import *
from .kernel.kernel import SumoKernel
from .timer import ActionTimer
from .utils.random_routes import generate_random_routes

"""
TODO: The entire environments setup needs to be reformatted.
"""

class SumoEnv(ABC):

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.path = os.path.split(self.config["net-file"])[0] # "foo/bar/car" => "foo/bar"
        self.config["route-files"] = os.path.join(self.path, "traffic.rou.xml")

        self.kernel = SumoKernel(self.config)
        self.action_timer = ActionTimer(len(self.kernel.tls_hub))
        self.rand_routes_on_reset = self.config.get("rand_routes_on_reset", True)
        self.reset()

    def reset(self):
        # Start the simulation and get details surrounding the world.
        if self.rand_routes_on_reset:
            self.rand_routes()
        self.kernel.start()

    def rand_routes(self) -> None:
        net_name = self.config["net-file"]
        rand_args = self.config.get("rand_route_args", dict())
        rand_args["n_routefiles"] = 1 # NOTE: Simplifies the process.
        generate_random_routes(net_name=net_name, path=self.path, **rand_args)


    ## ================================================================= ##
    ## ABSTRACT METHODS THAT NEED TO BE IMPLEMENTED BY CHILDREN CLASSES. ##
    ## ----------------------------------------------------------------- ##
    # @abstractmethod
    # def action_space(self):
    #     pass

    # @abstractmethod
    # def observation_space(self):
    #     pass

    @abstractmethod
    def step(self, actions):
        pass

    @abstractmethod
    def _do_action(self, actions: Any) -> Any:
        pass

    @abstractmethod
    def _get_reward(self):
        pass
