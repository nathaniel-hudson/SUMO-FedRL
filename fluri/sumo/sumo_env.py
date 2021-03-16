import gym
import numpy as np
import os

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from fluri.sumo.const import *
from fluri.sumo.kernel.kernel import SumoKernel
from fluri.sumo.timer import ActionTimer
from fluri.sumo.utils.random_routes import generate_random_routes


class SumoEnv(ABC):

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # "foo/bar/car" => "foo/bar"
        self.path = os.path.split(self.config["net-file"])[0]

        # Check if the user provided a route-file to be used for simulations and if the
        # user wants random routes to be generated for EACH trial (rand_routes_on_reset).
        # If user's want random routes generated (i.e., "route-files" is None in provided
        # config), then a private flag, `__first_rand_routes_flag`, is set to True. This
        # forces the `reset()` function to generate at least ONE random route file before
        # being updated to False. This will never change back to True (hence the privacy).
        if self.config.get("route-files", None) is None:
            self.config["route-files"] = os.path.join(
                self.path, "traffic.rou.xml")
            self.rand_routes_on_reset = self.config.get(
                "rand_routes_on_reset", True)
            self.__first_rand_routes_flag = True
        else:
            self.rand_routes_on_reset = False

        self.kernel = SumoKernel(self.config)
        self.action_timer = ActionTimer(len(self.kernel.tls_hub))
        self.reset()

    def reset(self) -> Any:
        """Start the simulation and get details surrounding the world.

        Returns
        -------
        Any
            The observation of the state space upon resetting the simulation/environment.
        """
        if self.rand_routes_on_reset or self.__first_rand_routes_flag:
            self.rand_routes()
            self.__first_rand_routes_flag = False
        self.kernel.start()
        self.action_timer.restart()
        return self._observe()

    def rand_routes(self) -> None:
        """Generate random routes based on the details in the configuration dict provided
           at initialization.
        """
        net_name = self.config["net-file"]
        rand_args = self.config.get("rand_route_args", dict())
        # NOTE: Simplifies process, so leave this for now.
        rand_args["n_routefiles"] = 1
        generate_random_routes(net_name=net_name, path=self.path, **rand_args)
        # TODO: Implement dynamic seed.

    def close(self) -> None:
        self.kernel.close()

    ## ================================================================================ ##
    ## .........ABSTRACT METHODS THAT NEED TO BE IMPLEMENTED BY CHILDREN CLASSES....... ##
    ## -------------------------------------------------------------------------------- ##
    @abstractmethod
    def step(self, actions) -> Tuple[Any, Any, Any, Any]:
        pass

    @abstractmethod
    def _do_action(self, actions: Any) -> Any:
        pass

    @abstractmethod
    def _get_reward(self) -> float:
        pass

    @abstractmethod
    def _observe(self) -> Any:
        pass
