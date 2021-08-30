import gym
import numpy as np
import os
import random

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from ray.rllib.env import MultiAgentEnv
from seal.sumo.config import *
from seal.sumo.kernel.kernel import SumoKernel
from seal.sumo.timer import ActionTimer
from seal.sumo.utils.random_routes import generate_random_routes

DEFAULT_SEED = 0


class AbstractSumoEnv(ABC, MultiAgentEnv):

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rand_route_args = config.get("rand_route_args", {})
        self.use_dynamic_seed = config.get("use_dynamic_seed", True)
        self.ranked = config.get("ranked", DEFUALT_RANKED)
        self.env_seed = self.rand_route_args.get("seed", DEFAULT_SEED)
        # Ex: "foo/bar" => "foo"
        self.path = os.path.split(self.config["net-file"])[0]

        # Check if the user provided a route-file to be used for simulations and if the
        # user wants random routes to be generated for EACH trial (rand_routes_on_reset).
        # If user's want random routes generated (i.e., "route-files" is None in provided
        # config), then a private flag, `__first_rand_routes_flag`, is set to True. This
        # forces the `reset()` function to generate at least ONE random route file before
        # being updated to False. This will never change back to True (hence the privacy).
        if self.config.get("route-files", None) is None:
            self.config["route-files"] = os.path.join(self.path,
                                                      "traffic.rou.xml")
            self.rand_routes_on_reset = self.config.get("rand_routes_on_reset",
                                                        True)
            self.__first_rand_routes_flag = True
        else:
            self.rand_routes_on_reset = False
            self.__first_rand_routes_flag = False

        self.kernel = SumoKernel(self.config)
        self.action_timer = ActionTimer(len(self.kernel.tls_hub))
        self.num_of_lanes = self.kernel.get_num_of_lanes()
        # self.road_capacity = self.kernel.get_road_capacity()
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
        """Generate random routes based on the details in the configuration
           dict provided at initialization.
        """
        netfile = self.config["net-file"]
        self.rand_route_args["n_routefiles"] = 1  # NOTE: Issues if > 1.
        if self.use_dynamic_seed:
            self.rand_route_args["seed"] = self.env_seed
            self.env_seed += 1
        generate_random_routes(netfile=netfile, path=self.path,
                               number_of_lanes=self.num_of_lanes,
                               **self.rand_route_args)

    def close(self) -> None:
        self.kernel.close()

    def seed(self) -> None:
        random.seed(self.env_seed)
        np.random.seed(self.env_seed)

    ## ============================================================================== ##
    ## .....ABSTRACT METHODS THAT NEED TO BE IMPLEMENTED BY CHILDREN CLASSES..... ##
    ## ============================================================================== ##
    @abstractmethod
    def step(self, actions) -> Tuple[Any, Any, Any, Any]:
        raise NotImplementedError("Cannot be called from Abstract "
                                  "Class `AbstractSumoEnv`.")

    @abstractmethod
    def _do_action(self, actions: Any) -> Any:
        raise NotImplementedError("Cannot be called from Abstract "
                                  "Class `AbstractSumoEnv`.")

    @abstractmethod
    def _get_reward(self) -> float:
        raise NotImplementedError("Cannot be called from Abstract "
                                  "Class `AbstractSumoEnv`.")

    @abstractmethod
    def _observe(self) -> Any:
        raise NotImplementedError("Cannot be called from Abstract "
                                  "Class `AbstractSumoEnv`.")
