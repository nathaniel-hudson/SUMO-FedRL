import gym
import numpy as np
import os
import traci

from collections import OrderedDict
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from typing import Dict, Tuple

from .const import *
from .kernel.kernel import SumoKernel
from .sumo_env import SumoEnv
from .utils.random_routes import generate_random_routes

class MultiPolicyEnv(MultiAgentEnv):

    def __init__(self, config):
        self.config = config
        self.path = os.path.split(self.config["net-file"])[0] # "foo/bar/car" => "foo/bar"
        self.config["route-files"] = os.path.join(self.path, "traffic.rou.xml")

        self.kernel = SumoKernel(self.config)
        self.rand_routes_on_reset = self.config.get("rand_routes_on_reset", True)
        self.reset()

    # @property
    def action_space(self) -> spaces.Dict:
        """Property for the action space for this environment. The space is a spaces.Dict
           object where each item's key is a trafficlight ID used in SUMO and the value is
           a spaces.Box object for binary values.

        Returns
        -------
        spaces.Dict
            Action space.
        """
        return spaces.Dict({
            tls.id: spaces.Box(low=0, high=1, shape=(1,), dtype=int)
            for tls in self.kernel.tls_hub
        })

    # @property
    def observation_space(self, kind: np.dtype=np.float16) -> spaces.Dict:
        """Property for the observation space for this environemnt. The space is a 
           spaces.Dict object where each item's key is a trafficlight ID used in SUMO and 
           the value is another spaces.Dict object containing the 7 features of interest.

        Returns
        -------
        spaces.Dict
            Observation space.
        """
        # Get the maximum value of the numpy data type.
        try: high = np.iinfo(kind).max               # Handles numpy integer data types.
        except ValueError: high = np.finfo(kind).max # Handles numpy float data types.
        return spaces.Dict({
            tls.id: spaces.Dict({
                "num_vehicles":  spaces.Box(low=0, high=high, shape=(1,), dtype=kind),
                "avg_speed":     spaces.Box(low=0, high=high, shape=(1,), dtype=kind),
                "num_occupancy": spaces.Box(low=0, high=high, shape=(1,), dtype=kind),
                "wait_time":     spaces.Box(low=0, high=high, shape=(1,), dtype=kind),
                "travel_time":   spaces.Box(low=0, high=high, shape=(1,), dtype=kind),
                "num_halt":      spaces.Box(low=0, high=high, shape=(1,), dtype=kind),
                "curr_state":    spaces.Box(low=0, high=high, shape=(1,), dtype=kind),
            })
            for tls in self.kernel.tls_hub
        })

    def reset(self):
        # Start the simulation and get details surrounding the world.
        if self.rand_routes_on_reset:
            self.rand_routes()
        self.start()

        self.step_counter = 0
        self._restart_timer()
        return OrderedDict({
            tls.id: tls.get_observation()
            for tls in self.kernel.tls_hub
        })

    def step(self, action_dict: OrderedDict) -> Tuple[Dict, Dict, Dict, Dict]:
        self._do_action(action_dict)
        self.kernel.step()

        obs = {
            tls.id: tls.get_observation()
            for tls in self.kernel.tls_hub
        }
        reward = {
            tls.id: self._get_reward(obs[tls.id])
            for tls in self.kernel.tls_hub
        }
        done = self.kernel.done()
        info = {}

        return obs, reward, done, info

    def _do_action(self, action_dict):
        can_change = (self.action_timer == 0)
        did_change = None
        for tls in self.kernel.tls_hub:
            if action_dict[tls.id][0] and can_change[tls.index]:
                tls.next_phase()
            else:
                self._decr_timer(tls.index)

    def _get_reward(self, obs: OrderedDict) -> float:
        return -obs["num_halt"] - obs["wait_time"] - obs["travel_time"]


    ## ================================================================================ ##


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


    def _restart_timer(self, index=None) -> np.ndarray:
        if index is None:
            self.action_timer = MIN_DELAY * np.ones(shape=(len(self.kernel.tls_hub)))
        else:
            self.action_timer[index] = MIN_DELAY

    def _decr_timer(self, index) -> None:
        self.action_timer[index] = max(0, self.action_timer[index]-1)