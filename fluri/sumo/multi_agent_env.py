import gym
import numpy as np
import os
import traci

from collections import OrderedDict
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from typing import Dict, List, Tuple

from .const import *
from .kernel.kernel import SumoKernel
from .sumo_env import SumoEnv
from .utils.random_routes import generate_random_routes


class TLAgent(gym.Env):

    def __init__(self):
        pass


class MultiPolicySumoEnv(SumoEnv, MultiAgentEnv):

    def __init__(self, config):
        self.config = config
        self.path = os.path.split(self.config["net-file"])[0] # "foo/bar/car" => "foo/bar"
        self.config["route-files"] = os.path.join(self.path, "traffic.rou.xml")

        self.kernel = SumoKernel(self.config)
        self.rand_routes_on_reset = self.config.get("rand_routes_on_reset", True)

        # self.action_spaces = [spaces.Box(low=0, high=high, shape=(1,), dtype=kind)
        #                       for tls in self.kernel.tls_hub]
        self.observation_spaces = None
        
        self.action_space = self.action_spaces[0]
        self.observation_space = [self.observation_spaces[0]]

        assert all(act_space == self.action_space
                   for act_space in self.action_spaces.values()), \
            "Action spaces for all agents must be identical."

        assert all(obs_space == self.observation_space
                   for obs_space in self.env.observation_spaces.values()), \
            "Observation spaces for all agents must be identical."

        self.reset()

    @property
    def action_space(self) -> spaces.Dict:
        """Property for the action space for this environment. The space is a spaces.Dict
           object where each item's key is a trafficlight ID used in SUMO and the value is
           a spaces.Box object for binary values.

        Returns
        -------
        spaces.Dict
            Action space.
        """
        return spaces.MultiBinary(len(self.kernel.tls_hub))

    @property
    def observation_space(self, kind: np.dtype=np.float64) -> spaces.Dict:
        """Property for the observation space for this environemnt. The space is a 
           spaces.Dict object where each item's key is a trafficlight ID used in SUMO and 
           the value is another spaces.Dict object containing the 7 features of interest.

        Returns
        -------
        spaces.Dict
            Observation space.
        """
        # Get the maximum value of the numpy data type.
        try: 
            high = np.iinfo(kind).max # Handles numpy integer data types.
        except ValueError: 
            high = np.finfo(kind).max # Handles numpy float data types.
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
        super().reset()
        return self._observe()

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

    def _do_action(self, actions: List[int]) -> List[int]:
        taken_action = actions.copy()
        for tls in self.kernel.tls_hub:
            action = actions[tls.index]
            can_change = self.action_timer.can_change(tls.index)
            if action == 1 and can_change:
                tls.next_phase()
                self.action_timer.restart(tls.index)
            else:
                self.action_timer.decr(tls.index)
                taken_action[tls.index] = 0
        return List[int]

    def _get_reward(self, obs: OrderedDict) -> float:
        return -obs["num_halt"] - obs["wait_time"] - obs["travel_time"]

    def _observe(self) -> OrderedDict:
        return OrderedDict({
            tls.id: tls.get_observation()
            for tls in self.kernel.tls_hub
        })


    ## ================================================================================ ##


    def rand_routes(self) -> None:
        net_name = self.config["net-file"]
        rand_args = self.config.get("rand_route_args", dict())
        rand_args["n_routefiles"] = 1 # NOTE: Simplifies the process.
        generate_random_routes(net_name=net_name, path=self.path, **rand_args)