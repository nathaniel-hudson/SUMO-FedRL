import gym
import numpy as np
import traci

from collections import OrderedDict, defaultdict
from gym import spaces
from typing import Any, Dict, List, Tuple

from .const import *
from .sumo_env import SumoEnv

GUI_DEFAULT = True

"""
TODO:
    + We need to adjust this w.r.t. the updated vision of the state-space.
    + This code will be broken with the changes introduced during implementation of the
      multi-agent environment using the new state space representation. We will handle
      it LATER.
"""

class SinglePolicySumoEnv(SumoEnv, gym.Env):
    """Custom Gym environment designed for simple RL experiments using SUMO/TraCI."""
    metadata = {"render.modes": ["sumo", "sumo-gui"]}
    name = "SinglePolicySumoEnv-v1"
    
    def __init__(
        self, 
        config: Dict[str, Any]
    ):
        super().__init__(config)

    @property
    def action_space(self) -> spaces.MultiDiscrete:
        """Initializes an instance of the action space as a property of the class.
           TODO: We need to reconsider random sampling for the action space. Maybe we can
                 write this more simply thatn we currently have it.

        Returns
        -------
        spaces.MultiDiscrete
            The action space.
        """
        # return spaces.Box(low=0, high=1, shape=(len(self.kernel.tls_hub),), dtype=int)
        return spaces.Dict({
            tls.id: spaces.Box(low=0, high=1, shape=(1,), dtype=int)
            for tls in self.kernel.tls_hub
        })

    @property
    def observation_space(self, dtype: np.dtype=np.float64) -> spaces.Box:
        """Initializes an instance of the observation space as a property of the class.

        Parameters
        ----------
        dtype : type, optional
            Some numpy data type you wish to use to represent the space values, by default 
            np.float64.

        Returns
        -------
        spaces.Box
            The observation space.
        """
        # Get the maximum value of the numpy data (int or float) type.
        try:
            high = np.iinfo(dtype).max
        except ValueError:
            high = np.finfo(dtype).max
        n = len(self.kernel.tls_hub)
        return spaces.Box(low=0, high=high, shape=(n, N_FEATURES), dtype=dtype)
        # return spaces.Dict({
        #     "num_vehicles":  spaces.Box(low=0, high=high, shape=(n,), dtype=dtype),
        #     "avg_speed":     spaces.Box(low=0, high=high, shape=(n,), dtype=dtype),
        #     "num_occupancy": spaces.Box(low=0, high=high, shape=(n,), dtype=dtype),
        #     "wait_time":     spaces.Box(low=0, high=high, shape=(n,), dtype=dtype),
        #     "travel_time":   spaces.Box(low=0, high=high, shape=(n,), dtype=dtype),
        #     "num_halt":      spaces.Box(low=0, high=high, shape=(n,), dtype=dtype),
        #     "curr_state":    spaces.Box(low=0, high=high, shape=(n,), dtype=dtype),
        # })

    def reset(self):
        # Start the simulation and get details surrounding the world.
        super().reset()
        return self._observe()

    def step(self, action: List[int]) -> Tuple[np.ndarray, float, bool, dict]:
        """Performs a single step in the environment, as per the Open AI Gym framework.

        Parameters
        ----------
        action : List[int]
            The action to be taken by each traffic light in the road network.

        Returns
        -------
        Tuple[np.ndarray, float, bool, dict]
            The current observation, reward, if the simulation is done, and other info.
        """
        taken_action = self._do_action(action)
        self.kernel.step()

        obs = self._observe()
        reward = self._get_reward(obs)
        done = self.kernel.done()
        info = {"taken_action": taken_action}

        # print(f"SinglePolicySumoEnv.step() -> obs:\n{observation}\n")

        return obs, reward, done, info

    def _do_action(self, actions: List[int]) -> List[int]:
        """This function takes a list of integer values. The integer values correspond
           with a traffic light state. The list provides the integer state values for each
           traffic light in the simulation.

        Parameters
        ----------
        action : List[int]
            Action to perform for each traffic light.

        Returns
        -------
        List[int]
            The action that is taken. If the passed in action is legal, then that will be
            returned. Otherwise, the returned action will be the prior action.
        """
        taken_action = actions.copy()
        for tls in self.kernel.tls_hub:
            # act = actions[tls.index]
            act = actions[tls.id]
            can_change = self.action_timer.can_change(tls.index)

            # If this condition is true, then the RYG state of the current traffic light
            # `tls` will be changed to the selected `next_state` provided by `actions`.
            # This only occurs if the next state and current state are not the same, the
            # transition is valid, and the `tls` is available to change. If so, then
            # the change is made and the timer is reset.
            if act == 1 and can_change:
                tls.next_phase()
                self.action_timer.restart(tls.index)
                taken_action[tls.index] = 1
            # Otherwise, keep the state the same, update the taken action, and then 
            # decrease the remaining time by 1.
            else:
                self.action_timer.decr(tls.index)
                taken_action[tls.index] = 0

        return taken_action

    def _get_reward(self, obs: np.ndarray) -> float:
        """For now, this is a simple function that returns -1 when the simulation is not
           done. Otherwise, the function returns 0. The goal's for the agent to prioritize
           ending the simulation quick.

        Returns
        -------
        float
            The reward for this step.
        """
        # num_halt = np.array(obs["num_halt"])
        # wait_time = np.array(obs["wait_time"])
        # travel_time = np.array(obs["travel_time"])
        # return sum(-1*num_halt) + sum(-1*wait_time) + sum(-1*travel_time)
        num_halt = obs[:, NUM_HALT]
        wait_time = obs[:, WAIT_TIME]
        travel_time = obs[:, TRAVEL_TIME]
        return sum(-1*num_halt) + sum(-1*wait_time) + sum(-1*travel_time)

    def _observe(self) -> np.ndarray:
        # obs = defaultdict(list)
        # for tls in self.kernel.tls_hub:
        #     tls_obs = tls.get_observation()
        #     for feature, value in tls_obs.items():
        #         obs[feature].append(value)
        # return dict(obs)
        obs = np.array([tls.get_observation() for tls in self.kernel.tls_hub])
        return obs