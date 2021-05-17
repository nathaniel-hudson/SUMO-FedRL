import gym
import numpy as np

from gym import spaces
from typing import Any, Dict, List, Tuple

from .const import *
from .sumo_env import SumoEnv

GUI_DEFAULT = True


class SinglePolicySumoEnv(SumoEnv, gym.Env):
    """Custom Gym environment designed for simple RL experiments using SUMO/TraCI."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    @property
    def action_space(self) -> spaces.MultiDiscrete:
        """Initializes an instance of the action space as a property of the class.

        Returns
        -------
        spaces.MultiDiscrete
            The action space.
        """
        return spaces.Dict({
            tls.id: spaces.Box(low=0, high=1, shape=(1,), dtype=int)
            for tls in self.kernel.tls_hub
        })

    @property
    def observation_space(self) -> spaces.Box:
        """Initializes an instance of the observation space as a property of the class.

        Returns
        -------
        spaces.Box
            The observation space.
        """
        dtype = np.float64
        high = np.finfo(dtype).max
        n_tls = len(self.kernel.tls_hub)
        n_features = N_RANKED_FEATURES if self.ranked else N_UNRANKED_FEATURES
        return spaces.Box(low=0, high=high, shape=(n_tls, n_features), dtype=dtype)


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
        """Negative reward function based on the number of halting vehicles, waiting time,
           and travel time. These values are summed across all trafficlights.

        Parameters
        ----------
        obs : np.ndarray
            Numpy array (containing float64 values) representing the observations across
            trafficlights.

        Returns
        -------
        float
            The reward for this step.
        """
        # TODO: This needs to be adjusted to be more fair in comparison to MARL approaches.
        # num_veh = obs[:, NUM_VEHICLES]
        # num_halt = obs[:, NUM_HALT]
        # wait_time = obs[:, WAIT_TIME]
        # travel_time = obs[:, TRAVEL_TIME]
        # return sum(-1*num_veh) + sum(-1*num_halt) + sum(-1*wait_time) + sum(-1*travel_time)

        # NOTE: This should address the mentioned problem above.
        return -1 * np.mean(obs[:, NUM_HALT])

    def _observe(self, ranked: bool=False) -> np.ndarray:
        """Get a the observations across all the trafficlights (represented by a single
           numpy array).

        Returns
        -------
        np.ndarray
            Trafficlight observations.
        """
        obs = np.array([tls.get_observation(self.ranked) for tls in self.kernel.tls_hub])
        if ranked:
            pass
        return obs
