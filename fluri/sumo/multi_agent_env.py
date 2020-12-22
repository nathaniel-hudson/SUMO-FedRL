import gym
import numpy as np
import os
import traci

from collections import OrderedDict
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from typing import Any, Dict, List, Tuple

from .const import *
from .kernel.kernel import SumoKernel
from .sumo_env import SumoEnv
from .utils.random_routes import generate_random_routes


class TLAgent(gym.Env):

    def __init__(self):
        pass


class MultiPolicySumoEnv(SumoEnv, MultiAgentEnv):

    def __init__(self, config):
        super().__init__(config)

    @property
    def action_space(self):
        first = self.kernel.tls_hub.index2id[0]
        return self.kernel.tls_hub[first].action_space
        
    @property
    def observation_space(self):
        first = self.kernel.tls_hub.index2id[0]
        return self.kernel.tls_hub[first].observation_space

    def action_spaces(self, tls_id):
        return self.kernel.tls_hub[tls_id].action_space

    def observation_spaces(self, tls_id):
        return self.kernel.tls_hub[tls_id].observation_space

    def step(self, action_dict: Dict[Any, int]) -> Tuple[Dict, Dict, Dict, Dict]:
        self._do_action(action_dict)
        self.kernel.step()

        obs = self._observe()
        reward = {
            tls.id: self._get_reward(obs[tls.id])
            for tls in self.kernel.tls_hub
        }
        done = {"__all__": self.kernel.done()}
        info = {}

        return obs, reward, done, info

    def _do_action(self, actions: Dict[Any, int]) -> List[int]:
        """Perform the provided action for each trafficlight.

        Parameters
        ----------
        actions : Dict[Any, int]
            The action that each trafficlight will take

        Returns
        -------
        List[int]
            Returns the action taken --- influenced by which moves are legal or not.
        """
        taken_action = actions.copy()
        for tls in self.kernel.tls_hub:
            action = actions[tls.id]
            can_change = self.action_timer.can_change(tls.index)
            if action == 1 and can_change:
                tls.next_phase()
                self.action_timer.restart(tls.index)
            else:
                self.action_timer.decr(tls.index)
                taken_action[tls.index] = 0
        return List[int]

    def _get_reward(self, obs: np.ndarray) -> float:
        """Negative reward function based on the number of halting vehicles, waiting time,
           and travel time.

        Parameters
        ----------
        obs : np.ndarray
            Numpy array (containing float64 values) representing the observation.

        Returns
        -------
        float
            The reward for this step
        """
        return -obs[NUM_HALT] - obs[WAIT_TIME] - obs[TRAVEL_TIME]

    def _observe(self) -> Dict[Any, np.ndarray]:
        """Get the observations across all the trafficlights, indexed by trafficlight id.

        Returns
        -------
        Dict[Any, np.ndarray]
            Observations from each trafficlight.
        """
        return {tls.id: tls.get_observation() for tls in self.kernel.tls_hub}