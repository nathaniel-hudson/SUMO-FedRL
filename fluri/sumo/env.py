import numpy as np

from fluri.sumo.config import *
from fluri.sumo.abstract_env import AbstractSumoEnv
from gym import spaces
from ray.rllib.env import MultiAgentEnv
from typing import Any, Dict, List, Tuple


class SumoEnv(AbstractSumoEnv, MultiAgentEnv):

    def __init__(self, config):
        super().__init__(config)

    @property
    def multi_action_space(self) -> spaces.Space:
        return spaces.Dict({
            idx: self.kernel.tls_hub[idx].action_space
            for _, idx in self.kernel.tls_hub.index2id.items()
        })

    @property
    def action_space(self) -> spaces.Space:
        """This is the action space defined for a *single* traffic light. It is
           defined this way to support RlLib more easily.

        Returns:
            Space: Action space for a single traffic light.
        """
        first = self.kernel.tls_hub.index2id[0]
        return self.kernel.tls_hub[first].action_space

    @property
    def observation_space(self) -> spaces.Space:
        """This is the observation space defined for a *single* traffic light. It is
           defined this way to support RlLib more easily.

        Returns:
            Space: Observation space for a single traffic light.
        """
        first = self.kernel.tls_hub.index2id[0]
        return self.kernel.tls_hub[first].observation_space

    def action_spaces(self, tls_id) -> spaces.Space:
        return self.kernel.tls_hub[tls_id].action_space

    def observation_spaces(self, tls_id) -> spaces.Space:
        return self.kernel.tls_hub[tls_id].observation_space

    def step(self, action_dict: Dict[Any, int]) -> Tuple[Dict, Dict, Dict, Dict]:
        taken_action = self._do_action(action_dict)
        self.kernel.step()

        obs = self._observe()
        reward = {
            tls.id: self._get_reward(obs[tls.id])
            for tls in self.kernel.tls_hub
        }
        done = {"__all__": self.kernel.done()}
        # info = {"taken_action": taken_action,
        #         "total_reward": sum(reward.values())}
        info = {}

        return obs, reward, done, info

    def _do_action(self, actions: Dict[Any, int]) -> List[int]:
        """Perform the provided action for each trafficlight.

        Args:
            actions (Dict[Any, int]): The action that each trafficlight will take

        Returns:
            Dict[Any, int]: Returns the action taken -- influenced by which moves are
                legal or not.
        """
        taken_action = actions.copy()
        for tls in self.kernel.tls_hub:
            if actions[tls.id] == 1 and self.action_timer.can_change(tls.index):
                tls.next_phase()
                self.action_timer.restart(tls.index)
            else:
                self.action_timer.decr(tls.index)
                taken_action[tls.index] = 0
        return taken_action

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
        return -(obs[LANE_OCCUPANCY] + obs[HALTED_LANE_OCCUPANCY])

    def _observe(self) -> Dict[Any, np.ndarray]:
        """Get the observations across all the trafficlights, indexed by trafficlight id.

        Returns
        -------
        Dict[Any, np.ndarray]
            Observations from each trafficlight.
        """
        obs = {tls.id: tls.get_observation() for tls in self.kernel.tls_hub}
        if self.ranked:
            self._get_ranks(obs)
        return obs

    def _get_ranks(self, obs: Dict) -> None:
        """Appends global and local ranks to the observations in an inplace fashion.

        Args:
            obs (Dict): Observation provided by a trafficlight.
        """
        pairs = [(tls_id, tls_state[LANE_OCCUPANCY])
                 for tls_id, tls_state in obs.items()]
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        graph = self.kernel.tls_hub.tls_graph  # Adjacency list representation.

        # Calculate the GLOBAL ranks for each tls in the road network.
        for global_rank, (tls_id, _) in enumerate(pairs):
            try:
                obs[tls_id][GLOBAL_RANK] = 1 - (global_rank / (len(graph)-1))
            except ZeroDivisionError:
                obs[tls_id][GLOBAL_RANK] = 1

        # Calculate LOCAL ranks based on global ranks from above.
        for tls_id in graph:
            local_rank = 0
            for neighbor in graph[tls_id]:
                if obs[tls_id][GLOBAL_RANK] > obs[neighbor][GLOBAL_RANK]:
                    local_rank += 1
            try:
                obs[tls_id][LOCAL_RANK] = 1 - \
                    (local_rank / len(graph[tls_id]))
                # ^^ We do *not* subtract the denominator by 1 (as we do with global
                #    rank) because `len(graph[tls_id])` does not include `tls_id` as a
                #    node in the sub-network when it should be included. This means that
                #    +1 node cancels out the -1 node.
            except ZeroDivisionError:
                obs[tls_id][LOCAL_RANK] = 1
