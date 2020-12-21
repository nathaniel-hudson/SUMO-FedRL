import gym
import numpy as np
import traci

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

class SingleSumoEnv(SumoEnv):
    """Custom Gym environment designed for simple RL experiments using SUMO/TraCI."""
    metadata = {"render.modes": ["sumo", "sumo-gui"]}
    name = "SingleSumoEnv-v1"
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        scale_factor: float=0.5
    ):
        super().__init__(config, scale_factor)

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
        return spaces.MultiDiscrete([len(tls.possible_states)
                                     for tls in self.kernel.tls_hub])

    @property
    def observation_space(self, kind: np.dtype=np.float16) -> spaces.Box:
        """Initializes an instance of the observation space as a property of the class.

        Returns
        -------
        spaces.Box
            The observation space.
        """
        # Get the maximum value of the numpy data (int or float) type.
        try:
            high = np.iinfo(kind).max
        except ValueError:
            high = np.finfo(kind).max
        n = len(self.kernel.tls_hub)
        return spaces.Dict({
            "num_vehicles":  spaces.Box(low=0, high=high, shape=(1 ,n), dtype=kind),
            "avg_speed":     spaces.Box(low=0, high=high, shape=(1, n), dtype=kind),
            "num_occupancy": spaces.Box(low=0, high=high, shape=(1, n), dtype=kind),
            "wait_time":     spaces.Box(low=0, high=high, shape=(1, n), dtype=kind),
            "travel_time":   spaces.Box(low=0, high=high, shape=(1, n), dtype=kind),
            "num_halt":      spaces.Box(low=0, high=high, shape=(1, n), dtype=kind),
            "curr_state":    spaces.Box(low=0, high=high, shape=(1, n), dtype=kind),
        })
        # world_space = spaces.Box(
        #     low=0,
        #     high=1,
        #     shape=self.kernel.world.get_dimensions(),
        #     dtype=np.float32
        # )
        # return world_space

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
        traci.simulationStep()
        self.kernel.update()

        observation = self.kernel.world.observe()
        reward = self._get_reward()
        done = self.kernel.done()
        info = {"taken_action": taken_action}

        return observation, reward, done, info

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
        can_change = (self.action_timer == 0)
        taken_action = actions.copy()

        for tls in self.kernel.tls_hub:
            curr_state = tls.state
            next_state = tls.get_state(actions[tls.index])
            is_valid = tls.valid_next_state(next_state)

            # If this condition is true, then the RYG state of the current traffic light
            # `tls` will be changed to the selected `next_state` provided by `actions`.
            # This only occurs if the next state and current state are not the same, the
            # transition is valid, and the `tls` is available to change. If so, then
            # the change is made and the timer is reset.
            if (curr_state != next_state) and is_valid and can_change[tls.index]:
                traci.trafficlight.setRedYellowGreenState(tls.id, next_state)
                self.action_timer[tls.index] = self._restart_timer()
            # Otherwise, keep the state the same, update the taken action, and then 
            # decrease the remaining time by +1.
            else:
                traci.trafficlight.setRedYellowGreenState(tls.id, curr_state)
                taken_action[tls.index] = tls.possible_states.index(curr_state)
                print(tls.index)
                self._decr_timer(tls.index)

        return taken_action

    def _get_reward(self) -> float:
        """For now, this is a simple function that returns -1 when the simulation is not
           done. Otherwise, the function returns 0. The goal's for the agent to prioritize
           ending the simulation quick.

        Returns
        -------
        float
            The reward for this step.
        """
        return -1.0 if not (self.kernel.done()) else 0.0
