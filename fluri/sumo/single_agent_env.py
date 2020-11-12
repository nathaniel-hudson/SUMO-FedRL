import gym
import numpy as np
import traci

from gym import spaces
from typing import Any, Dict, List, Tuple

from . import sumo_util as utils
from .const import *
from .sumo_env import SumoEnv
from .sumo_sim import SumoSim
from .trafficlights import TrafficLights

GUI_DEFAULT = True

class SingleSumoEnv(SumoEnv):
    """Custom Gym environment designed for simple RL experiments using SUMO/TraCI."""
    metadata = {"render.modes": ["sumo", "sumo-gui"]}
    name = "SingleSumoEnv-v1"
    WORLD_KEY = "world"
    TLS_KEY  = "traffic_lights"
    
    def __init__(self, sim: SumoSim, world_shape: Tuple[int, int]=None):
        super().__init__(sim, world_shape)

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
        return spaces.MultiDiscrete([len(self.trafficlights.states[tls_id])
                                     for tls_id in self.trafficlights.states])

    @property
    def observation_space(self) -> spaces.Box:
        """Initializes an instance of the observation space as a property of the class.

        Returns
        -------
        spaces.Box
            The observation space.
        """
        world_space = spaces.Box(
            low=0,
            high=10,
            shape=self.get_obs_dims(),
            dtype=np.int8
        )
        return world_space

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
        self.world = self._get_world()

        observation = self.world
        reward = self._get_reward()
        done = self.sim.done()
        info = {"taken_action": taken_action}

        return observation, reward, done, info

    def is_valid_action(self, tls_id: str, curr_action: str, next_action: str) -> bool:
        """Determines if `next_action` is valid given the current action (`curr_action`).

        Parameters
        ----------
        tls_id : str
            The traffic light ID.
        curr_action : str
            The state of the current action.
        next_action : str
            The state of the next action.

        Returns
        -------
        bool
            True if `next_action` is valid, False otherwise.
        """
        curr_node = utils.get_node_id(tls_id, curr_action)
        next_node = utils.get_node_id(tls_id, next_action)
        is_valid = next_node in self.trafficlights.network.neighbors(curr_node)
        return is_valid

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
        can_change = self.mask == 0
        taken_action = actions.copy()

        for tls_id, curr_action in self.trafficlights.curr_states.items():
            next_action = self._interpret_action(tls_id, actions)
            is_valid = self.is_valid_action(tls_id, curr_action, next_action)

            if curr_action != next_action and is_valid and can_change[int(tls_id)]:
                traci.trafficlight.setRedYellowGreenState(tls_id, next_action)
                self.mask[int(tls_id)] = -2 * MIN_DELAY

            else:
                traci.trafficlight.setRedYellowGreenState(tls_id, curr_action)
                self.mask[int(tls_id)] = min(0, self.mask[int(tls_id)] + 1)
                taken_action[int(tls_id)] = self.trafficlights.states[tls_id].\
                                            index(curr_action)

        self.trafficlights.update_curr_states()
        return taken_action

    def _get_world(self) -> np.ndarray:
        """Returns the current observation of the state space, represented by the world
           space for recognizing vehicle locations and the current state of all traffic
           lights.

        Returns
        -------
        Dict[np.ndarray, np.ndarray]
            Get the current observation of the environment.
        """
        world = np.zeros(shape=(self.obs_h, self.obs_w), dtype=np.int32)
        veh_ids = list(traci.vehicle.getIDList())
        for veh_id in veh_ids:
            # Get the (scaled-down) x- or y-coordinates for the observation world.
            x, y = traci.vehicle.getPosition(veh_id)
            x = int(x * self.w_scalar)
            y = int(y * self.h_scalar)

            # Add a normalized weight to the respective coordinate in the world. For it to
            # be normalized, we need to change `dtype` to a float-based value.
            world[y, x] += 1 #/ len(veh_ids)

        return world

    def _interpret_action(self, tls_id: str, action: List[int]) -> str:
        """Actions  are passed in as a numpy  array of integers. However, this needs to be
           interpreted as an action state (e.g., `GGrr`) based on the TLS possible states.
           So,  given an ID  tls=2 and  an action  a=[[1], [3], [0], ..., [2]] (where each 
           integer  value  corresponds with  the index  of the  states for a given traffic 
           light), return the state corresponding to the index provided by action.

        Parameters
        ----------
        tls_id : str
            ID of the designated traffic light.
        action : List[int]
            Action vector where each element selects the action for the respective traffic
            light.

        Returns
        -------
        str
            The string state that corresponds with the selected action for the given
            traffic light.
        """
        return self.trafficlights.states[tls_id][action[int(tls_id)]]

    def _get_reward(self) -> float:
        """For now, this is a simple function that returns -1 when the simulation is not
           done. Otherwise, the function returns 0. The goal's for the agent to prioritize
           ending the simulation quick.

        Returns
        -------
        float
            The reward for this step.
        """
        return -1.0 if not (self.sim.done()) else 0.0
