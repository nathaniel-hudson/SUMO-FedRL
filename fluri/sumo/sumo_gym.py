import gym
import numpy as np
import random
import time
import traci

from collections import defaultdict
from gym import spaces

from . import sumo_util as utils
from .const import *
from .sumo_sim import SUMOSim

from typing import Any, Dict, List, Tuple

GUI_DEFAULT = True

"""
TODO:
[_] Setup the boundary bounds for limiting the view an agent can have.
"""

class TrafficLights:

    def __init__(self, sim: SUMOSim):
        self.sim = sim
        self.ids = self.sim.get_traffic_light_ids()
        self.states = self.sim.get_all_possible_tls_states()
        self.network = utils.make_tls_state_network(self.states)
        self.num = len(self.ids)
        self.curr_states = self.random_states()

        # TODO: Currenlty not being considered.
        self.radii = None

    def random_states(self):
        states = {}
        for tls_id in self.ids:
            states[tls_id] = random.choice([
                self.network.nodes[u]["state"] for u in self.network.neighbors(tls_id)
            ])
        return states

    def update_curr_states(self):
        self.curr_states = self.sim.get_all_curr_tls_states()


class SUMOGym(gym.Env):
    """Custom Gym environment designed for simple RL experiments using SUMO/TraCI."""
    metadata = {"render.modes": ["sumo", "sumo-gui"]}
    name = "SUMO-v1"
    GRID_KEY = "grid"
    TLS_KEY  = "traffic_lights"
    
    def __init__(self, sim: SUMOSim):
        self.sim = sim
        self.reset()

    def __interpret_action(self, tls_id: str, action: List[int]) -> str:
        """Actions  are passed in as a numpy  array of integers. However, this needs to be
           interpreted as an action state (e.g., `GGrr`) based on the TLS possible states.
           So,  given an ID  tls=2 and  an action  a=[[1], [3], [0], ..., [2]] (where each 
           integer  value  corresponds with  the index  of the  states for a given traffic 
           light), return the state corresponding to the index provided by action."""
        return self.trafficlights.states[tls_id][action[int(tls_id)]]


    def __do_action(self, action: List[int]) -> List[int]:
        """TODO"""
        can_change = self.mask == 0
        taken_action = action.copy()

        for tls_id, curr_action in self.trafficlights.curr_states.items():
            next_action = self.__interpret_action(tls_id, action)

            curr_node = utils.get_node_id(tls_id, curr_action)
            next_node = utils.get_node_id(tls_id, next_action)
            is_valid = next_node in self.trafficlights.network.neighbors(curr_node)

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


    def __get_observation(self) -> Dict[np.ndarray, np.ndarray]:
        """Returns the current observation of the state space, represented by the grid
           space for recognizing vehicle locations and the current state of all traffic
           lights.
        """
        x_min, y_min, x_max, y_max = self.bounding_box
        width = int(x_max - x_min)
        height = int(y_max - y_min)
        obs = {
            self.GRID_KEY: np.zeros(shape=(width, height), dtype=np.int32),
            self.TLS_KEY:  np.zeros(shape=(self.trafficlights.num), dtype=np.int32)
        }

        for veh_id in list(traci.vehicle.getIDList()):
            x, y = traci.vehicle.getPosition(veh_id)
            obs[self.GRID_KEY][int(x), int(y)] = 1

        for tls_id, curr_state in self.trafficlights.curr_states.items():
            index = self.trafficlights.states[tls_id].index(curr_state)
            obs[self.TLS_KEY][int(tls_id)] = index

        return obs[self.GRID_KEY]
        # return obs


    def __get_reward(self) -> float:
        """TODO"""
        return -1.0


    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        """Performs a single step in the environment, as per the Open AI Gym framework."""
        taken_action = self.__do_action(action)
        traci.simulationStep()

        observation = self.__get_observation()
        reward = self.__get_reward()
        done = self.sim.done()
        info = {"taken_action": taken_action}

        return observation, reward, done, info


    def reset(self) -> Dict[str, Any]:
        """TODO"""
        self.start()
        self.trafficlights = TrafficLights(self.sim)
        self.mask = (-2 * MIN_DELAY) * np.ones(shape=(self.trafficlights.num))
        self.bounding_box = self.sim.get_bounding_box()
        return self.__get_observation()

    def close(self) -> None:
        self.sim.close()

    def start(self) -> None:
        self.sim.start()

    @property
    def action_space(self):
        """Initializes an instance of the action space as a property of the class."""
        ## TODO: We need to adjust the `sample()` function for this action_space such that
        ##       it restricts available actions based on the current action.
        return spaces.MultiDiscrete([
            len(self.trafficlights.states[tls_id]) for tls_id in self.trafficlights.states
        ])


    @property
    def observation_space(self):
        """Initializes an instance of the observation space as a property of the class."""
        x_min, y_min, x_max, y_max = self.bounding_box
        width = int(x_max - x_min)
        height = int(y_max - y_min)

        grid_shape = (width, height)
        grid_space = spaces.Box(
            low=0, 
            high=1, 
            shape=grid_shape,
            dtype=np.int8
        )
        tls_space = spaces.MultiDiscrete([
            len(self.trafficlights.states[tls_id]) for tls_id in self.trafficlights.states
        ])

        return grid_space

        return spaces.Dict({
            self.GRID_KEY: grid_space, 
            self.TLS_KEY:  tls_space
        })


if __name__ == "__main__":
    """
    This simple running example demonstrates how to setup a configuration to run a full
    training loop using the SUMOGym environment with the SUMOSim wrapper to simplify the
    setup needed for SUMO and TraCI.

    This is a very *simple* example. For meaningful training via reinforcement learning,
    you would likely need more complex environments and routing scenarios for compelling
    results for your agent(s).
    """
    import matplotlib.pyplot as plt

    from collections import defaultdict
    from os.path import join

    """Execute the TraCI training loop."""
    path = join("configs", "example")
    sim = SUMOSim(config={
        "gui": GUI_DEFAULT,
        "net-file": join(path, "traffic.net.xml"),
        "route-files": join(path, "traffic.rou.xml"),
        "additional-files": join(path, "traffic.det.xml"),
        "tripinfo-output": join(path, "tripinfo.xml")
    })

    env = SUMOGym(sim)
    env.reset()
    done = False
    step = 0
    data = {
        "actions": [],
        "steps": []
    }
    
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        data["actions"].append(info["taken_action"])
        data["steps"].append(step)
        step += 1
        data["actions"].append(info["taken_action"])
        data["steps"].append(step)

    sim.close()

    plt.plot("steps", "actions", data=data)
    plt.ylabel("Agent Actions")
    plt.xlabel("Steps")
    plt.show()
