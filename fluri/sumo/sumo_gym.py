import gym
import numpy as np
import sumo_util as utils
import traci

from const import *
from collections import defaultdict
from gym import spaces
from sumo_sim import SUMOSim

from typing import Any, Dict, List, Tuple

"""
TODO:
[x] We need to dynamically collect the names of the traffic light programs for a given
    SUMO config to support the action performance.
[_] Streamline the entire SUMO/TraCI interface with `SUMOGym`.
[_] Finalize the representation of the SUMO-Gym (i.e., what are all of the needed class
    members, how tightly encapsulated is the SUMOSimulation interface going to be, how
    is the `curr_light` member going to be represented?).
[_] Setup the boundary bounds for limiting the view an agent can have.
"""

class SUMOGym(gym.Env):
    """Custom Gym environment designed for simple RL experiments using SUMO/TraCI."""
    name = "SUMO-v1"
    metadata = {"render.modes": ["sumo", "sumo-gui"]}
    MAIN_TASK = 0
    TRANS_TASK = 1
    GRID_KEY = "grid"
    TLS_KEY  = "traffic_lights"
    
    def __init__(self, sim: SUMOSim):
        self.sim = sim

        self.trafficlights = None
        self.n_trafficlights = None
        self.curr_trafficlights = None
        self.trafficlight_network = None
        
        self.mask = None
        self.bounding_box = ((0.0, 0.0), (1020.0, 1020.0))

        # TODO: These are parameters for features that are currently not being considered.
        self.trafficlight_radius = 0


    def __do_action(self, action) -> None:
        """TODO"""
        return None

        for tls_id, curr_state in self.curr_trafficlights.items():
            curr_action = self.trafficlights[tls_id].index(curr_state)
            next_action = self.action[int(tls_id)]

            curr_node = utils.get_node_id(tls_id, curr_action)
            next_node = utils.get_node_id(tls_id, next_action)
            is_valid = next_action in self.trafficlight_network.neighbors(curr_node)
            if is_valid and self.can_change(tls_id): ## TODO: We need to implement `can_change()`.
                traci.trafficlight.setRedYellowGreenState(tls_id, next_action)
                if curr_action != next_action:
                    ## TODO: We need to add in the part related to the mask for delay.
                    pass

        trafficlight_that_can_change = (self.mask == 0)
        for tlsID in range(len(trafficlight_that_can_change)):
            if trafficlight_that_can_change[tlsID]:
                program_id = action[tlsID] // 2
                traci.trafficlight.setProgram(str(tlsID), str(program_id))
                self.curr_light[tlsID] = (program_id, self.MAIN_TASK)


    def __get_observation(self) -> Dict[np.ndarray, np.ndarray]:
        """Returns the current observation of the state space, represented by the grid
           space for recognizing vehicle locations and the current state of all traffic
           lights.
        """
        (x_min, y_min), (x_max, y_max) = self.bounding_box
        width = int(x_max - x_min)
        height = int(y_max - y_min)
        obs = {
            self.GRID_KEY: np.zeros(shape=(width, height), dtype=np.int32),
            self.TLS_KEY:  np.zeros(shape=(len(self.trafficlights)), dtype=np.int32)
        }

        for veh_id in list(traci.vehicle.getIDList()):
            x, y = traci.vehicle.getPosition(veh_id)
            obs[self.GRID_KEY][int(x), int(y)] = 1

        for tls_id, curr_state in self.curr_trafficlights.items():
            index = self.trafficlights[tls_id].index(curr_state)
            obs[self.TLS_KEY][int(tls_id)] = index

        return obs


    def __get_reward(self) -> float:
        """TODO"""
        return -1.0


    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        """Performs a single step in the environment, as per the Open AI Gym framework."""
        self.__do_action(action)
        traci.simulationStep()

        observation = self.__get_observation()
        reward = self.__get_reward()
        done = self.sim.done()
        info = {}

        return observation, reward, done, info


    def reset(self):
        """TODO"""
        if self.sim.is_loaded():
            self.sim.close()
        
        self.sim.start()
        self.trafficlights = self.sim.get_all_possible_tls_states()
        self.n_trafficlights = len(self.trafficlights)
        self.trafficlight_network = utils.make_tls_state_network(self.trafficlights)

        self.mask = np.zeros(shape=(self.n_trafficlights))
        self.bounding_box = ((0.0, 0.0), (1020.0, 1020.0))


    @property
    def action_space(self):
        """Initializes an instance of the action space as a property of the class."""
        ## TODO: We need to adjust the `sample()` function for this action_space such that
        ##       it restricts available actions based on the current action.
        return spaces.MultiDiscrete([
            len(self.trafficlights[tls_id]) for tls_id in self.trafficlights
        ])


    @property
    def observation_space(self):
        """Initializes an instance of the observation space as a property of the class."""
        (x_min, y_min), (x_max, y_max) = ((0.0, 0.0), (1020.0, 1020.0))
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
            len(self.trafficlights[tls_id]) for tls_id in self.trafficlights
        ])

        return spaces.Dict({
            self.GRID_KEY: grid_space, 
            self.TLS_KEY:  tls_space
        })


if __name__ == "__main__":
    from collections import defaultdict
    from configs.example.runner import generate_routefile
    from os.path import join

    path = join("configs", "example")
    sim = SUMOSim(config={
        "gui": False,
        "net-file": join(path, "traffic.net.xml"),
        "route-files": join(path, "traffic.rou.xml"),
        "additional-files": join(path, "traffic.det.xml"),
        "tripinfo-output": join(path, "tripinfo.xml")
    })

    env = SUMOGym(sim)
    done = False
    step = 0

    """Execute the TraCI training loop."""
    env.reset()

    action = env.action_space.sample()
    obs = env.observation_space.sample()
    print(f"Action -> {action}\nObs -> {obs}")
    print(obs["traffic_lights"])
    
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        step += 1

    traci.close()
