import networkx as nx
import numpy as np
import random
import traci
import warnings
import xml.etree.ElementTree as ET

from collections import OrderedDict
from gym import spaces
from scipy import stats
from typing import Any, Dict, List, Set, Tuple, Union

from fluri.sumo.utils.core import get_node_id
from fluri.sumo.config import *
from fluri.sumo.kernel.const import *
from fluri.sumo.kernel.trafficlight.space import trafficlight_space


class TrafficLight:
    """This class represents an indidivual `trafficlight` (tls) in SUMO. The purpose is to
       simplify the necessary code for the needs of the RL environments in FLURI.
    """

    index: int
    id: int
    program: List[int]
    num_phases: int
    state: int
    phase: str
    ranked: bool

    def __init__(
        self,
        index: int,
        tls_id: int,
        netfile: str,
        sort_phases: bool=SORT_DEFAULT,
        force_all_red: bool=False,
        ranked: bool=True
    ):
        # The `index` data member is for the consistently simple indexing for actions
        # that are represented via lists. This is important for the `stable-baselines`
        # implementation that does not support Dict spaces.
        self.index = index
        self.id = tls_id
        self.program = self.get_program(netfile, sort_phases, force_all_red)
        self.num_phases = len(self.program)
        self.state = random.randrange(self.num_phases)
        self.phase = self.program[self.state]
        self.ranked = ranked

    def update(self) -> None:
        """Update the current state by interacting with `traci`."""
        try:
            self.phase = traci.trafficlight.getRedYellowGreenState(self.id)
            self.state = self.program.index(self.phase)
        except traci.exceptions.FatalTraCIError:
            pass

    def next_phase(self) -> None:
        next_state = (self.state+1) % self.num_phases
        next_phase = self.program[next_state]
        try:
            traci.trafficlight.setRedYellowGreenState(self.id, self.phase)
            self.state = next_state
            self.phase = next_phase
        except traci.exceptions.FatalTraCIError:
            pass

    def get_program(
        self,
        road_netfile: str,
        sort_phases: bool=SORT_DEFAULT,
        force_all_red: bool=False
    ) -> List[str]:
        """Get the possible traffic light phases for the specific traffic light via the
           given `tls_id`.

        Args:
            road_netfile (str): The traffic light id.
            sort_phases (bool, optional): Sorts the possible phases if True. Defaults to
                SORT_DEFAULT.
            force_all_red (bool, optional): Requires there's a state of all red lights if
                True. Defaults to False.

        Returns:
            List[str]: A list of all the possible phases the given traffic light can take.
        """
        with open(road_netfile, "r") as f:
            tree = ET.parse(f)
            logic = tree.find(f"tlLogic[@id='{self.id}']")
            states = [phase.attrib["state"] for phase in logic]

            if force_all_red:
                all_reds = len(states[0]) * "r"
                if all_reds not in states:
                    states.append(all_reds)

            return states if (sort_phases == False) else sorted(states)

    def get_observation(self) -> np.ndarray:
        # Initialize the observation list (obs).
        n_features = N_RANKED_FEATURES if self.ranked else N_UNRANKED_FEATURES
        obs = [0 for _ in range(n_features)]


        # Extract the lane-specific features.
        max_lane_speeds = mean_lane_speeds = 0
        total_lane_length = vehicle_lengths = halted_vehicle_lengths = 0
        for l in traci.trafficlight.getControlledLanes(self.id):
            mean_lane_speeds += traci.lane.getLastStepMeanSpeed(l)
            max_lane_speeds += traci.lane.getMaxSpeed(l)
            total_lane_length += traci.lane.getLength(l)
            for v in traci.lane.getLastStepVehicleIDs(l):
                vehicle_lengths += traci.vehicle.getLength(v)
                if traci.vehicle.getSpeed(v) < HALTING_SPEED:
                    halted_vehicle_lengths += traci.vehicle.getLength(v)

        obs[CONGESTION] = vehicle_lengths / total_lane_length
        obs[HALT_CONGESTION] = halted_vehicle_lengths / total_lane_length
        obs[AVG_SPEED] = min(1.0, mean_lane_speeds / max_lane_speeds) 
        # ^^ We need to "clip" average speed because drivers sometimes exceed the 
        #    speed limit.

        # Extract descriptive statistics features for the current traffic light state.
        curr_tls_state = traci.trafficlight.getRedYellowGreenState(self.id)
        curr_tls_state_arr = [STATE_STR_TO_INT[phase]
                              for phase in curr_tls_state]
        obs[CURR_STATE_MODE] = stats.mode(curr_tls_state_arr)[0].item() / NUM_TLS_STATES
        obs[CURR_STATE_STD] = np.std(curr_tls_state_arr) / NUM_TLS_STATES

        return np.array(obs)

    @property
    def action_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=1, shape=(1,), dtype=int)

    @property
    def observation_space(self) -> spaces.Tuple:
        return trafficlight_space(self.ranked)

        # dtype = np.float64
        # high = np.finfo(dtype).max
        # n_features = N_RANKED_FEATURES if self.ranked else N_UNRANKED_FEATURES
        # return spaces.Box(low=0, high=high, shape=(n_features,), dtype=dtype)

        # TODO: Implement statistics-based traffic light state representation.

        # TODO: We NEED to fix the bounding issues here. The `high` upper bound is too
        # large for bounded features (i.e., congestion, rank).
        '''
        dtype = np.float64
        max_num_halt = 0  # TODO: Dynamically get this.
        max_speed = 75  # TODO: Dynamically get this.
        max_num_states = 5  # TODO: Dynamically get this.
        space_list = [  # Add unranked features first/
            spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=dtype),
            spaces.Box(low=0.0, high=max_num_halt, shape=(1,), dtype=int),
            spaces.Box(low=0.0, high=max_speed, shape=(1,), dtype=dtype),
            spaces.Discrete(max_num_states)
        ]
        if self.ranked:
            space_list.extend([
                spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=dtype),
                spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=dtype)
            ])
        return spaces.Tuple(tuple(space_list))
        '''
