import networkx as nx
import numpy as np
import random
import traci
import xml.etree.ElementTree as ET

from gym import spaces
from scipy import stats
from typing import List

from seal.sumo.config import *
from seal.sumo.kernel.const import *
from seal.sumo.kernel.trafficlight.space import trafficlight_space


class TrafficLight:
    """This class represents an indidivual `trafficlight` (tls) in SUMO. The purpose is to
       simplify the necessary code for the needs of the RL environments in seal.
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

    @property
    def action_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=1, shape=(1,), dtype=int)

    @property
    def observation_space(self) -> spaces.Tuple:
        return trafficlight_space(self.ranked)

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
            self.state = next_state
            self.phase = next_phase
            traci.trafficlight.setRedYellowGreenState(self.id, self.phase)
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
        max_lane_speeds = vehicle_speeds = 0
        lane_lengths = vehicle_lengths = halted_lengths = 0
        for l in traci.trafficlight.getControlledLanes(self.id):
            lane_lengths += traci.lane.getLength(l)
            max_speed = traci.lane.getMaxSpeed(l)
            for v in traci.lane.getLastStepVehicleIDs(l):
                speed = traci.vehicle.getSpeed(v)
                vehicle_speeds += speed
                max_lane_speeds += max_speed
                vehicle_lengths += traci.vehicle.getLength(v)
                if speed < HALTING_SPEED:
                    halted_lengths += traci.vehicle.getLength(v)

        # NEW.
        lane_lengths = sum(traci.lane.getLength(l)
                           for l in traci.trafficlight.getControlledLanes(self.id))
        vehicle_lengths = sum(sum(traci.vehicle.getLength(v)
                                  for v in traci.lane.getLastStepVehicleIDs(l))
                              for l in traci.trafficlight.getControlledLanes(self.id))
        halted_lengths = sum(sum(traci.vehicle.getLength(v)
                                 for v in traci.lane.getLastStepVehicleIDs(l)
                                 if traci.vehicle.getLength(v) < HALTING_SPEED)
                             for l in traci.trafficlight.getControlledLanes(self.id))

        obs[LANE_OCCUPANCY] = min(vehicle_lengths/lane_lengths, 1.0)
        obs[HALTED_LANE_OCCUPANCY] = min(halted_lengths/lane_lengths, 1.0)

        # TODO: Split into auxiliary functions...

        try:
            # We need to "clip" average speed because drivers sometimes exceed the
            # speed limit.
            obs[SPEED_RATIO] = min(1.0, vehicle_speeds / max_lane_speeds)
        except ZeroDivisionError:
            # This happens when 0 vehicles are on lanes controlled by the traffic light.
            obs[SPEED_RATIO] = 0.0

        # Extract descriptive statistics features for the current traffic light state.
        curr_tls_state = traci.trafficlight.getRedYellowGreenState(self.id)
        # print(f">> curr_tls_state={curr_tls_state}")
        for light in curr_tls_state:
            if light == "r":
                obs[PHASE_STATE_r] += 1/len(curr_tls_state)
            elif light == "y":
                obs[PHASE_STATE_y] += 1/len(curr_tls_state)
            elif light == "g":
                obs[PHASE_STATE_g] += 1/len(curr_tls_state)
            elif light == "G":
                obs[PHASE_STATE_G] += 1/len(curr_tls_state)
            elif light == "u":
                obs[PHASE_STATE_u] += 1/len(curr_tls_state)
            elif light == "o":
                obs[PHASE_STATE_o] += 1/len(curr_tls_state)
            elif light == "O":
                obs[PHASE_STATE_O] += 1/len(curr_tls_state)

        return np.array(obs)

    def __get_lane_occupancy(self) -> float:
        pass

    def __get_halted_lane_occupancy(self) -> float:
        pass

    def __get_speed_ratio(self) -> float:
        pass

    def __get_phase_mode(self) -> float:
        pass

    def __get_phase_std(self) -> float:
        pass
