import networkx as nx
import numpy as np
import random
import traci
import warnings
import xml.etree.ElementTree as ET

from collections import OrderedDict
from typing import Any, Dict, List, Set, Tuple, Union

from ..utils import helper as utils
from ..utils.core import get_node_id
from ..const import *

SORT_DEFAULT = True
NEXT_STATES = {
    "G": set(["G", "g", "y"]),
    "g": set(["G", "g", "y"]),
    "y": set(["y", "r"]),
    "r": set(["G", "g", "r"])
}

class TrafficLight:
    """This class represents an indidivual `trafficlight` (tls) in SUMO. The purpose is to
       simplify the necessary code for the needs of the RL environments in FLURI.
    """
    def __init__(
        self, 
        tls_id: int, 
        netfile: str,
        sort_phases: bool=SORT_DEFAULT,
        index: int=None,
        force_all_red: bool=False
    ):
        # The `index` data member is for the consistently simple indexing for actions
        # that are represented via lists. This is important for the `stable-baselines`
        # implementation that does not support Dict spaces.
        self.index: int = index 
        self.id: int = tls_id
        self.program: List[str] = self.get_program(netfile, sort_phases, force_all_red)
        self.num_phases: int = len(self.program)
        self.state: int = random.randrange(self.num_phases)
        self.phase: str = self.program[self.state]

    def curr_state(self) -> None:
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

    def valid_next_state(self, next_state: str) -> bool:
        """Determines if `next_state` is valid given the current state.

        Parameters
        ----------
        next_state : str
            The proposed next state provided by the user.

        Returns
        -------
        bool
            Returns True if `next_state` is a valid transition, False otherwise.
        """
        curr_node = get_node_id(self.id, self.state)
        next_node = get_node_id(self.id, next_state)
        is_valid = next_node in self.action_transition_net.neighbors(curr_node)
        return is_valid

    def get_observation(self) -> OrderedDict:
        """Returns an observation of the traffic light, with each of the features of 
           interest.

        Returns
        -------
        np.ndarray
            Array containing the values of each of the features.
        """
        total_num_vehs = 0
        total_avg_speed = 0
        total_num_occupancy = 0
        total_wait_time = 0
        total_travel_time = 0
        total_num_halt = 0

        for lane in traci.trafficlight.getControlledLanes(self.id):
            num_vehs, avg_speed, num_occupancy, wait_time, travel_time, num_halt = \
                traci.lane.getLastStepVehicleNumber(lane), \
                traci.lane.getLastStepMeanSpeed(lane), \
                traci.lane.getLastStepOccupancy(lane), \
                traci.lane.getWaitingTime(lane), \
                traci.lane.getTraveltime(lane), \
                traci.lane.getLastStepHaltingNumber(lane)
            
            total_num_vehs += num_vehs
            total_avg_speed += avg_speed
            total_num_occupancy += num_occupancy
            total_wait_time += wait_time
            total_travel_time += travel_time
            total_num_halt += num_halt

        return OrderedDict({
            "num_vehicles":  total_num_vehs,
            "avg_speed":     total_avg_speed,
            "num_occupancy": total_num_occupancy,
            "wait_time":     total_wait_time,
            "travel_time":   total_travel_time,
            "num_halt":      total_num_halt,
            "curr_state":    self.state,
        })

    def get_program(
        self,
        road_netfile: str, 
        sort_phases: bool=SORT_DEFAULT,
        force_all_red: bool=False
    ) -> List[str]:
        """Get the possible traffic light phases for the specific traffic light via the
            given `tls_id`.

        Parameters
        ----------
        tls_id : str
            The traffic light id.
        sort_phases : bool, optional
            Sorts the possible phases if True, by default SORT_DEFAULT

        Returns
        -------
        List[str]
            A list of all the possible phases the given traffic light can take.
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


class TrafficLightHub:
    """
    TODO: Fill in.
    """

    def __init__(
        self, 
        road_netfile: str, 
        sort_phases: bool=SORT_DEFAULT, 
        obs_radius: int=None
    ):
        self.road_netfile = road_netfile
        self.ids = sorted([tls_id for tls_id in self.get_traffic_light_ids()])
        self.hub = {
            tls_id: TrafficLight(tls_id, self.road_netfile, sort_phases, index=i)
            for i, tls_id in enumerate(self.ids)
        }

        # TODO: Currently not being considered.
        self.radii = None

    def get_traffic_light_ids(self) -> List[str]:
        """Get a list of all the traffic light IDs in the *.net.xml file.

        Returns
        -------
        List[str]
            A list of all the traffic light IDs.
        """
        with open(self.road_netfile, "r") as f:
            tree = ET.parse(f)
            junctions = tree.findall("junction")
            trafficlights = []
            for j in junctions:
                if j.attrib["type"] == "traffic_light":
                    trafficlights.append(j.attrib["id"])
            return trafficlights

    def update_current_states(self) -> None:
        """Update the current states by interfacing with SUMO directly using SumoKernel.
        """
        for tls in self.hub.values():
            tls.update_current_state()

    def __iter__(self):
        return self.hub.values().__iter__()

    def __getitem__(self, tls_id: str) -> TrafficLight:
        return self.hub[tls_id]

    def __len__(self) -> int:
        return len(self.ids)

