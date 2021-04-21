import networkx as nx
import numpy as np
import random
import traci
import warnings
import xml.etree.ElementTree as ET

from collections import OrderedDict
from gym import spaces
from typing import Any, Dict, List, Set, Tuple, Union

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
        index: int,
        tls_id: int,
        netfile: str,
        sort_phases: bool=SORT_DEFAULT,
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
        """Returns an observation of the traffic light, with each of the features of
           interest.

        Returns:
            np.ndarray: Array containing the values of each of the features.
        """
        num_vehs = 0
        avg_speed = 0
        num_occupancy = 0
        wait_time = 0
        travel_time = 0
        num_halt = 0

        lanes = traci.trafficlight.getControlledLanes(self.id)
        for l in lanes:
            num_vehs += traci.lane.getLastStepVehicleNumber(l)
            # TODO: Unfair avg.
            avg_speed += traci.lane.getLastStepMeanSpeed(l) / len(l)
            wait_time += traci.lane.getWaitingTime(l) ## NOTE: This should be averaged across all vehicles...
            num_halt += traci.lane.getLastStepHaltingNumber(l)

            # NOTE: Incompatible with the real-world testbed.
            # num_occupancy += traci.lane.getLastStepOccupancy(l)

            # NOTE: Difficult to track in real-world testbed.
            # travel_time += traci.lane.getTraveltime(l)

        # return np.array([num_vehs, avg_speed, num_occupancy, wait_time,
        #                  travel_time, num_halt, self.state])

        return np.array([num_vehs, avg_speed, wait_time, num_halt, self.state])

    @property
    def action_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=1, shape=(1,), dtype=int)

    @property
    def observation_space(self) -> spaces.Box:
        dtype = np.float64
        high = np.finfo(dtype).max
        return spaces.Box(low=0, high=high, shape=(N_FEATURES,), dtype=dtype)


class TrafficLightHub:
    """A simple data structure that stores all of the trafficlights in a given SUMO
       simulation. This class should be used for initializing and creating instances of
       trafficlight objects (for simplicity). Additionally, this class supports indexing
       and iteration.
    """

    def __init__(
        self,
        road_netfile: str,
        sort_phases: bool=SORT_DEFAULT,
    ):
        self.road_netfile = road_netfile
        self.ids = sorted([tls_id for tls_id in self.get_traffic_light_ids()])
        self.index2id = {index:  tls_id for index,
                         tls_id in enumerate(self.ids)}
        self.id2index = {tls_id: index for index,
                         tls_id in enumerate(self.ids)}
        self.hub = OrderedDict({
            tls_id: TrafficLight(index, tls_id, self.road_netfile, sort_phases)
            for index, tls_id in self.index2id.items()
        })

    def get_traffic_light_ids(self) -> List[str]:
        """Get a list of all the traffic light IDs in the provided *.net.xml file.

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

    def update(self) -> None:
        """Update the current states by interfacing with SUMO directly using SumoKernel.
        """
        for tls in self.hub.values():
            tls.update()

    def __iter__(self) -> iter:
        return self.hub.values().__iter__()

    def __getitem__(self, tls_id: str) -> TrafficLight:
        return self.hub[tls_id]

    def __len__(self) -> int:
        return len(self.ids)
