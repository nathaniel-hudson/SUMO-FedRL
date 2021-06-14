import networkx as nx
import numpy as np
import random
import traci
import warnings
import xml.etree.ElementTree as ET

from collections import OrderedDict
from gym import spaces
from typing import Any, Dict, List, Set, Tuple, Union

from fluri.sumo.utils.core import get_node_id
from fluri.sumo.const import *
from fluri.sumo.kernel.const import *


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
        force_all_red: bool=False,
        ranked: bool=True
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

    def __depr_get_observation(self) -> np.ndarray:
        """Returns an observation of the traffic light, with each of the features of
           interest.

        Returns:
            np.ndarray: Array containing the values of each of the features.
        """
        num_vehs = 0
        avg_speed = 0
        wait_time = 0
        num_halt = 0

        lanes = traci.trafficlight.getControlledLanes(self.id)
        for l in lanes:
            num_vehs += traci.lane.getLastStepVehicleNumber(l)
            # TODO: Unfair avg.
            avg_speed += traci.lane.getLastStepMeanSpeed(l) / len(l)
            # NOTE: This should be averaged across all vehicles...
            wait_time += traci.lane.getWaitingTime(l)
            num_halt += traci.lane.getLastStepHaltingNumber(l)

        return np.array([num_vehs, avg_speed, wait_time, num_halt, self.state])

    def get_observation(self) -> np.ndarray:
        congestion = 0
        halting_vehs = 0
        speed = 0

        lanes = traci.trafficlight.getControlledLanes(self.id)
        for l in lanes:
            vehs = traci.lane.getLastStepVehicleIDs(l)
            lane_length = traci.lane.getLength(l)
            vehs_length = sum(traci.vehicle.getLength(v) for v in vehs)

            congestion += (vehs_length/lane_length) / len(lanes)
            halting_vehs += traci.lane.getLastStepHaltingNumber(l) #/ len(lanes)
            speed += traci.lane.getLastStepMeanSpeed(l) / len(lanes)

        state = [congestion, halting_vehs, speed, self.state]
        if self.ranked:
            # Local and global ranks (to be filled later).
            state.extend([0, 0])
        return np.array(state)


    @property
    def action_space(self) -> spaces.Box:
        return spaces.Box(low=0, high=1, shape=(1,), dtype=int)


    @property
    def observation_space(self) -> spaces.Box:
        dtype = np.float64
        high = np.finfo(dtype).max
        n_features = N_RANKED_FEATURES if self.ranked else N_UNRANKED_FEATURES
        return spaces.Box(low=0, high=high, shape=(n_features,), dtype=dtype)

        ## TODO: We NEED to fix the bounding issues here. The `high` upper bound is too
        ##       large for bounded features (i.e., congestion, rank).
        # max_lane_len = 0
        # spaces = [ # Add unranked features first/
        #     spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64),
        #     spaces.Box(low=0.0, high=max_lane_len, shape=(1,), dtype=np.float64),
        #     spaces.Box(low=0.0, high=max_speed, shape=(1,), dtype=np.float64),
        #     spaces.Discrete(max_num_states)

        # ] 
        # if self.ranked:
        #     spaces.extend([
        #         spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64),
        #         spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64)
        #     ])
        # return spaces.Tuple(tuple(spaces))


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
        ranked: bool=RANK_DEFAULT
    ):
        self.road_netfile = road_netfile
        self.ids = sorted([tls_id for tls_id in self.get_traffic_light_ids()])
        self.index2id = {index:  tls_id for index,
                         tls_id in enumerate(self.ids)}
        self.id2index = {tls_id: index for index,
                         tls_id in enumerate(self.ids)}
        self.hub = OrderedDict({
            tls_id: TrafficLight(index, tls_id, self.road_netfile, sort_phases, 
                                 ranked=ranked)
            for index, tls_id in self.index2id.items()
        })
        self.ranked = ranked
        self.tls_graph = self.get_tls_graph()

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

    def get_tls_graph(self) -> Dict[str, List[str]]:
        graph = {}

        # Uses TRACI and causes issues due to SUMO not being started before
        # this is called.
        # for tls_id in self.ids:
        #     neighbors = set()
        #     lanes = traci.trafficlight.getControlledLanes(tls_id)
        #     for other_id in self.ids:
        #         if tls_id == other_id:
        #             continue
        #         other_lanes = traci.trafficlight.getControlledLanes(other_id)
        #         for lane in other_lanes:
        #             if lane in lanes:
        #                 neighbors.add(other_id)
        #     graph[tls_id] = list(neighbors)

        tls_id_set = set(self.ids)
        with open(self.road_netfile, "r") as f:
            tree = ET.parse(f)
            edges = tree.findall("edge")
            for tls_id in tls_id_set:
                neighbors = set()
                other_tls_id_set = tls_id_set - {tls_id}
                for e in edges:
                    for other_tls_id in other_tls_id_set:
                        cond = e.attrib.get("from", None) == tls_id and \
                               e.attrib.get("to", None) == other_tls_id
                        if cond:
                            neighbors.add(other_tls_id)
                graph[tls_id] = list(neighbors)

        return graph

    def update(self) -> None:
        """Update the current states by interfacing with SUMO directly using SumoKernel.
        """
        for tls in self.hub.values():
            tls.update()

    def __iter__(self) -> iter:
        return iter(self.hub.values())

    def __getitem__(self, tls_id: str) -> TrafficLight:
        return self.hub[tls_id]

    def __len__(self) -> int:
        return len(self.ids)
