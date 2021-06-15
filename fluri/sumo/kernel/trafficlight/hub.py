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
from fluri.sumo.kernel.trafficlight.light import TrafficLight


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
