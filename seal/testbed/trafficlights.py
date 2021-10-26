import numpy as np
import random
import warnings

from collections import OrderedDict
from gym import spaces
from typing import Any, Dict, List, Set, Tuple, Union
from seal.sumo.config import *
from seal.sumo.kernel.const import *
from seal.testbed.kernel.trafficlight.traffic_light import TrafficLight

class TrafficNetwork():
    def __init__(self, config, ranked, lights, pynode):
        self.config = config
        self.road_netfile = config["net-file"]
        self.ids = sorted(lights)   #list
        self.hub = OrderedDict({tls_id: TrafficLight(tls_id, self.config, ranked, pynode) for tls_id in lights})
        self.ranked = ranked
        self.tls_graph = self.get_tls_graph()

    def get_tls_graph(self):
        graph = {}
        # tls_id_set = set(self.ids)
        # with open(self.road_netfile, "r") as f:
        #     tree = ET.parse(f)
        #     edges = tree.findall("edge")
        #     for tls_id in tls_id_set:
        #         neighbors = set()
        #         other_tls_id_set = tls_id_set - {tls_id}
        #         for e in edges:
        #             for other_tls_id in other_tls_id_set:
        #                 cond = e.attrib.get("from", None) == tls_id and \
        #                     e.attrib.get("to",   None) == other_tls_id
        #                 if cond:
        #                     neighbors.add(other_tls_id)
        #         graph[tls_id] = list(neighbors)
        graph[0] = [1]
        graph[1] = [0]
        # graph[2] = [1,3]
        # graph[3] = [2,0]
        return graph

    def update(self):
        for tls in self.hub.values():
            tls.update()

    def __iter__(self) -> iter:
        return iter(self.hub.values())

    def __getitem__(self, tls_id: str) -> TrafficLight:
        return self.hub[tls_id]

    def __len__(self) -> int:
        return len(self.ids)