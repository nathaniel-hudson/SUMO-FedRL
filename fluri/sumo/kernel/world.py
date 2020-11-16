import numpy as np
import xml.etree.ElementTree as ET

from typing import Tuple

class World(object):

    def __init__(self, road_netfile: str, scale_factor: float=1.0):
        assert (0.0 < scale_factor) and (scale_factor <= 1.0)

        self.road_netfile = road_netfile
        self.scale_factor = scale_factor
        self.bounding_box = self.get_bounding_box()
        self.observation_box = None # self.get()
        self.text = ""

    def get_bounding_box(self) -> Tuple[float]:
        with open(self.road_netfile, "r") as f:
            # Load the provided XML file for the road network and get the location tag.
            # There should only be one location tag (hence find the first one).
            tree = ET.parse(f)
            location = tree.find("location")

            # Get the string attribute for the boundary box and then convert it into a 
            # Tuple of floats.
            boundary_box = location.attrib["convBoundary"] 
            (x1, y1), (x2, y2) = tuple([float(val) for val in boundary_box.split(",")])
            return (x1, y1), (x2, y2)
            # Bounding box is returned as: (x_min, y_min, x_max, y_max).

    def scale_world() -> np.ndarray:
        pass

    def obs_coord_to_world_coord(self, coord: Tuple[int, int]) -> Tuple[int, int]:
        pass

    def world_coord_to_obs_coord(self, coord: Tuple[int, int]) -> Tuple[int, int]:
        pass
