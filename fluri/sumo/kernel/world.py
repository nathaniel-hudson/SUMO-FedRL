import numpy as np
import traci
import xml.etree.ElementTree as ET

from typing import Tuple

X1, Y1, X2, Y2 = range(4)

class World(object):

    def __init__(
        self, 
        road_netfile: str, 
        scale_factor: float=1.0
    ):
        assert (0.0 < scale_factor) and (scale_factor <= 1.0)

        self.road_netfile = road_netfile
        self.scale_factor = scale_factor
        self.bounding_box = self.get_bounding_box()
        
        self.__width  = int(self.bounding_box[X2] - self.bounding_box[X1])
        self.__height = int(self.bounding_box[Y2] - self.bounding_box[Y1])
        self.__scaled_width  = int(self.__width  * self.scale_factor)
        self.__scaled_height = int(self.__height * self.scale_factor)

        self.origin = np.array([0, 0])
        self.world_origin = np.array([self.bounding_box[X1], self.bounding_box[Y1]])
        self.origin_scalar = self.origin - self.world_origin
        
        shape = (self.__height, self.__width)
        scaled_shape = (self.__scaled_height, self.__scaled_width)
        self.world = np.zeros(shape=shape, dtype=np.float32)
        self.scaled_world = np.zeros(shape=scaled_shape, dtype=np.float32)
        
        
    def get_bounding_box(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        with open(self.road_netfile, "r") as f:
            # Load the provided XML file for the road network and get the location tag.
            # There should only be one location tag (hence find the first one).
            tree = ET.parse(f)
            location = tree.find("location")

            # Get the string attribute for the boundary box and then convert it into a 
            # Tuple of floats.
            boundary_box = location.attrib["convBoundary"] 
            (x1, y1, x2, y2) = tuple([float(val) for val in boundary_box.split(",")])
            return (x1, y1, x2, y2)

    def get_top_left_coord(self) -> Tuple[int, int]:
        return (self.bounding_box[0], self.bounding_box[1])

    def get_bottom_right_coord(self) -> Tuple[int, int]:
        return (self.bounding_box[2], self.bounding_box[3])

    def update(
        self, 
        normalize: bool=False
    ) -> None:
        """Update the internal representations of the world in the SUMO simulation -- both
           the original and the scaled-down representation. The internal representations
           for both are numpy arrays. Each grid space represents the number of vehicles
           in that respective grid space (normalized for the scaled-down array if asked).

        Parameters
        ----------
        normalize : bool, optional
            Normalizes the number of vehicles in a given grid point in the scaled-down 
            representation by dividing each grid's value by the total number of vehicles 
            (if True), by default False.
        """
        shape = (self.__height, self.__width)
        scaled_shape = (self.__scaled_height, self.__scaled_width)
        world = np.zeros(shape=shape, dtype=np.float32)
        scaled_world = np.zeros(shape=scaled_shape, dtype=np.float32)

        veh_ids = list(traci.vehicle.getIDList())
        for veh_id in veh_ids:
            # Get the x- or y-coordinates for the world. Add a normalized weight to the 
            # respective coordinate in the world. For it to be normalized, we need to 
            # change `dtype` to a float-based value.
            x, y = self.convert_coords(traci.vehicle.getPosition(veh_id))
            world[y, x] += 1
            x, y = int(x * self.scale_factor), int(y * self.scale_factor)
            scaled_world[y, x] += 1

        self.world = world
        self.scaled_world = scaled_world if (normalize == False) \
                            else scaled_world / len(veh_ids)


    def observe(
        self, 
        top_left: Tuple[int, int]=None, 
        bottom_right: Tuple[int, int]=None,
        use_scaled_world: bool=True
    ) -> np.ndarray:
        assert all([top_left, bottom_right]) == any([top_left, bottom_right]), \
            "Values for `top_left` and `bottom_right` must both either be None or tuples."

        if top_left == None and bottom_right == None:
            world = self.scaled_world if use_scaled_world else self.world
            return world.copy()
        else:
            x1, y1 = top_left
            x2, y2 = bottom_right
            w, h = self.scaled_world.shape if use_scaled_world \
                else (self.__width, self.__height)

            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 >= w: x2 = self.w - 1
            if y2 >= h: y2 = self.h - 1

            world = self.scaled_world if use_scaled_world else self.world
            return world[x1:x2, y1:y2]


    def get_dimensions(self, scaled_down: bool=True) -> np.ndarray:
        if scaled_down:
            return self.scaled_world.shape
        else:
            return self.world.shape

    def convert_coords(self, coord: Tuple[int, int], use_scaled_world: bool=True) -> Tuple[int, int]:
        clip = lambda val, max_val: max(0, min(val, max_val))
        
        x, y = coord
        if use_scaled_world:
            x, y = clip(x, self.__scaled_width), clip(y, self.__scaled_height)
        else:
            x, y = clip(x, self.__width), clip(y, self.__height)

        point = np.array([float(x), float(y)]) + self.origin_scalar
        return point.astype(int)


    def obs_coord_to_world_coord(self, coord: Tuple[int, int]) -> Tuple[int, int]:
        pass

    def world_coord_to_obs_coord(self, coord: Tuple[int, int]) -> Tuple[int, int]:
        pass
