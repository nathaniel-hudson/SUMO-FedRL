from random import randint
from typing import List, Tuple, Union
from os.path import join
from seal.sumo.config import VEHICLE_LENGTH
from seal.sumo.utils import random_trips

VALID_DISTRIBUTIONS = ["arcsine", "uniform", "zipf"]
DEFAULT_END_TIME = 3600

def __extract_number_of_vehicles(n_vehicles: Union[int, Tuple[int, int]]) -> int:
    if isinstance(n_vehicles, int):
        assert n_vehicles > 0
    if isinstance(n_vehicles, tuple):
        assert len(n_vehicles) == 2, \
            "`n_vehicles` must be of len 2 if provided a tuple."
        a, b = n_vehicles
        assert a < b, \
            "`n_vehicles` must be a valid and sorted range."
        n_vehicles = randint(a, b)
    return n_vehicles

def __extract_end_time(end_time: Union[int, Tuple[int, int]]) -> int:
    if isinstance(end_time, tuple):
        assert len(end_time) == 2, \
            "`end_time` must be of len 2 if provided a tuple."
        a, b = end_time
        assert a < b, \
            "`end_time` must be a valid and sorted range."
        end_time = randint(a, b)
    return end_time

def generate_random_routes(
    net_name: str,
    n_vehicles: Union[int, Tuple[int, int]]=(500, 1000),
    generator: str="uniform",
    n_routefiles: int=1,
    end_time: Union[int, Tuple[int, int]]=DEFAULT_END_TIME,
    seed: float=None,
    path: str=None,
    road_capacity: float=1.0,                                # TODO
    vehicle_length: float=VEHICLE_LENGTH,                    # TODO
    congestion_coeff: Union[float, Tuple[float, float]]=1.0, # TODO
    dynamic_congestion: bool=True                            # TODO
) -> List[str]:
    """This function generates a *.rou.xml file for vehicles in the given road network.

    Args:
        net_name (str): Filename of the SUMO *.net.xml file.
        n_vehicles (Union[int, Tuple[int, int]]): Number of vehicles to be used in the 
            simulation.
        generator (str): A token that specifies the random distribution that will be 
            used to assigning routes to vehicles.
        n_routefiles (int): Number of routefiles to be generated. Typically no reason 
            to change this input.
        end_time (Union[int, Tuple[int, int]]): When the simulation ends --- this affects 
            the number of vehicles.
        seed (float): A random seed to fix the generator distribution, by default None.
    
    Returns:
        List[str]: A list containing the names of the randomly-generated route files.
    """
    assert generator.lower() in VALID_DISTRIBUTIONS

    end_time = __extract_end_time(end_time)
    n_vehicles = __extract_number_of_vehicles(n_vehicles)

    print(f">>> random_routes.py: generate_random_routes(): seed={seed}")
    # ...
    if dynamic_congestion:
        n_vehicles = int(0.025 * congestion_coeff * (road_capacity/vehicle_length))
        print(f">>> random_routes.py: `n_vehicles` = {n_vehicles}")

    begin_time = 0
    routes = []
    for i in range(n_routefiles):
        routefile = "traffic.rou.xml" \
                    if (n_routefiles == 1) \
                    else f"traffic_{i}.rou.xml"
        if path is not None:
            routefile = join(path, routefile)

        # TODO: Run a small script here that generates a n_vehicles value based on a 
        #       float  and the lane occupancy of the netfile under consideration.

        # Use with the most recent version of randomTrips.py on GitHub.
        tripfile = join(path, "trips.trips.xml")
        args = ["--net-file", net_name, "--route-file", routefile, "-b", begin_time,
                "-e", end_time, "--length", "--period", end_time/n_vehicles,
                "--seed", str(seed), "--output-trip-file", tripfile]
        opts = random_trips.get_options(args=args)

        routes.append(routefile)
        random_trips.main(opts)

    return routes


if __name__ == "__main__":
    # Simple example of how to run the above function.
    generate_random_routes("traffic.net.xml", 100, "uniform")
