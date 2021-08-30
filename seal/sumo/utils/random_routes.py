import random

from typing import List, Tuple, Union
from os.path import join
from seal.sumo.config import VEHICLE_LENGTH
from seal.sumo.utils import random_trips

VALID_DISTRIBUTIONS = ["arcsine", "uniform", "zipf"]
HOUR = 3600
DEFAULT_END_TIME = HOUR / 8  # (equivalent to 30 minutes)


def __extract_number_of_vehicles(n_vehicles: Union[int, Tuple[int, int]]) -> int:
    if isinstance(n_vehicles, int):
        assert n_vehicles > 0
    if isinstance(n_vehicles, tuple):
        assert len(n_vehicles) == 2, \
            "`n_vehicles` must be of len 2 if provided a tuple."
        a, b = n_vehicles
        assert a < b, \
            "`n_vehicles` must be a valid and sorted range."
        n_vehicles = random.randint(a, b)
    return n_vehicles


def __extract_end_time(end_time: Union[int, Tuple[int, int]]) -> int:
    if isinstance(end_time, tuple):
        assert len(end_time) == 2, \
            "`end_time` must be of len 2 if provided as a tuple."
        a, b = end_time
        assert a < b, \
            "`end_time` must be a valid and sorted tuple."
        end_time = random.randint(a, b)
    return end_time


def __extract_congestion_coeff(
    congestion_coeff: Union[float, Tuple[float, float]]
) -> float:
    if isinstance(congestion_coeff, tuple):
        assert len(congestion_coeff) == 2, \
            "`congestion_coeff` must be of len 2 if provided as a tuple."
        a, b = congestion_coeff
        assert a < b, \
            "`congestion_coeff` must be a valid and sorted tuple."
        congestion_coeff = random.uniform(a, b)
    return congestion_coeff


def generate_random_routes(
    netfile: str,
    number_of_lanes: int,
    generator: str="uniform",
    vehicles_per_lane_per_hour: int=90,
    number_of_hours: int=1,
    n_routefiles: int=1,
    end_time: Union[int, Tuple[int, int]]=DEFAULT_END_TIME,
    seed: float=None,
    path: str=None,
    # 90,  # Try upping this to 90 (was 60) ---- 30
) -> List[str]:
    """This function generates a *.rou.xml file for vehicles in the given road network.

    Args:
        netfile (str): Filename of the SUMO *.net.xml file.
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

    random.seed(seed)
    end_time = __extract_end_time(end_time)

    n_vehicles = vehicles_per_lane_per_hour * \
        number_of_lanes * \
        number_of_hours

    begin_time = 0
    routes = []
    for i in range(n_routefiles):
        routefile = "traffic.rou.xml" \
                    if (n_routefiles == 1) \
                    else f"traffic_{i}.rou.xml"
        if path is not None:
            routefile = join(path, routefile)

        # Use with the most recent version of randomTrips.py on GitHub.
        tripfile = join(path, "trips.trips.xml")
        args = ["--net-file", netfile, "--route-file", routefile, "-b", begin_time,
                "-e", end_time, "--length", "--period", HOUR/n_vehicles,
                "--seed", str(seed), "--output-trip-file", tripfile,
                "--fringe-factor", 100]  # ,
        # '--trip-attributes="carFollowModel=\"IDM\" tau=\"1.0\""']
        opts = random_trips.get_options(args=args)

        routes.append(routefile)
        random_trips.main(opts)

    return routes


if __name__ == "__main__":
    # Simple example of how to run the above function.
    generate_random_routes("traffic.net.xml", 100, "uniform")
