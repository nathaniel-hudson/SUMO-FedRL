from random import randint
from typing import List, Tuple, Union
from os.path import join

from . import random_routes_support as rrs

VALID_DISTRIBUTIONS = ["arcsine", "uniform", "zipf"]

def generate_random_routes(
    net_name: str, 
    n_vehicles: Union[int, Tuple[int, int]]=(100, 1000),
    generator: str="uniform",
    n_routefiles: int=1,
    end_time: Union[int, Tuple[int, int]]=(1500, 3000),
    seed: float=None,
    path: str=None,
) -> List[str]:
    """This function generates a *.rou.xml file for vehicles in the given road network.

    Parameters
    ----------
    net_name : str
        Filename of the SUMO *.net.xml file.
    n_vehicles : int
        Number of vehicles to be used in the simulation.
    generator : str
        A token that specifies the random distribution that will be used to assigning
        routes to vehicles.
    seed : float, optional
        A random seed to fix the generator distribution, by default None
    """
    if isinstance(n_vehicles, int): assert n_vehicles > 0
    assert generator.lower() in VALID_DISTRIBUTIONS

    if isinstance(n_vehicles, tuple):
        assert len(n_vehicles) == 2, "`n_vehicles` must be of len 2 if provided a tuple."
        a, b = n_vehicles
        assert a < b, "`n_vehicles` must be a valid and sorted range."
        n_vehicles = randint(a, b)

    if isinstance(end_time, tuple):
        assert len(end_time) == 2, "`end_time` must be of len 2 if provided a tuple."
        a, b = end_time
        assert a < b, "`end_time` must be a valid and sorted range."
        end_time = randint(a, b)
    
    begin_time = 0
    routes = []
    for i in range(n_routefiles):
        routefile = "traffic.rou.xml" \
                    if (n_routefiles == 1) \
                    else f"traffic_{i}.rou.xml"
        if path is not None:
            routefile = join(path, routefile)
        opts = rrs.set_options(
            netfile=net_name, 
            routefile=routefile, 
            begin=begin_time,
            end=end_time, 
            length=True, 
            period=end_time/n_vehicles,
            generator=generator.lower(), 
            seed=seed,
            dir=path
        )
        routes.append(routefile)
        res = rrs.main(opts)

    return routes

if __name__ == "__main__":
    # Simple example of how to run the above function.
    generate_random_routes("traffic.net.xml", 100, 'uniform')
    