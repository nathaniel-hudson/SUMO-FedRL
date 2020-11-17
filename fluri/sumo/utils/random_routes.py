from typing import List
from os.path import join

from . import random_routes_support as rrs

VALID_DISTRIBUTIONS = ["arcsine", "uniform"]

def generate_random_routes(
    net_name: str, 
    num_vehicles: str, 
    generator: str,
    num_routefiles: int=1,
    begin_time: int=0,
    end_time: int=3600,
    seed: float=None,
    dir: str=None,
) -> List[str]:
    """This function generates a *.rou.xml file for vehicles in the given road network.

    Parameters
    ----------
    net_name : str
        Filename of the SUMO *.net.xml file.
    num_vehicles : int
        Number of vehicles to be used in the simulation.
    generator : str
        A token that specifies the random distribution that will be used to assigning
        routes to vehicles.
    seed : float, optional
        A random seed to fix the generator distribution, by default None
    """
    assert num_vehicles > 0
    assert generator.lower() in VALID_DISTRIBUTIONS
    
    routes = []
    for i in range(num_routefiles):
        routefile = "traffic.rou.xml" if (num_routefiles == 1) \
                    else f"traffic_{i}.rou.xml"
        if dir is not None:
            routefile = join(dir, routefile)
        opts = rrs.set_options(netfile=net_name, 
                               routefile=routefile, 
                               begin=begin_time,
                               end=end_time, 
                               length=True, 
                               period=end_time/num_vehicles,
                               generator=generator.lower(), 
                               seed=seed,
                               dir=dir)
        routes.append(routefile)
        res = rrs.main(opts)
        # print(f"Routes generated -> {res}")

    return routes

if __name__ == "__main__":
    # Simple example of how to run the above function.
    generate_random_routes("traffic.net.xml", 100, 'uniform')
    