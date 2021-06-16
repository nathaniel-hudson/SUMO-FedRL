import os
import time
import traci

"""
(
    array([0.02997602]), 
    array([0.]), 
    array([1.0149547]), 
    array([3.]), 
    array([1.22474487]), 
    array([0.]),
    array([0.66666667])
), 

Tuple(
    Box(0.0, 1.0, (1,), float32), 
    Box(0.0, 1.0, (1,), float32), 
    Box(0.0, 1.0, (1,), float32), 
    Box(0.0, 7.0, (1,), float32), 
    Box(0.0, 7.0, (1,), float32), 
    Box(0.0, 1.0, (1,), float32), 
    Box(0.0, 1.0, (1,), float32)
)
"""

if __name__ == "__main__":
    from fluri.sumo.kernel.trafficlight.space import trafficlight_space

    space = trafficlight_space()
    sample = space.sample()
    print(f"Unranked space:\n{space}")
    print(f"Unranked sample:\n{sample}\n")

    space = trafficlight_space(True)
    sample = space.sample()
    print(f"Ranked space:\n{space}")
    print(f"Ranked sample:\n{sample}")

    exit(0)

    netfile = os.path.join("configs", "complex_inter", "complex_inter.net.xml")
    cmd = f"sumo --no-warnings true --no-step-log --net-file {netfile}".split()
    traci.start(cmd)
    lanes = traci.lane.getIDList()
    print(f"Lane IDs: {lanes}\nNumber of lanes: {len(lanes)}\n")
    trafficlights = traci.trafficlight.getIDList()
    print(f"Traffic light IDs: {trafficlights}\n"
          f"Number of traffic lights: {len(trafficlights)}\n")
    
    print(f"Overlap between lane IDs and traffic light IDs: "
          f"{set(lanes).intersection(set(trafficlights))}\n")

    tls_id = "gneJ0"
    controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
    print(f"Trafficlight `{tls_id}` controls the following lanes:"
          f"\n{set(controlled_lanes)}\n")

    controlled_links = traci.trafficlight.getControlledLinks(tls_id)
    print(f"Trafficlight `{tls_id}` controls the following links:"
          f"\n{controlled_links}")
    print(f"Number of links controlled by `{tls_id}`: {len(controlled_links)}\n")

    phase = traci.trafficlight.getRedYellowGreenState(tls_id)
    print(f"Current phase for `{tls_id}`: {phase}\n")

    traci.close()