import os
import time
import traci


if __name__ == "__main__":
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