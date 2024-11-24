#!/usr/bin/env python


from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import random

from collections import defaultdict
from sumolib import checkBinary  # noqa

import traci  # noqa
# import sumolib ## NOTE: Using `sumolib` causes issues with Traci and running SUMO.


def generate_routefile(): 
    N = 28  

    with open("traffic.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="typeWE" accel="0.8" decel="4.5" sigma="0.5" length="2.85" minGap="1" maxSpeed="30" guiShape="passenger"/>
        <vType id="typeNS" accel="0.8" decel="5" sigma="0.5" length="2.7" minGap="1" maxSpeed="30" guiShape="emergency"/>
        <vType id="typeSN1" accel="0.8" decel="4.5" sigma="0.5" length="2.7" minGap="1" maxSpeed="30" guiShape="bus"/>
        <vType id="typeSE" accel="0.8" decel="4.5" sigma="0.5" length="2.7" minGap="1" maxSpeed="30" guiShape="bus"/>
        <vType id="typeNW" accel="0.8" decel="4.5" sigma="0.5" length="2.7" minGap="1" maxSpeed="30" guiShape="emergency"/>
        
        <route id="right" edges="51o 1i 2o 52i" />
        <route id="left" edges="52o 2i 1o 51i" />
        <route id="down" edges="54o 4i 3o 53i" />
        <route id="up" edges="53o 3i 4o 54i" />
        <route id="upleft" edges="54o 4i 1o 51i" />
        <route id="southright" edges="53o 3i 2o 52i" />""", file=routes)
        vehNr = 0
        for i in range(N):
            if random.uniform(0, 1) < 0.2:
                print('    <vehicle id="right_%i" type="typeWE" route="right" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < 0.2:
                print('    <vehicle id="left_%i" type="typeWE" route="left" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < 0.2:
                print('    <vehicle id="southright_%i" type="typeSE" route="southright" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < 0.2:
                print('    <vehicle id="upleft_%i" type="typeNW" route="upleft" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < 0.2:
                print('    <vehicle id="down_%i" type="typeNS" route="down" depart="%i" color="1,0,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < 0.1:
                print('    <vehicle id="up_%i" type="typeSN1" route="up" depart="%i"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
        print("</routes>", file=routes)


def run(net):
    """execute the TraCI control loop"""
    step = 0

    wrapper = traci
    # we start with phase 2 where EW has green
    wrapper.trafficlight.setPhase("0", 2)

    vehicle_traversals = defaultdict(list)

    trafficlights = list(traci.trafficlight.getIDList())
    # print(f"List of traffic light IDs: {trafficlights}")
    for light_id in trafficlights:
        lanes = set(traci.trafficlight.getControlledLanes(light_id))
        # x, y = net.getNode(light_id).getCoord()
        print(f"\t> traffliclight({light_id}) controls: {lanes}")

    while traci.simulation.getMinExpectedNumber() > 0:
        # print(f"List of vehicle IDs: {list(traci.vehicle.getIDList())}")
        traci.simulationStep()
        # print(f"Step {step}.")
        for veh_id in list(traci.vehicle.getIDList()):
            x, y = traci.vehicle.getPosition(veh_id)
            vehicle_traversals[veh_id].append((x,y))

        if traci.trafficlight.getPhase("0") == 2:
            if traci.inductionloop.getLastStepVehicleNumber("0") > 0:
                # there is a vehicle from the north, switch
                traci.trafficlight.setPhase("0", 3)
            else:
                # otherwise try to keep green for EW
                traci.trafficlight.setPhase("0", 2)
        step += 1


    print(traci.simulation.getNetBoundary())
    traci.close()
    sys.stdout.flush()
    
    # print(vehicle_traversals)


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":

    options = get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    if "D" in sumoBinary:
       sumoBinary = sumoBinary.split("D")[0]
    # first, generate the route file for this simulation
    generate_routefile()

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-c", "traffic.sumocfg",
                             "--tripinfo-output", "tripinfo.xml"])
    
    # net = sumolib.net.readNet("traffic.net.xml")
    run(None)
