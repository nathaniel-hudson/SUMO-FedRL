
import matplotlib.pyplot as plt

from fluri.sumo.sumo_gym import SUMOGym
from fluri.sumo.sumo_sim import SUMOSim
from os.path import join

"""
This simple running example demonstrates how to setup a configuration to run a full
training loop using the SUMOGym environment with the SUMOSim wrapper to simplify the
setup needed for SUMO and TraCI.

This is a very *simple* example. For meaningful training via reinforcement learning,
you would likely need more complex environments and routing scenarios for compelling
results for your agent(s).
"""

def main() -> None:
    # Execute the TraCI training loop.
    path = join("configs", "example")
    sim = SUMOSim(config = {
        "gui": False,
        "net-file": join(path, "traffic.net.xml"),
        "route-files": join(path, "traffic.rou.xml"),
        "additional-files": join(path, "traffic.det.xml"),
        "tripinfo-output": join(path, "tripinfo.xml")
    })

    env = SUMOGym(sim)
    data = {
        "actions": [],
        "steps": []
    }

    def add_record(y1, y2) -> None: 
        data["actions"].append(y1)
        data["steps"].append(y2)

    env.reset()
    done, step = False, 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        add_record(info["taken_action"], step)
        step += 1
        add_record(info["taken_action"], step)

    sim.close()

    plt.plot("steps", "actions", data=data)
    plt.ylabel("Agent Actions")
    plt.xlabel("Steps")
    plt.show()

if __name__ == "__main__":
    main()