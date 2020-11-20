
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from fluri.sumo.single_agent_env import SingleSumoEnv
from fluri.sumo.kernel.kernel import SumoKernel
from fluri.sumo.utils.random_routes import generate_random_routes
from os.path import join

sns.set_style("ticks")

"""
This simple running example demonstrates how to setup a configuration to run a full
training loop using the SingleSumoEnv environment with the SumoKernel wrapper to simplify 
the setup needed for SUMO and TraCI.

This is a very *simple* example. For meaningful training via reinforcement learning,
you would likely need more complex environments and routing scenarios for compelling
results for your agent(s).
"""

def main(
    n_episodes: int, 
    n_vehicles: int, 
    gui: bool
) -> None:
    path = join("configs", "two_inter")
    netfile = join(path, "two_inter.net.xml")
    env = SingleSumoEnv(config={
        "gui": gui,
        "net-file": netfile,
        "rand_route_args": {
            "n_vehicles": 1000,
            "end_time": 300
        }
    })

    data = {
        "actions": [],
        "steps": [],
        "ep": [],
    }

    def add_record(action, step, ep_id) -> None: 
        data["actions"].append(action[0])
        data["steps"].append(step)
        data["ep"].append(ep_id)

    for ep in range(n_episodes):
        print(f">> Episode number ({ep+1}/{n_episodes})")
        env.reset()
        done, step = False, 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            add_record(info["taken_action"], step, ep)
            step += 1
            add_record(info["taken_action"], step, ep)

    env.close()

    data = pd.DataFrame.from_dict(data)
    sns.jointplot(x="steps", y="actions", kind="kde", data=data)
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=1, help="Number of episodes.")
    parser.add_argument("--n_vehicles", type=int, default=7500, help="Number of vehicles.")
    parser.add_argument("--gui", dest='gui', action="store_true")
    parser.add_argument("--no-gui", dest='gui', action="store_false")
    parser.set_defaults(gui=True)
    args = parser.parse_args()
    
    main(n_episodes=args.n_episodes, n_vehicles=args.n_vehicles, gui=args.gui)
