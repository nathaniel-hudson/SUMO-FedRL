
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from fluri.sumo.single_agent_env import SingleSumoEnv
from fluri.sumo.sumo_sim import SumoSim
from os.path import join

sns.set_style("ticks")

"""
This simple running example demonstrates how to setup a configuration to run a full
training loop using the SingleSumoEnv environment with the SumoSim wrapper to simplify the
setup needed for SUMO and TraCI.

This is a very *simple* example. For meaningful training via reinforcement learning,
you would likely need more complex environments and routing scenarios for compelling
results for your agent(s).
"""

def main(n_episodes:int=10, gui:bool=True) -> None:
    # Execute the TraCI training loop.
    path = join("configs", "example")
    sim = SumoSim(config = {
        "gui": gui,
        "net-file": join(path, "traffic.net.xml"),
        "route-files": join(path, "traffic.rou.xml"),
        "additional-files": join(path, "traffic.det.xml"),
        "tripinfo-output": join(path, "tripinfo.xml")
    })

    env = SingleSumoEnv(sim, world_shape=(20, 20))

    print(f"Obs shape -> {env.get_obs_dims()}\n"
          f"Sim shape -> {env.get_sim_dims()}")
    print(f"Action shape -> {env.action_space.shape}\n"
          f"Observation shape -> {env.observation_space.shape}")

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
    main(n_episodes=10, gui=False)