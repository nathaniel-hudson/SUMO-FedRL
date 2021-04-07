import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from collections import defaultdict
from fluri.sumo.single_agent_env import SinglePolicySumoEnv
from os.path import join

sns.set_style("ticks")


def main(n_episodes: int=1, use_gui: bool=False) -> None:
    """A simple example for how to setup a single-policy simulation with FLURI using 
       random actions.

    Args:
        n_episodes (int, optional): The number of episodes to run through. Defaults to 1.
        use_gui (bool, optional):  Show the GUI with SUMO if True. Defaults to False.
    """
    # STEP 1: Initialize the environment for the Single-Policy RL case.
    env = SinglePolicySumoEnv(config={
        "gui": use_gui,
        "net-file": join("configs", "two_inter", "two_inter.net.xml"),
        "rand_route_args": {
            "n_vehicles": (100, 500),
            "end_time": 300
        }
    })

    # STEP 2: Run through the simulation.
    eval_data = defaultdict(list)
    last_reward = defaultdict(float)
    for ep in range(n_episodes):
        env.reset()
        done, step = False, 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

            eval_data["cum_reward"].append(reward + last_reward[ep])
            eval_data["reward"].append(reward)
            eval_data["step"].append(step)
            eval_data["episode"].append(ep)
            last_reward[ep] = eval_data["cum_reward"][-1]
            
            step += 1
    env.close()
    
    # STEP 3: Plot the results.
    eval_data = pd.DataFrame.from_dict(eval_data)
    sns.lineplot(x="step", y="cum_reward", style="episode", data=eval_data)
    plt.grid(linestyle="--")
    plt.legend(fancybox=False, edgecolor="black")
    plt.show()


if __name__ == "__main__":
    main(n_episodes=5, use_gui=False)