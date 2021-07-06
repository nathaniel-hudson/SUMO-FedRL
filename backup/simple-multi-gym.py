import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import traci

from collections import defaultdict
from seal.sumo.env import SumoEnv
from seal.sumo.kernel.kernel import SumoKernel
from seal.sumo.utils.random_routes import generate_random_routes
from os.path import join
from typing import Tuple

def main() -> pd.DataFrame:
    env = SumoEnv(config={
        "gui": False,
        "net-file": join("configs", "two_inter", "two_inter.net.xml"),
        "rand_route_args": {
            "n_vehicles": (100, 500),
            "end_time": 300
        }
    })

    tmp_data = defaultdict(list)
    total_reward = defaultdict(list)
    n_episodes = 1

    for ep in range(n_episodes):
        env.reset()
        done, step = False, 0
        last_reward = defaultdict(float)

        while not done:
            action = env.action_space.sample()
            obs, rewards, done, info = env.step(action)

            for tls_id, reward in rewards.items():
                total_reward["tls_id"].append(tls_id)
                total_reward["reward"].append(reward)
                total_reward["step"].append(step)
                total_reward["cumulative"].append(reward + last_reward[tls_id])
                last_reward[tls_id] += reward
            step += 1

    env.close()

    # Do a simple lineplot of the cumulative reward taken over time.
    sns.set_style("ticks")
    total_reward = pd.DataFrame.from_dict(total_reward)
    sns.lineplot(
        x="step", 
        y="cumulative", 
        style="tls_id", 
        hue="tls_id", 
        data=total_reward
    )
    plt.show()

    return total_reward

if __name__ == "__main__":
    main()