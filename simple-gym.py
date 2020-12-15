import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from collections import defaultdict
from fluri.sumo.single_agent_env import SingleSumoEnv
from fluri.sumo.kernel.kernel import SumoKernel
from fluri.sumo.utils.random_routes import generate_random_routes
from os.path import join

sns.set_style("ticks")
data = defaultdict(list)

def add_record(action, step, policy, reward) -> None: 
    for i, a_i in enumerate(action):
        data["tls"].append(i)
        data["action"].append(a_i)
        data["step"].append(step)
        data["policy"].append(policy)
        data["reward"].append(reward)

if __name__ == "__main__":
    n_episodes = 1
    env = SingleSumoEnv(config={
        "gui": True,
        "net-file": join("configs", "two_inter", "two_inter.net.xml"),
        "rand_route_args": {
            "n_vehicles": (100, 500),
            "end_time": 300
        }
    })

    for ep in range(n_episodes):
        env.reset()
        done, step, total_reward = False, 0, 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            add_record(info["taken_action"], step, "Random", total_reward)
            step += 1

    env.close()

    # Do a simple lineplot of the actions taken over time.
    data = pd.DataFrame.from_dict(data)
    sns.lineplot(x="step", y="action", hue="policy", style="tls", data=data)
    plt.show()
    