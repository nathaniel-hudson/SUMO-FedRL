"""
For this document, we will setup a basic RL pipeline using our SUMOGym environment.
The RL tool we will incorporate is `stablebaselines`.
"""

import gym
import matplotlib.pyplot as plt
import seaborn as sns

from fluri.sumo.sumo_gym import SUMOGym
from fluri.sumo.sumo_sim import SUMOSim
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from os.path import join

sns.set_style("ticks")

data = None

def init_data() -> None:
    global data
    data = {
        "action": [],
        "step": [],
        "policy": []
    }

def add_record(action, step, policy) -> None:
    global data
    data["action"].append(action)
    data["step"].append(step)
    data["policy"].append(policy)

def main() -> None:
    path = join("configs", "example")
    sim = SUMOSim(config = {
        "gui": False,
        "net-file": join(path, "traffic.net.xml"),
        "route-files": join(path, "traffic.rou.xml"),
        "additional-files": join(path, "traffic.det.xml"),
        "tripinfo-output": join(path, "tripinfo.xml")
    })
    
    init_data()
    env = SUMOGym(sim)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10)

    obs = env.reset()
    done, step = False, 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        add_record(info["taken_action"], step, "random")
        step += 1
        add_record(info["taken_action"], step, "random")
    env.close()

    obs = env.reset()
    done, step = False, 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        add_record(info["taken_action"], step, "RL")
        step += 1
        add_record(info["taken_action"], step, "RL")
    env.close()

    sns.lineplot(x="step", y="action", hue="policy", style="policy", data=data)
    plt.show()


if __name__ == "__main__":
    main()