import gym
import matplotlib.pyplot as plt
import seaborn as sns

from fluri.sumo.single_agent_env import SingleSumoEnv
from fluri.sumo.sumo_sim import SumoKernel

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from os.path import join

WORLD_SHAPE = (25, 25)
data = None

def init_data() -> None:
    global data
    data = {
        "action": [],
        "step": [],
        "policy": [],
        "total_reward": [],
    }

def add_record(action, step, policy, total_reward) -> None:
    global data
    data["action"].append(action[0])
    data["step"].append(step)
    data["policy"].append(policy)
    data["total_reward"].append(total_reward)

def main(model) -> None:
    path = join("configs", "example")
    sim = SumoKernel(config={
        "gui": True,
        "net-file": join(path, "traffic.net.xml"),
        "route-files": join(path, "traffic.rou.xml"),
        "additional-files": join(path, "traffic.det.xml"),
        "tripinfo-output": join(path, "tripinfo.xml")
    })
    
    init_data()
    env = SingleSumoEnv(sim, world_dim=WORLD_SHAPE)

    ## Random Policy.
    obs = env.reset()
    done, step, total_reward = False, 0, 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        add_record(info["taken_action"], step, "random", total_reward)
        step += 1

    ## RL Policy.
    obs = env.reset()
    done, step, total_reward = False, 0, 0
    while not done:
        mask = []
        action, _states = model.predict(obs, deterministic=True)
        # print(f"Action #{step} -> {action}\n")
        obs, reward, done, info = env.step(action)
        total_reward += reward
        add_record(info["taken_action"], step, "RL", total_reward)
        step += 1
    env.close()
    
    sns.lineplot(x="step", y="action", hue="policy", style="policy", data=data)
    plt.show()

    sns.lineplot(x="step", y="total_reward", hue="policy", style="policy", data=data)
    plt.show()

if __name__ == "__main__":
    # Set up the configuration for the environment and load the pre-trained model.
    path = join("configs", "example")
    config = {
        "gui": False,
        "net-file": join(path, "traffic.net.xml"),
        "route-files": join(path, "traffic.rou.xml"),
        "additional-files": join(path, "traffic.det.xml"),
        "tripinfo-output": join(path, "tripinfo.xml")
    }
    sim = SumoKernel(config=config)
    env = SingleSumoEnv(sim, grid_shape=WORLD_SHAPE)
    model = PPO.load("baseline_model")

    # Run the simple evaluation.
    main(model)