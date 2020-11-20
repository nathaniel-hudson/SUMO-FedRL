import gym
import matplotlib.pyplot as plt
import seaborn as sns

from fluri.sumo.single_agent_env import SingleSumoEnv
from fluri.sumo.kernel.kernel import SumoKernel

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
    init_data()
    path = join("configs", "two_inter")
    env = SingleSumoEnv(config={
        "gui": False,
        "net-file": join(path, "two_inter.net.xml"),
        "tripinfo-output": join(path, "tripinfo.xml")
    })

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
    model = PPO.load("simple_model")
    main(model)