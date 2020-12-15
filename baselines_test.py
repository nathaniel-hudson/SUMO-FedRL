import gym
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict

from fluri.sumo.single_agent_env import SingleSumoEnv
from fluri.sumo.kernel.kernel import SumoKernel

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from os.path import join

data = defaultdict(list) # {"action": [], "step": [], "policy": [], "total_reward": []}

def add_record(action, step, policy, reward) -> None: 
    for i, a_i in enumerate(action):
        data["tls"].append(i)
        data["action"].append(a_i)
        data["step"].append(step)
        data["policy"].append(policy)
        data["reward"].append(reward)

if __name__ == "__main__":
    # Load the pre-trained model and initialize the environment.
    model = PPO.load("two_inter_model")
    env = SingleSumoEnv(config={
        "gui": True,
        "net-file": join("configs", "two_inter", "two_inter.net.xml")
    })

    # Run through an episode using a random policy.
    obs = env.reset()
    done, step, total_reward = False, 0, 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        add_record(info["taken_action"], step, "Random", total_reward)
        step += 1

    # Run through an episode using the trained PPO policy.
    obs = env.reset()
    done, step, total_reward = False, 0, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        add_record(info["taken_action"], step, "PPO", total_reward)
        step += 1
    env.close()
    
    # Plot the results
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    sns.lineplot(x="step", y="action", hue="policy", style="tls", ax=ax1, data=data)
    sns.lineplot(x="step", y="total_reward", hue="policy", ax=ax2, data=data)
    plt.show()