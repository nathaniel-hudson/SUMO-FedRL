"""
For this document, we will setup a basic RL pipeline using our SingleSumoEnv environment.
The RL tool we will incorporate is `stablebaselines`.
"""

from fluri.sumo.single_agent_env import SingleSumoEnv
from fluri.sumo.sumo_sim import SumoSim
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
        "policy": []
    }

def add_record(action, step, policy) -> None:
    global data
    data["action"].append(action[0])
    data["step"].append(step)
    data["policy"].append(policy)

def train(config, total_timesteps: int=int(2e6)):
    init_data()
    sim = SumoSim(config=config)
    env = SingleSumoEnv(sim, world_shape=WORLD_SHAPE)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save("simple_model")
    env.close()

def main(total_timesteps: int=int(2e6)) -> None:
    path = join("configs", "example")
    config = {
        "gui": False,
        "net-file": join(path, "traffic.net.xml"),
        "route-files": join(path, "traffic.rou.xml"),
        "additional-files": join(path, "traffic.det.xml"),
        "tripinfo-output": join(path, "tripinfo.xml")
    }
    train(config, total_timesteps)

if __name__ == "__main__":
    main(total_timesteps=int(2e6))