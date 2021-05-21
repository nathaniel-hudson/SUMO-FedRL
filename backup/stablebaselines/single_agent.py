from os.path import join
from stable_baselines3 import PPO
from ...sumo.kernel.kernel import SumoKernel
from ...sumo.single_agent_env import SinglePolicySumoEnv


def train(total_timesteps: int=int(2e6)) -> None:
    path = join("configs", "example")
    config = {
        "gui": False,
        "net-file": join(path, "traffic.net.xml"),
        "route-files": join(path, "traffic.rou.xml"),
        "additional-files": join(path, "traffic.det.xml"),
        "tripinfo-output": join(path, "tripinfo.xml")
    }

    sim = SumoKernel(config=config)
    env = SinglePolicySumoEnv(sim)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save("simple_model")
    env.close()
