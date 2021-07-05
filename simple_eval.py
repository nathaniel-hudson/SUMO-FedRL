import ray
import fluri.tester.config as test_config

from fluri.sumo.env import SumoEnv
from fluri.trainer.util import GLOBAL_POLICY_VAR, get_env_config, eval_policy_mapping_fn
from os.path import join
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy

# TODO: Finish implementing this so that we can give it to Pratham as an example.

RANKED_POLICY_CHECKPOINT = join("out", "models", "FedRL", "complex_inter", "ranked_38", 
                                "checkpoint_000001", "checkpoint-1")
UNRANKED_POLICY_CHECKPOINT = join("out", "models", "FedRL", "complex_inter", 
                                  "unranked_6", "checkpoint_000001", "checkpoint-1")

def get_netfile(code: str) -> str:
    if code == "complex":
        return join("configs", "complex_inter", "complex_inter.net.xml")
    elif code == "single":
        return join("configs", "single_inter", "single_inter.net.xml")
    elif code == "two":
        return join("configs", "two_inter", "two_inter.net.xml")
    else:
        raise ValueError("Parameter `code` must be in ['complex', 'single', 'two'].")

def load_policy(checkpoint, env_config) -> PPOTrainer:
    temp_env = SumoEnv(env_config)
    tls_ids = [tls.id for tls in temp_env.kernel.tls_hub]
    multiagent = {
        "policies": {idx: (None, temp_env.observation_space, temp_env.action_space, {})
                    for idx in tls_ids + [GLOBAL_POLICY_VAR]},
        "policy_mapping_fn": eval_policy_mapping_fn  # This is NEEDED for evaluation.
    }
    policy = PPOTrainer(env=SumoEnv, config={
        "env_config": env_config,
        "framework": "torch",
        "in_evaluation": True,
        "log_level": "ERROR",
        "lr": 0.001,
        "multiagent": multiagent,
        "num_gpus": 0,
        "num_workers": 0,
    })
    policy.restore(checkpoint) 
    # ^^ Replace this with maybe a short script where we just load in the weights into
    #    `GLOBAL_POLICY_VAR` policy. We can save these weights as an .h5 file and then
    #    go from there. That way we just ignore the rest of the checkpoint, which 
    #    contains extra data we don't care about.
    return policy

# mapped_policy = trainer.config["multiagent"]["policy_mapping_fn"]("dummy")
# trainer.config["multiagent"]["policy_mapping_fn"] = global_policy

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import defaultdict

    ranked = True
    checkpoint = RANKED_POLICY_CHECKPOINT if ranked \
                 else UNRANKED_POLICY_CHECKPOINT
    
    for netfile in ["complex", "single", "two"]:
        ray.init()
        print(f">>> Performing evaluation using '{netfile}' net-file.")
        env_config = get_env_config(**{
            "gui": False, 
            "net-file": get_netfile(netfile), 
            "rand_routes_on_reset": True, 
            "ranked": ranked
        })
        policy = load_policy(checkpoint, env_config)

        env = SumoEnv(env_config)
        obs, tls_rewards, done = env.reset(), defaultdict(list), False
        while not done:
            actions = {agent_id: policy.compute_action(agent_obs, policy_id=agent_id)
                    for agent_id, agent_obs in obs.items()}
            obs, rewards, dones, info = env.step(actions)
            for tls, r in rewards.items():
                tls_rewards["tls_id"].append(tls)
                tls_rewards["reward"].append(r)
                tls_rewards["netfile"].append(netfile)
            done = next(iter(dones.values()))
        env.close()
        ray.shutdown()

    sns.displot(tls_rewards, kind="ecdf", col="netfile", x="reward", hue="tls_id")
    plt.show()
