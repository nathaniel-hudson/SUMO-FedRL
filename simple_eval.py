import ray
import fluri.tester.config as test_config

from fluri.sumo.multi_agent_env import MultiPolicySumoEnv
from os.path import join
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy

EVAL_CONFIG = {
    "env_config": {}
}

ranked_policy_checkpoint = join(
    "out", "models", "FedRL", "complex_inter", "ranked_37", "checkpoint_100", "checkpoint-100"
)

unranked_policy_checkpoint = join(
    "out", "models", "FedRL", "complex_inter", "unranked_5", "checkpoint_100", "checkpoint-100"
)

# PROBLEM: I think this needs to be identical to how it was defined during the training.
# This is a problem in the case that we are exporting a policy for the test bed.
# env_config = {
#     "env_config": {
#         # Your arguments for your environment class go here.
#     },
#     "framework": "torch",
#     "in_evaluation": True,
#     "log_level": "ERROR",
#     "lr": 0.001,
#     "num_workers": 0,
#     "multiagent": {
#         "policies": {"gneJ0": (PPOTorchPolicy, )}
#     }
# }

def global_policy(key):
    return "GLOBAL"

# def load_policy(checkpoint):
#     ...

env_config = {
    "gui": False,
    "net-file": join("configs", "complex_inter", "complex_inter.net.xml"),
    "rand_routes_on_reset": True,
    "ranked": True,
}

ray.init()
env = MultiPolicySumoEnv(env_config)
tls_ids = [tls.id for tls in env.kernel.tls_hub]
multiagent = {
    "policies": {idx: (None, env.observation_space, env.action_space, {})
                 for idx in tls_ids},# + [test_config.GLOBAL_POLICY]},
    "policy_mapping_fn": lambda idx: idx
}
trainer = PPOTrainer(env=MultiPolicySumoEnv, config={
    "env_config": env_config,
    "framework": "torch",
    "in_evaluation": True,
    "log_level": "ERROR",
    "lr": 0.001,
    # "multiagent": {
    #     "policies": policies,
    #     "policy_mapping_fn": lambda agent_id: agent_id
    # },
    "multiagent": multiagent,
    #
    "num_gpus": 0,
    "num_workers": 0,
})
trainer.restore(ranked_policy_checkpoint)
trainer.config["multiagent"]["policy_mapping_fn"] = global_policy

print(">> IT WORKED!!")