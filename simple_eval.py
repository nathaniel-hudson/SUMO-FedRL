import pickle
import ray
import seal.trainer.util as util

from seal.sumo.env import SumoEnv
from os.path import join
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy


# These are dummy model weights used with laughably small amounts of training. However
# these Pickle (.pkl) files contain the model weights of the policy that will be used
# for evaluation. We will set this policy's weights to one of these based on whether
# we want the ranked or unranked policy.
RANKED_WEIGHTS_PKL = join("example_weights", "FedRL",
                          "complex_inter", "ranked.pkl")
UNRANKED_WEIGHTS_PKL = join("example_weights", "FedRL",
                            "complex_inter", "unranked.pkl")


def get_netfile(code: str) -> str:
    """This is just a convenient function that returns the netfile path based on a code.
       Valid netfile codes include the following: 'complex', 'single', and 'two'.

    Args:
        code (str): The code corresponding to a valid netfile path

    Raises:
        ValueError: Occurs if an invalid code is provided

    Returns:
        str: Path to the netfile based on the passed-in code.
    """
    if code == "complex":
        return join("configs", "complex_inter", "complex_inter.net.xml")
    elif code == "single":
        return join("configs", "single_inter", "single_inter.net.xml")
    elif code == "two":
        return join("configs", "two_inter", "two_inter.net.xml")
    else:
        raise ValueError(
            "Parameter `code` must be in ['complex', 'single', 'two'].")


def load_policy(weights_pkl, env_config) -> PPOTrainer:
    """Return a Ray RlLib PPOTrainer class with the multiagent policy setup specifically
       for the netfile we are performing evaluation on. Further, this will apply the
       weights from prior training to the testing/evaluating policy network.

    Args:
        weights_pkl (str): Path to the weights Pickle file that is saved during training.
        env_config (Dict[str, Any]): Configuration to use for the environment class.

    Returns:
        PPOTrainer: Trainer object with the test policy network for evaluation.
    """
    temp_env = SumoEnv(env_config)
    tls_ids = [tls.id for tls in temp_env.kernel.tls_hub]
    multiagent = {
        "policies": {idx: (None, temp_env.observation_space, temp_env.action_space, {})
                     for idx in tls_ids + [util.GLOBAL_POLICY_VAR]},
        # This is NEEDED for evaluation.
        "policy_mapping_fn": util.eval_policy_mapping_fn
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
    with open(weights_pkl, "rb") as f:
        weights = pickle.load(f)
    policy.get_policy(util.GLOBAL_POLICY_VAR).set_weights(weights)
    return policy


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import defaultdict

    # Designate whether we wish to use the ranked policy and state space or not.
    ranked = True
    weights_pkl = RANKED_WEIGHTS_PKL if ranked \
        else UNRANKED_WEIGHTS_PKL

    # Initialize the dictionary object to record the evaluation data (i.e., `tls_rewards`)
    # and then begin the evaluation by looping over each of the netfiles.
    tls_rewards = defaultdict(list)
    for netfile in ["complex", "single", "two"]:
        ray.init()
        print(f">>> Performing evaluation using '{netfile}' net-file.")
        env_config = util.get_env_config(**{
            "gui": False,
            "net-file": get_netfile(netfile),
            "rand_routes_on_reset": True,
            "ranked": ranked
        })
        policy = load_policy(weights_pkl, env_config)
        env = SumoEnv(env_config)
        obs, done = env.reset(), False
        step = 1
        while not done:
            actions = {agent_id: policy.compute_action(agent_obs, policy_id=agent_id)
                       for agent_id, agent_obs in obs.items()}
            obs, rewards, dones, info = env.step(actions)
            for tls, r in rewards.items():
                tls_rewards["tls_id"].append(tls)
                tls_rewards["reward"].append(r)
                tls_rewards["netfile"].append(netfile)
                tls_rewards["step"].append(step)
            done = next(iter(dones.values()))
            step += 1
        env.close()
        ray.shutdown()

    # Plot the results.
    sns.displot(tls_rewards, kind="ecdf", col="netfile", x="reward")
    plt.show()
