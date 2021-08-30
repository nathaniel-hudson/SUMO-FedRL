import pickle
import ray
import seal.trainer.util as util

from netfiles import *
from seal.sumo.config import *
from seal.sumo.env import SumoEnv
from os.path import join
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy


# The list of netfiles we wish to train on.
# NETFILES = [GRID_3x3, GRID_5x5, GRID_7x7, GRID_9x9]
# NETFILES = [GRID_9x9]
# NETFILES = [DOUBLE_LOOP]
NETFILES = {
    "grid_3x3": GRID_3x3,
    "grid_5x5": GRID_5x5,
    "double_loop": DOUBLE_LOOP
}

# These are dummy model weights used with laughably small amounts of training. However
# these Pickle (.pkl) files contain the model weights of the policy that will be used
# for evaluation. We will set this policy's weights to one of these based on whether
# we want the ranked or unranked policy.
RANKED_WEIGHTS_PKL = join("example_weights", "new-state-space", "FedRL",
                          "complex_inter", "ranked.pkl")
UNRANKED_WEIGHTS_PKL = join("example_weights", "new-state-space", "FedRL",
                            "complex_inter", "unranked.pkl")


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
        "policy_mapping_fn": util.eval_policy_mapping_fn
        # NOTE: The above "policy_mapping_fn" is NEEDED for evaluation.
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
        "explore": False,
    })
    with open(weights_pkl, "rb") as f:
        weights = pickle.load(f)
    policy.get_policy(util.GLOBAL_POLICY_VAR).set_weights(weights)
    return policy

# =========================================================================== #


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import defaultdict

    # Designate whether we wish to use the ranked policy and state space or not.
    ranked = True
    weights_pkl = RANKED_WEIGHTS_PKL if ranked \
        else UNRANKED_WEIGHTS_PKL

    # Initialize the dictionary object to record the evaluation data (i.e.,
    # `tls_rewards`) and then begin the evaluation by looping over each of the netfiles.
    tls_rewards = defaultdict(list)
    feature_pairs = [
        ("lane_occupancy", LANE_OCCUPANCY),
        ("halted_occupancy", HALTED_LANE_OCCUPANCY),
        ("speed_ratio", SPEED_RATIO),
        ("phase_state_r", PHASE_STATE_r),
        ("phase_state_y", PHASE_STATE_y),
        ("phase_state_g", PHASE_STATE_g),
        ("phase_state_G", PHASE_STATE_G),
        ("phase_state_u", PHASE_STATE_u),
        ("phase_state_o", PHASE_STATE_o),
        ("phase_state_O", PHASE_STATE_O),
        ("local_rank", LOCAL_RANK),
        ("global_rank", GLOBAL_RANK),
        ("local_halt_rank", LOCAL_HALT_RANK),
        ("global_halt_rank", GLOBAL_HALT_RANK)
    ]
    # state_features = {
    #     "lane_occupancy": [],
    #     "halted_occupancy": [],
    #     "speed_ratio": [],
    #     "phase_state_r": [],
    #     "phase_state_y": [],
    #     "phase_state_g": [],
    #     "phase_state_G": [],
    #     "phase_state_u": [],
    #     "phase_state_o": [],
    #     "phase_state_O": [],
    #     "local_rank": [],
    #     "global_rank": [],
    #     "local_halt_rank": [],
    #     "global_halt_rank": []
    # }
    feature_data = defaultdict(list)
    for netfile_label, netfile_path in NETFILES.items():
        ray.init()
        print(f">>> Performing evaluation using '{netfile_label}' net-file.")
        env_config = util.get_env_config(**{
            "gui": False,
            "net-file": netfile_path,
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
            # obs, rewards, dones, info = env.step(actions)
            obs, rewards, dones, info = env.step(None)
            n_vehicles = env.kernel.get_num_of_vehicles()
            for tls, r in rewards.items():
                tls_rewards["tls_id"].append(tls)
                tls_rewards["reward"].append(r)
                tls_rewards["netfile"].append(netfile_label)
                tls_rewards["step"].append(step)
                tls_rewards["n_vehicles"].append(n_vehicles)
            done = next(iter(dones.values()))
            step += 1
            for feature_label, feature_index in feature_pairs:
                for tls in obs:
                    feature_data["feature"].append(feature_label)
                    feature_data["value"].append(obs[tls][feature_index])
                    feature_data["netfile"].append(netfile_label)
                    feature_data["tls"].append(tls)
        env.close()
        ray.shutdown()

    # Plot the results.
    sns.displot(data=feature_data, kind="ecdf", hue="netfile", x="value",
                col="feature", col_wrap=len(feature_pairs)//2)
    plt.show()

    sns.displot(data=tls_rewards, kind="ecdf", hue="netfile", x="reward")
    plt.show()

    sns.relplot(data=tls_rewards, kind="line", hue="netfile",
                x="step", y="n_vehicles", ci=None)
    plt.show()
