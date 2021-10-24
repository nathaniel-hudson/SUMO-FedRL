import os
import pickle
import ray
import seal.trainer.util as util

from netfiles import *
from os.path import join
from pandas import DataFrame
from seal.sumo.config import *
from seal.sumo.env import SumoEnv
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from typing import Any, Dict


# The list of netfiles we wish to train on.
# NETFILES = [GRID_3x3, GRID_5x5, GRID_7x7, GRID_9x9]
# NETFILES = [GRID_9x9]
# NETFILES = [DOUBLE_LOOP]
NETFILES = {
    "grid_3x3": GRID_3x3,
    "grid_5x5": GRID_5x5,
    "double_loop": DOUBLE_LOOP
}

FEATURE_PAIRS = [
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

# These are dummy model weights used with laughably small amounts of training. However
# these Pickle (.pkl) files contain the model weights of the policy that will be used
# for evaluation. We will set this policy's weights to one of these based on whether
# we want the ranked or unranked policy.
# weights_path_base = ("example_weights", "new-state-space")
# RANKED_WEIGHTS_PKL = join("example_weights", "new-state-space", "FedRL",
#                           "complex_inter", "ranked.pkl")
# UNRANKED_WEIGHTS_PKL = join("example_weights", "new-state-space", "FedRL",
#                             "complex_inter", "unranked.pkl")


def load_policy(weights_pkl: str, env_config: Dict[str, Any]) -> PPOTrainer:
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

    for idx in tls_ids + [util.GLOBAL_POLICY_VAR]:
        policy.get_policy(idx).set_weights(weights)

    # policy.get_policy(util.GLOBAL_POLICY_VAR).set_weights(weights)
    return policy


def run_trial(
    netfile_path: str,
    ranked: bool,
    tls_rewards: Dict[str, Any],
    feature_data: Dict[str, Any],
    weights_path: str,
    use_policy: bool=True,
    gui: bool=False,
    mc_run: int=None
) -> None:
    ray.init(include_dashboard=False)
    env_config = util.get_env_config(**{
        "gui": gui,
        "net-file": netfile_path,
        "rand_routes_on_reset": True,
        "ranked": ranked,
        "horizon": 360
    })
    netfile = netfile_path.split(os.sep)[-1].split(".")[0]
    policy = load_policy(weights_path, env_config)
    policy_id = weights_path.split(os.sep)[-1]
    trainer = weights_path.split(os.sep)[2]
    trainer_intersection = weights_path.split(os.sep)[3]
    trainer_ranked = "ranked" if ranked else "unranked"
    env = SumoEnv(env_config)
    obs, done = env.reset(), False
    step = 1
    while not done:
        if use_policy:
            # NOTE: The `policy_id` argument matters here. If it's incorrect, you'll get
            #       random behavior.
            actions = {
                agent_id: policy.compute_action(agent_obs, policy_id=agent_id)
                for agent_id, agent_obs in obs.items()
            }
            obs, rewards, dones, info = env.step(actions)
        else:
            obs, rewards, dones, info = env.step(None)
        n_vehicles = env.kernel.get_num_of_vehicles()
        for tls, r in rewards.items():
            tls_rewards["tls_id"].append(tls)
            tls_rewards["reward"].append(r)
            tls_rewards["netfile"].append(netfile)
            tls_rewards["step"].append(step)
            tls_rewards["n_vehicles"].append(n_vehicles)
            tls_rewards["policy"].append(policy_id)

            tls_rewards["trainer"].append(trainer)
            tls_rewards["trainer_intersection"].append(trainer_intersection)
            tls_rewards["trainer_ranked"].append(trainer_ranked)
            if mc_run is not None:
                tls_rewards["mc_run"].append(mc_run)

        done = next(iter(dones.values()))
        step += 1
        for feature_label, feature_index in FEATURE_PAIRS:
            for tls in obs:
                if feature_index in obs[tls]:
                    feature_data["feature"].append(feature_label)
                    feature_data["value"].append(obs[tls][feature_index])
                    feature_data["netfile"].append(netfile)
                    feature_data["tls"].append(tls)
                    feature_data["policy"].append(policy_id)
                    feature_data["ranked"].append(ranked)

                    feature_data["trainer"].append(trainer)
                    feature_data["trainer_intersection"].append(
                        trainer_intersection)
                    feature_data["trainer_ranked"].append(trainer_ranked)
                    if mc_run is not None:
                        feature_data["mc_run"].append(mc_run)
    env.close()
    ray.shutdown()


# =========================================================================== #


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import defaultdict

    # Initialize the dictionary object to record the evaluation data (i.e.,
    # `tls_rewards`) and then begin the evaluation by looping over each of the netfiles.
    feature_data = defaultdict(list)
    tls_rewards = defaultdict(list)

    for trainer in ["FedRL", "MARL", "SARL"]:
        for trainer_intersection in ["double", "grid-3x3", "grid-5x5"]:
            for ranked in [True, False]:
                ranked_str = "ranked" if ranked else "unranked"
                if trainer == "FedRL":
                    filename = f"v3_pos-reward-aggr_{ranked_str}.pkl"
                else:
                    filename = f"v3_{ranked_str}.pkl"

                weights_path = join("example_weights", "Final", trainer,
                                    trainer_intersection, filename)
                for netfile_label, netfile_path in NETFILES.items():
                    print(f">>> Performing evaluation using '{netfile_label}' "
                          f"net-file ({ranked_str}).")
                    for mc_run in range(5):
                        run_trial(
                            netfile_path,
                            ranked,
                            tls_rewards=tls_rewards,
                            feature_data=feature_data,
                            weights_path=weights_path,
                            use_policy=True,
                            gui=False,
                            mc_run=mc_run
                        )

    # Plot the results.
    # sns.displot(data=feature_data, kind="ecdf", hue="netfile", x="value",
    #             col="feature", col_wrap=len(FEATURE_PAIRS)//2)
    # plt.show()

    # sns.displot(data=tls_rewards, kind="ecdf", hue="netfile", x="reward")
    # plt.show()

    # sns.relplot(data=tls_rewards, kind="line", hue="netfile",
    #             x="step", y="n_vehicles", ci=None)
    # plt.show()

    experiment_data_dir = join("out", "experiments", "digital")
    feature_df = DataFrame.from_dict(feature_data)
    feature_df.to_csv(join(experiment_data_dir, "features.csv"))
    reward_df = DataFrame.from_dict(tls_rewards)
    reward_df.to_csv(join(experiment_data_dir, "rewards.csv"))
