import os
import pickle
import ray
import seal.trainer.util as util

from netfiles import *
from os.path import join
from pandas import DataFrame
from seal import TRIPINFO_OUT_FILENAME
from seal.logging import *
from seal.sumo.config import *
from seal.sumo.env import SumoEnv
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from typing import Any, Dict, Optional
import xml.etree.ElementTree as ET
import time
from collections import defaultdict


# The list of netfiles we wish to train on.
# ICCPS_NETFILES = {
#     "grid_3x3": GRID_3x3,
#     "grid_5x5": GRID_5x5,
#     "grid_7x7": DOUBLE_LOOP
# }

NETFILES = {
    "grid_3x3": GRID_3x3,
    "grid_5x5": GRID_5x5,
    "grid_7x7": GRID_7x7
}

ICCPS_WEIGHTS_PATH_PREFIX = ("example_weights", "ICCPS", "Final")
SMARTCOMP_PATH_PREFIX = ("out", "SMARTCOMP", "weights")
WEIGHTS_PATH_PREFIX = SMARTCOMP_PATH_PREFIX
# WEIGHTS_PATH_PREFIX = SMARTCOMP_PATH_PREFIX  # alias

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

    return policy


def run_trial(
        netfile_path: str,
        ranked: bool,
        trainer: str,
        tls_rewards: Dict[str, Any],
        feature_data: Dict[str, Any],
        tripinfo_data: Dict[str, Any],
        weights_path: str,
        gui: bool = False,
        mc_run: int = None
) -> None:
    env_config = util.get_env_config(**{
        "gui": gui,
        "net-file": netfile_path,
        "rand_routes_on_reset": True,
        "ranked": ranked,
        # "horizon": 3#60  # NOTE: Lookie here!!!
    })
    netfile = netfile_path.split(os.sep)[-1].split(".")[0]

    if weights_path is not None:
        policy = load_policy(weights_path, env_config)
        policy_id = weights_path.split(os.sep)[-1]
        intersection = weights_path.split(os.sep)[-2]
        use_policy = True
    else:
        policy = None
        policy_id = "Timed-Phase"
        intersection = None
        use_policy = False

    trainer_ranked = "ranked" if ranked else "unranked"
    env = SumoEnv(env_config)
    obs, done = env.reset(), False
    step = 1
    trainer = "Timed-Phase" if trainer is None else trainer
    while not done:
        
        if use_policy:
            # Must have the `policy_id` argument, otherwise you'll get random behavior.
            actions = {agent_id: policy.compute_action(agent_obs, policy_id=agent_id)
                       for agent_id, agent_obs in obs.items()}
        else:
            actions = None
        
        obs, rewards, dones, _ = env.step(actions)
        n_vehicles = env.kernel.get_num_of_vehicles()
        for tls, r in rewards.items():
            tls_rewards["tls_id"].append(tls)
            tls_rewards["reward"].append(r)
            tls_rewards["netfile"].append(netfile)
            tls_rewards["step"].append(step)
            tls_rewards["n_vehicles"].append(n_vehicles)
            tls_rewards["policy"].append(policy_id)
            tls_rewards["trainer"].append(trainer)
            tls_rewards["trainer_intersection"].append(intersection)
            tls_rewards["trainer_ranked"].append(trainer_ranked)
            if mc_run is not None:
                tls_rewards["mc_run"].append(mc_run)

        step += 1
        done = next(iter(dones.values()))
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
                    feature_data["trainer_intersection"].append(intersection)
                    feature_data["trainer_ranked"].append(trainer_ranked)
                    if mc_run is not None:
                        feature_data["mc_run"].append(mc_run)
    env.close()

    # Open the `tripinfo.xml` file to get aggregate trip data.
    tree = ET.parse(TRIPINFO_OUT_FILENAME)
    root = tree.getroot()
    for trip in root.findall("tripinfo"):
        tripinfo_data["trainer"].append(trainer)
        tripinfo_data["trainer_intersection"].append(intersection)
        tripinfo_data["netfile"].append(netfile)
        tripinfo_data["trainer_ranked"].append(trainer_ranked)
        tripinfo_data["depart_delay"].append(trip.get("departDelay"))
        tripinfo_data["travel_time"].append(trip.get("duration"))
        tripinfo_data["route_length"].append(trip.get("routeLength"))
        tripinfo_data["waiting_time"].append(trip.get("waitingTime"))
        tripinfo_data["waiting_count"].append(trip.get("waitingCount"))
        tripinfo_data["reroute_number"].append(trip.get("rerouteNo"))
        if mc_run is not None:
            tripinfo_data["mc_run"].append(mc_run)


# =========================================================================== #

def get_filename(trainer: Optional[str], ranked: bool) -> str:
    version = "v4"
    ranked_str = "ranked" if ranked else "unranked"
    if trainer == "Timed-Phase":
        return None
    elif trainer == "FedRL":
        aggr = "naive-aggr"
        return f"{version}_{aggr}_{ranked_str}.pkl"
    elif trainer == "MARL" or trainer == "SARL":
        return f"{version}_{ranked_str}.pkl"
    else:
        raise ValueError("Invalid trainer value for `get_filename()`.")


def get_weight_path(
        trainer: str, 
        trainer_intersection: str, 
        filename: str
) -> Optional[str]:
    if trainer == "Timed-Phase":
        return None
    elif trainer in ["FedRL", "MARL", "SARL"]:
        return join(*WEIGHTS_PATH_PREFIX, trainer, trainer_intersection, filename)
    else:
        raise ValueError("Invalid trainer value for `get_weight_path()`.")

# =========================================================================== #


if __name__ == "__main__":
    # Initialize the dictionary object to record the evaluation data (i.e.,
    # `tls_rewards`) and then begin the evaluation by looping over each of the netfiles.
    NUM_MC_RUNS = 10
    feature_data = defaultdict(list)
    tls_rewards = defaultdict(list)
    tripinfo_data = defaultdict(list)
    run_start_template = "Eval Trial ({}/{}) | trainer='{}::{}::{}' | netfile='{}'"
    times = []

    ray.init(include_dashboard=False)
    for trainer in ["FedRL", "MARL", "SARL", "Timed-Phase"]:
        for trainer_intersection in ["grid-3x3", "grid-5x5", "grid-7x7"]:
            for ranked in [False]: # [True, False]:
                ranked_str = "ranked" if ranked else "unranked"
                filename = get_filename(trainer, ranked)
                weights_path = get_weight_path(trainer, trainer_intersection, filename)
                for (netfile_label, netfile_path) in NETFILES.items():
                    for mc_run in range(NUM_MC_RUNS):
                        logging.info(run_start_template.format(
                            mc_run+1, NUM_MC_RUNS,
                            trainer, trainer_intersection, ranked_str,
                            netfile_label 
                        ))
                        start = time.perf_counter()
                        run_trial(netfile_path,
                                  ranked,
                                  trainer=trainer,
                                  tls_rewards=tls_rewards,
                                  feature_data=feature_data,
                                  tripinfo_data=tripinfo_data,
                                  weights_path=weights_path,
                                  gui=False,
                                  mc_run=mc_run)
                        runtime = time.perf_counter() - start
                        times.append(runtime)

    ray.shutdown()
    end = time.perf_counter()

    print("\n\n")
    logging.info(f"Done! Experiment trials took {sum(times) / len(times)} "
                 "seconds on average.")

    experiment_data_dir = join("out", "experiments", "smartcomp-digital")
    feature_df = DataFrame.from_dict(feature_data)
    feature_df.to_csv(join(experiment_data_dir, "features.csv"))
    reward_df = DataFrame.from_dict(tls_rewards)
    reward_df.to_csv(join(experiment_data_dir, "rewards.csv"))
    tripinfo_df = DataFrame.from_dict(tripinfo_data)
    tripinfo_df.to_csv(join(experiment_data_dir, "tripinfo.csv"))
