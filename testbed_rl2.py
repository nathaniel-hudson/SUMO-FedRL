#!/usr/bin/env python3
# license removed for brevity
import rospy
from mocap_optitrack.msg import Velocity_Arr
from std_msgs.msg import UInt16
from std_msgs.msg import String
import os
#libraries for RL
import numpy as np
import pickle
import ray
import seal.util as util
from seal.testbed.testbed_env import TestbedEnv
from os.path import join
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

RANKED_WEIGHTS_PKL = join("example_weights", "FedRL",
						  "complex_inter", "ranked.pkl")

ROAD_NETWORK_FILE = "defined_network.txt"

class TrafficManager(object):
	def __init__(self):
		self.info = None
		self.ctrl = 0

		self.loop_rate = rospy.Rate(10)

		#publisher
		self.pub = rospy.Publisher('/signal_states', UInt16, queue_size=10)

		#subscriber
		self.sub = rospy.Subscriber("/traffic_info", String, self.callback)

	def callback(self, msg):
		self.info = msg.data

	def get_network(self, file):
		tls_ids = []
		with open(file, 'r') as f:
			for data in f:
				tls_ids.append(data.split(' ')[0])
		return tls_ids

	def rl_agent(self):
		rospy.loginfo("agent says Hi!")
		policy = self.load_policy(weights_pkl, env_config)
		while not rospy.is_shutdown():
			if self.info != None:
				self.ctrl = 1
			self.pub.publish(self.ctrl)
			self.loop_rate.sleep()

	# def load_policy(self, weights_pkl, env_config) -> PPOTrainer:
	# 	"""Return a Ray RlLib PPOTrainer class with the multiagent policy setup specifically
	# 	   for the netfile we are performing evaluation on. Further, this will apply the
	# 	   weights from prior training to the testing/evaluating policy network.

	# 	Args:
	# 		weights_pkl (str): Path to the weights Pickle file that is saved during training.
	# 		env_config (Dict[str, Any]): Configuration to use for the environment class.

	# 	Returns:
	# 		PPOTrainer: Trainer object with the test policy network for evaluation.
	# 	"""
	# 	temp_env = TestbedEnv(env_config)
	# 	tls_ids = self.get_network(ROAD_NETWORK_FILE)
	# 	# tls_ids = [tls.id for tls in temp_env.kernel.tls_hub]
	# 	multiagent = {
	# 		"policies": {idx: (None, temp_env.observation_space, temp_env.action_space, {})
	# 					 for idx in tls_ids + [util.GLOBAL_POLICY_VAR]},
	# 		# This is NEEDED for evaluation.
	# 		"policy_mapping_fn": util.eval_policy_mapping_fn
	# 	}
	# 	policy = PPOTrainer(env=TestbedEnv, config={
	# 		"env_config": env_config,
	# 		"framework": "torch",
	# 		"in_evaluation": True,
	# 		"log_level": "ERROR",
	# 		"lr": 0.001,
	# 		"multiagent": multiagent,
	# 		"num_gpus": 0,
	# 		"num_workers": 0,
	# 	})
	# 	with open(weights_pkl, "rb") as f:
	# 		weights = pickle.load(f)
	# 	policy.get_policy(util.GLOBAL_POLICY_VAR).set_weights(weights)
	# 	return policy


if __name__ == '__main__':
	rospy.init_node("n", anonymous = True)
	pynode = TrafficManager()

	# Designate whether we wish to use the ranked policy and state space or not.
	ranked = True
	weights_pkl = RANKED_WEIGHTS_PKL if ranked else UNRANKED_WEIGHTS_PKL

	# # Initialize the dictionary object to record the evaluation data (i.e., `tls_rewards`)
	# # and then begin the evaluation by looping over each of the netfiles.
	tls_rewards = defaultdict(list)     #tls_rewrds is essentially a dictionary.

	ray.init()
	print("evaluation in testbed setup.")

	env_config = util.get_env_config(**{
	    "gui": False,
	    "net-file": None,
	    "rand_routes_on_reset": False,
	    "ranked": ranked
	})
	pynode.rl_agent()

	#     env = SumoEnv(env_config)
	#     obs, done = env.reset(), False
	#     step = 1
	#     while not done:
	#         actions = {agent_id: policy.compute_action(agent_obs, policy_id=agent_id)
	#                    for agent_id, agent_obs in obs.items()}
	#         obs, rewards, dones, info = env.step(actions)
	#         for tls, r in rewards.items():
	#             tls_rewards["tls_id"].append(tls)
	#             tls_rewards["reward"].append(r)
	#             tls_rewards["netfile"].append(netfile)
	#             tls_rewards["step"].append(step)
	#         done = next(iter(dones.values()))
	#         step += 1
	#     env.close()
	#     ray.shutdown()

	# # Plot the results.
	# sns.displot(tls_rewards, kind="ecdf", col="netfile", x="reward")
	# plt.show()