import rospy
from mocap_optitrack.msg import Velocity_Arr
from mocap_optitrack.msg import Bot_Arr
from std_msgs.msg import UInt16
from std_msgs.msg import String
import os
#libraries for RL
import numpy as np
import pickle
import ray
import seal.util as util
from os.path import join
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class ROSManager(object):
	def __init__(self):
		# self.traffic_info = None	#removed, added subscriber['info'] instead..
		self.ctrl = 0
		self.loop_rate = rospy.Rate(10)

		#publishers
		# self.pub = rospy.Publisher('/signal_states', UInt16, queue_size=10)

		#subscriber
		self.sub = rospy.Subscriber("/light_sub", Bot_Arr, self.callback)

	def init_publishers(self, lights):
		#generate publihsers for each traffic light.
		self.publishers = {}	#collection of publishers, each corresponding to a traffic light addressed by the light_id.
		self.subscribers = {}
		self.subscribers['info'] = None
		static_name = '/light_'
		for light_id in lights:
			self.publishers[light_id] = {}
			# self.subscribers[light_id] = {}
			self.publishers[light_id]['pub'] = rospy.Publisher(str(static_name+str(light_id)), String, queue_size=10)
			# self.subscribers[light_id]['sub'] = rospy.Subscriber(static_name+"sub_"+str(light_id), Bot_Arr, self.callback, light_id)
			self.publishers[light_id]['status'] = 0
			# self.subscribers[light_id]['info'] = None
		print("publishers instantiated to send traffic light commands.")

	# def callback(self, msg, arg):
	# 	light_id = arg
	# 	self.subscribers[light_id]['info'] = msg.x
	def callback(self, msg):
		self.subscribers['info'] = msg.array

	def get_network(self, file):
		tls_ids = []
		with open(file, 'r') as f:
			for data in f:
				tls_ids.append(data.split(' ')[0])
		print("traffic lights in network: ", tls_ids)
		return tls_ids
	
	def get_status(self, light_id):
		return self.publishers[light_id]['status']

	def set_status(self, light_id, state, phase):
		self.publishers[light_id]['status'] = state
		print("setting status")
		self.publishers[light_id]['pub'].publish(phase)
	
	# def get_observations(self):
	# 	return self.traffic_info

	def get_observations(self):	#light_id
		# print("obs: ", self.subscribers['info'])
		return self.subscribers['info']