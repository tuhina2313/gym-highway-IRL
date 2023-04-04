from turtle import heading
from unicodedata import decimal
import numpy as np
import gym
import random
import collections

import utils
import highway_dqn
import maxent_highway
from fastdtw import fastdtw
from copy import deepcopy
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


GAMMA = 0.9

class RewardWrapper(gym.RewardWrapper):
   def __init__(self, env):
      super().__init__(env)
   
   def reward(self, rew):
      observe = self.observation_type.observe()
      ground_theta = [1.0, 0.0,  0.2, 0.0, 0.2, 0.03, 3.5, 0.5, 100.0]
      agg_theta = [0.08, 0.0,   0.11, 0.0,   0.0,   0.0,   0.4, 0.91,  0.0]
      rash_theta = [0.11, 0.0, 0.37, 0.0, 0.01, 0.0, 0.36, 0.85, 100.0]
      feat = utils.feature_func(observe)
      rew = np.dot(feat, ground_theta)
      return rew

   def set_state(self, state):
      print("State in Function: ", state)
      print("Observation on env: ",self.observation_type.observe())
      self.env = deepcopy(state)
      # obs = np.array(list(self.env.unwrapped.state))
      # return obs
      

class StepWrapper(gym.Wrapper):
   def __init__(self, env: gym.Env):
      super().__init__(env)

   def step(self, action):
   # ContinuousAction: Tuple of the form [throttle (acc), steering angle]
   # ContinuousAction: Between [-1, -1] and [1, 1]
      print(action) 
   #   if action == 1:
   #       action = random.choice(['0','1', '2'])
   #   if action == '3':
   #       action = random.choice(['3', '4'])
      obs, reward, terminated, info = self.env.step(action)
      return obs, reward, terminated, info, action
   
   


def main():

   discount = GAMMA 
   epochs = 200
   learning_rate = 0.01
   num_features = 9
   n_traj = 100

   trajectories = []
   env = RewardWrapper(gym.make("highway-v0"))
   # env = StepWrapper(env)
   # env = gym.make("highway-v0")

   np.random.seed(50)
   env.seed(50)
   action_space = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }

   config = {
       "action": {'type': 'DiscreteMetaAction'},
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": [ "x", "y", "vx", "vy", "heading" ],
            },
        "absolute": False,
        "lanes_count": 3,
        "show_trajectories": True,
        "manual_control": False,
        "real_time_rendering": True,
        "vehicles_count": 5,
        "screen_height": 150,
        "screen_width": 600,
        "order": sorted,
    }
   env.configure(config)
   env.reset() 


   ############## TESTING FOR STEP() WRAPPER #########################

   # obs1[0][0] = 10
   # print("First obs: ", obs1)
   # a_space = env.action_space.sample()
   # obs, reward, terminated, info = env.step(a_space)
   # print("Obs after taking action: ", obs)
   # obs2 = env.observation_type.observe()
   # print("Is it the same?: ", obs2)
   # env.set_state(obs1)
   # new_state = env.observation_type.observe()
   # print("Obs after running set_state(): ", new_state)

######################### TEST OBS #####################################
   # trajectory = utils.record_trajectories(env, max_timesteps = 30)
   # print(trajectory)

######################### EXPLICIT STOCHASTIC TRANSITIONS #####################################

   # result = highway_dqn.test_stochastic_transitions(env)
   # # utils.reward_bar_graphs(result)
   # state_rewards = np.array([res[1] for res in result ])
   # scaler = MinMaxScaler(feature_range=(-1, 1))
   # scaled = scaler.fit_transform([[x] for x in state_rewards])
   # print(scaled)   
   # x = np.arange(0, 12, 1, dtype=int)
   # plt.bar(x, list(scaled), align='edge', width=0.5, color='grey')
   # plt.tight_layout()
   # plt.show()
   # print(result)

######################### TRAINING DQN ######################################


   # highway_dqn.train_dqn(env, filename = "resultData/human_rash_noisy")
   # highway_dqn.test_model(env, filename = "resultData/human_rash_noisy", max_timesteps= 30)
   
######################### TESTING ROLLOUTS OF NOISY POLICY ###########################   
   # reward = highway_dqn.trajectory_rewards(env, model_file = "resultData/human_rash_noisy", n_episodes = 200)
   # utils.save_trajectories(reward, filename="resultData/human_rash_rollouts")
   # retrieved_R = utils.load_trajectories(filename="resultData/rollout_rewards")
   # retrieved_R = np.array(retrieved_R, dtype=float)
   # retrieved_R = retrieved_R[retrieved_R > 30]
   # discrete_R = np.around(retrieved_R, decimals = -1)
   # print("Rollouts Finished")
   # frequency = collections.Counter(discrete_R)
   # plt.bar(frequency.keys(), frequency.values(), 10.0, color='b')
   # plt.show()

   # trajectory, reward = highway_dqn.record_trajectories(env, model_file= "highway_dqn/Vanilla_dqn_noisy", max_timesteps = 30)
   # heading_traj = []
   # for ts in trajectory:
   #    obs_ts = ts[0]
   #    heading_t = obs_ts[0][4]
   #    heading_traj.append(heading_t)
   # min_h, max_h = min(heading_traj), max(heading_traj)
   # total_r = np.sum(retrieved_R)
   # print("DQN Tested")

######################### RECORDING DQN TRAJECTORIES ######################################

   # for i in range(n_traj):
   #     env.seed(50)
   #     env.reset()
   #     trajectory = highway_dqn.record_trajectories(env, model_file= "highway_dqn/Vanilla_dqn", max_timesteps = 15)
   #     trajectories.append(trajectory)
   # print("Recorded Trajectories")

######################### SAVING DQN TRAJECTORIES ######################################
   
   # utils.save_trajectories(trajectories, filename="newData/Expert_vanilla.pickle")

   # t_ground = utils.load_trajectories("newData/Expert_vanilla.pickle")
   # print("Loaded Trajectories")


########################## LOAD CASES ####################################################

   # t_cases = utils.load_trajectories("test_data/final_test.pickle")
   # t_crash =[]
   # for i in range(n_traj):
   #    t_crash.append(t_cases[1])
   # print("Loaded Trajectories")

   # trajectory = utils.record_trajectories(env, max_timesteps = 15)
   # trajectory = np.array(trajectory)
   # np.savetxt("demo_traj.txt", trajectory, fmt='%s')
   # print(trajectory)

   ############# RECORDING TRAJECTORIES ###################################

   trajectory = utils.record_trajectories(env, max_timesteps = 30)

   for i in range(n_traj):
       env.seed(50)
       env.reset()
       trajectories.append(trajectory)

   # buffer = []
   # for _ in range(100):
   #     trajectories.append(trajectory)

   utils.save_trajectories(trajectories, filename="resultData/Human_unreasonable.pickle")

   t_agg = utils.load_trajectories("resultData/Human_unreasonable.pickle")
#    t_ground = utils.load_trajectories("newData/Expert.pickle")

########################## IRL ####################################################

   feature_vector = np.zeros((num_features))
   # deep_irl.irl(env, t, feature_vector , action_space, epochs, discount, learning_rate)
   theta, expert_feat = maxent_highway.irl(env, t_agg, feature_vector , action_space, epochs, discount, learning_rate)
   print(theta)
   # theta_n = (theta-np.min(theta))/(np.max(theta)-np.min(theta))
   theta_nor = theta / np.sqrt(np.sum(theta**2))
   print("Normalised theta: ", theta_nor)
   exp_theta = expert_feat / np.sqrt(np.sum(expert_feat**2))
   print("Expert theta: ", np.around(exp_theta, decimals = 2))

if __name__ == "__main__":
    main()