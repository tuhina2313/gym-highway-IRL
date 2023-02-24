from unicodedata import decimal
import numpy as np
import gym
import random

import utils
import highway_dqn
import maxent_highway
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

GAMMA = 0.9

class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, rew):
        observe = self.observation_type.observe()
        ground_theta = [2.0, 0.0,  1.0, 0.0, 0.0, 0.03, 5.0, -3.0, 100.0]
      #   ground_theta = [0.11, 0.0, 0.78, 0.0, 0.52, 0.0, 0.27, 0.0, 0.22]
        feat = utils.feature_func(observe)
        rew = np.dot(feat, ground_theta)
        return rew
        

class StepWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        if action == 1:
            action = random.choice(['0','1', '2'])
        if action == '3':
            action = random.choice(['3', '4'])
        obs, reward, terminated, info, _ = self.env.step(action)
        return obs, reward, terminated, info, action
    


def main():
   discount = GAMMA 
   epochs = 200
   learning_rate = 0.01
   num_features = 9
   n_traj = 100
   trajectories = []
   env = RewardWrapper(gym.make("highway-v0"))
   # env = gym.make("highway-v0")

   np.random.seed(55)
   env.seed(55)
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
        "manual_control": True,
        "real_time_rendering": True,
        "vehicles_count": 5,
        "screen_height": 150,
        "screen_width": 600,
        "order": sorted,
    }
   env.configure(config)
   obs = env.reset() 
######################### TEST OBS #####################################
   trajectory = utils.record_trajectories(env, max_timesteps = 15)
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


   # highway_dqn.train_dqn(env, filename = "highway_dqn/groundtruth_continuous")
   # highway_dqn.test_model(env, filename = "highway_dqn/groundtruth_continuous", max_timesteps= 30)
   # print("DQN Trained")

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
#    for i in range(n_traj):
#        env.seed(50)
#        env.reset()
#        trajectory = utils.record_trajectories(env, max_timesteps)
#        trajectories.append(trajectory)

   # buffer = []
   # for _ in range(100):
   #     trajectories.append(trajectory)

#    utils.save_trajectories(trajectories, filename="newData/Expert.pickle")

# #    t_crash = utils.load_trajectories("crash.pickle")
#    t_ground = utils.load_trajectories("newData/Expert.pickle")

########################## IRL ####################################################

   # feature_vector = np.zeros((num_features))
   # # deep_irl.irl(env, t, feature_vector , action_space, epochs, discount, learning_rate)
   # theta, expert_feat = maxent_highway.irl(env, t_crash, feature_vector , action_space, epochs, discount, learning_rate)
   # print(theta)
   # # theta_n = (theta-np.min(theta))/(np.max(theta)-np.min(theta))
   # theta_nor = theta / np.sqrt(np.sum(theta**2))
   # print("Normalised theta: ", theta_nor)
   # exp_theta = expert_feat / np.sqrt(np.sum(expert_feat**2))
   # print("Expert theta: ", np.around(exp_theta, decimals = 2))

if __name__ == "__main__":
    main()