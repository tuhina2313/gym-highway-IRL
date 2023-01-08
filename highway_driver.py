from unicodedata import decimal
import numpy as np
import gym
import random

import utils
import deep_irl
import highway_dqn
import maxent_highway
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


GAMMA = 0.9

class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, rew):
        observe = self.observation_type.observe()
        ground_theta = [2.0, 0.0,  1.0, 0.0, 0.0, 0.03, 5.0, -3.0, -10.0]
        feat = utils.feature_func(observe)
        rew = np.dot(feat, ground_theta)
        return rew
        

def main():
   discount = GAMMA
   epochs = 200
   learning_rate = 0.01
   num_features = 9
   n_traj = 100
   trajectories = []
   # env = RewardWrapper(gym.make("highway-v0"))
   env = gym.make("highway-v0")

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
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": [ "x", "y", "vx", "vy", "heading" ],
            },
        "absolute": False,
        "lanes_count": 3,
        "show_trajectories": True,
        "manual_control": False,
        "real_time_rendering": False,
        "vehicles_count": 5,
        "screen_height": 150,
        "screen_width": 600,
        "order": sorted,
    }
   env.configure(config)
   env.reset() 

######################### TRAINING DQN ######################################


   # highway_dqn.train_dqn(env, filename = "highway_dqn/Vanilla_dqn")
   # highway_dqn.test_model(env, filename = "highway_dqn/Vanilla_dqn", max_timesteps= 15)
   # print("DQN Trained")

######################### RECORDING DQN TRAJECTORIES ######################################

   for i in range(n_traj):
       env.seed(50)
       env.reset()
       trajectory = highway_dqn.record_trajectories(env, model_file= "highway_dqn/Vanilla_dqn", max_timesteps = 15)
       trajectories.append(trajectory)
   print("Recorded Trajectories")
######################### SAVING DQN TRAJECTORIES ######################################
   utils.save_trajectories(trajectories, filename="newData/Expert_vanilla.pickle")

   t_ground = utils.load_trajectories("newData/Expert_vanilla.pickle")
   print("Loaded Trajectories")



   # trajectory = utils.record_trajectories(env, max_timesteps = 15)
#    print(trajectory)
   # np.savetxt("demo_traj.txt", trajectory, fmt='%s')

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

   feature_vector = np.zeros((num_features))
   # deep_irl.irl(env, t, feature_vector , action_space, epochs, discount, learning_rate)
   theta, expert_feat = maxent_highway.irl(env, t_ground, feature_vector , action_space, epochs, discount, learning_rate)
   print(theta)
   # theta_n = (theta-np.min(theta))/(np.max(theta)-np.min(theta))
   theta_nor = theta / np.sqrt(np.sum(theta**2))
   print(theta_nor)
   exp_theta = expert_feat / np.sqrt(np.sum(expert_feat**2))
   print(np.around(exp_theta, decimals = 2))

if __name__ == "__main__":
    main()