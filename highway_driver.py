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

def main():
   discount = GAMMA
   epochs = 200
   learning_rate = 0.01
   num_features = 6
   n_traj = 10
   trajectories = []
   max_timesteps = 10
   # env = ObservationWrapper(gym.make("highway-v0"))
   env = gym.make("highway-v0")

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
        "vehicles_count": 1,
        "screen_height": 150,
        "screen_width": 600,
    }
   env.configure(config)
   env.reset() 
   # trajectory = utils.record_trajectories(env, max_timesteps)
   # np.savetxt("demo_traj.txt", trajectory, fmt='%s')
   # highway_dqn.train_dqn(env)
   # highway_dqn.test_model(env)

   ############## RECORDING TRAJECTORIES ###################################
#    for i in range(n_traj):
#        env.seed(55)
#        env.reset()
#        trajectory = utils.record_trajectories(env, max_timesteps)
#        trajectories.append(trajectory)

# #    np.savetxt("trajs.txt", trajectories, fmt='%s')
#    utils.save_trajectories(trajectories, filename="same_lane_speeding.pickle")

   t = utils.load_trajectories("same_lane_speeding.pickle")

   feature_vector = np.zeros((num_features))
   # deep_irl.irl(env, t, feature_vector , action_space, epochs, discount, learning_rate)
   theta = maxent_highway.irl(env, t, feature_vector , action_space, epochs, discount, learning_rate)
   print(theta)

####################### ANALYSING REWARDS #################################
   theta = [0.66018253, -87.48911503, -44.98564138, -389.32662012, 42.51299763, 22.98094568]
   reward = []
   for traj in t:
    r = []
    for state in traj:
        feat = maxent_highway.feature_func(state[0])
        r.append(np.dot(feat, theta))
    reward.append(r)

   utils.get_reward_plot(reward)

#    env.reset()
#    obs, reward, done, _ = env.step(env.action_type.actions_indexes["IDLE"])
#    print(reward)

if __name__ == "__main__":
    main()