from email.mime import base
from cv2 import threshold
import numpy as np
import gym
import utils
import maxent_highway
import matplotlib.pyplot as plt
import highway_dqn

THRESHOLD = 0.75

class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, rew):
        observe = self.observation_type.observe()
      #   ground_theta = [2.0, 0.0,  1.0, 0.0, 0.0, 0.03, 5.0, -3.0, -10.0]
        ground_theta = [0.11, 0.0, 0.78, 0.0, 0.52, 0.0, 0.27, 0.0, 0.22]
        feat = utils.feature_func(observe)
        rew = np.dot(feat, ground_theta)
        return rew

env = RewardWrapper(gym.make("highway-v0"))
# env = Monitor(gym.make('highway-v0'), './video', force=True)

np.random.seed(50)
env.seed(50)

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
    "vehicles_count": 5,
    "screen_height": 150,
    "screen_width": 600,
}
env.configure(config)


def record_new_traj(n_traj):
   trajectories = []

   for _ in range(n_traj):
    np.random.seed(50)
    env.seed(50)
    env.reset() 
    trajectory = utils.record_trajectories(env, max_timesteps=15)
    trajectories.append(trajectory)
   return trajectories

def analyse_trajectory(traj):
    env = gym.make("highway-v0")

    for state in traj:
        action = state[1]
        #Sampling in environment 
        obs, reward, done, info = env.step(int(action))

def segregate_trajectories(t_pref, reward, threshold):
    index = 2
    segment = []
    traj1 = t_pref[index]
    optimal = t_pref[0]
    traj_R = reward[index]
    for state, i in traj1:
        while(traj_R[i] < threshold):
            segment.append(state)

    return segment

def analyse_R(t_pref):
    #Loading trajectories
    # t_ground = utils.load_trajectories("newData/Expert_vanilla.pickle")

    #Theta calculated from IRL
    # ground_theta = [0.13, 0.0,   0.02, 0.0,   0.06, 0.0,   0.20, 0.90, 1.0  ]
    ground_dqn = [0.1,  0.0,   0.1, 0.0,   0.05, 0.0,   0.46, 0.87, 1.0  ]
    # Record Preference trajectory 
    # trajectory = record_new_traj(4)
    # utils.save_trajectories(trajectory, filename="dqnData/final_test.pickle")

    test_run = []
    # test_run.append(t_ground[0])
    test_run.append(t_pref[0])
    test_run.append(t_pref[1])
    test_run.append(t_pref[2])
    test_run.append(t_pref[3])


    # Preparing Hybrid Trajectory
    # hybrid_traj = t_crash
    # test1 = hybrid_traj[0][0]
    # for i in range(5,len(t_ground[0])):
    #     hybrid_traj[i][0] = t_ground[0][i][0]

    # test_run.append(hybrid_traj)

    for state in test_run[3]:
        feat = maxent_highway.feature_func(state[0])

    reward = []
    for traj in test_run:
        r = []
        base_reward = 10
        r_cumulative = 0
        for state in traj:
            feat = maxent_highway.feature_func(state[0])  
            if feat[8] == -1:
                base_reward = 0
            r_cumulative = r_cumulative + np.dot(feat, ground_dqn)
            r.append(r_cumulative)
        print("Base reward: ", base_reward)
        r.append(r_cumulative+base_reward)
        reward.append(r)
    # utils.get_reward_plot(reward)
    x = np.arange(0, len(reward[0]), 1, dtype=int)
    plt.plot(list(x), list(reward[0]), color = "g")
    plt.plot(list(x), list(reward[1]), color = "b")
    plt.plot(list(x), list(reward[2]), color = "y")
    plt.plot(list(x), list(reward[3]), color = "r")

    segregate_trajectories(t_pref, reward, threshold= THRESHOLD)

    # plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    # plt.yticks(np.arange(min(reward[0]), max(reward[0])+1, 0.5))
    plt.show()

if __name__ == "__main__":
    # t_pref = utils.load_trajectories("test_data/final_test.pickle")
    # analyse_R(t_pref)
    highway_dqn.get_action_probabilities(env, filename = "highway_dqn/groundtruth", max_timesteps= 10)
    # analyse_trajectory(t_pref[2])