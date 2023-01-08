from turtle import distance
import numpy as np
import gym
from typing import TypeVar
# from maxent_highway import feature_func, N_LANE

import utils
import math
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import similaritymeasures
from stable_baselines3 import DQN
from highway_dqn import train_dqn

N_LANE = 3


def dtw(s, t):
    n, m = len(s), len(t)
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = math.sqrt((s[i-1][0] - t[j-1][0])**2 + (s[i-1][1] - t[j-1][1])**2)
            # take last min from a square box
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix


def frechet_distance(trajectories, method="Frechet"):
    """
    :param method: "Frechet" or "Area"
    """
    n = len(trajectories)
    dist_m = np.zeros((n, n))
    for i in range(n - 1):
        p = trajectories[i]
        for j in range(i + 1, n):
            q = trajectories[j]
            if method == "Frechet":
                dist_m[i, j] = similaritymeasures.frechet_dist(p, q)
            else:
                dist_m[i, j] = similaritymeasures.area_between_two_curves(p, q)
            dist_m[j, i] = dist_m[i, j]
    return dist_m

def get_reward_plot(r, colr):
    
    x = np.arange(0, len(r), 1, dtype=int)

    plt.plot(list(x), list(r), color = colr, label = "Trajectory 1")

    plt.xlabel("Timesteps")
    plt.ylabel("Reward R(s)")

    plt.show()

def feature_func(state):
# f_distance: Distance from the other vehicle
# f_distance_sameL, f_dist_L, f_dist_Laneabove
    def get_vehicle_lane(ego_state):
        bins = np.linspace(0, 1.0, N_LANE)
        return np.digitize(ego_state[1], bins)


    f_sameLane, f_laneAbove, f_laneBelow, f_velocity, f_heading, f_collision = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    obs_ego = state[0]
    obs_other = state[1]
    lane_offset = 1/N_LANE

    # ego_lane = get_vehicle_lane(obs_ego)
    invalid_obs = np.all((obs_other == 0)) 
    
    if invalid_obs == False:
        if np.around(obs_other[1], decimals=1) == 0.0:
            # Dealing with the same lane
            f_sameLane = obs_other[0]
            
            if f_sameLane < 0.03:
                f_collision = 1
        elif obs_other[1] < 0.0 and abs(obs_other[1]) - lane_offset < 0.01 and obs_other[0] > 0:
            f_laneAbove = math.sqrt(obs_other[0]**2 + obs_other[1]**2)
        elif obs_other[1] > 0.0 and abs(obs_other[1]) - lane_offset < 0.01 and obs_other[0] > 0:
            f_laneBelow = math.sqrt(obs_other[0]**2 + obs_other[1]**2)

    # f_heading: Feature to penlize switching lanes (so that vehicle moves in stright line)
    # A boolean indicating swtiching lanes
    if obs_ego[4] != 0:
        f_heading = 1

    # f_maxS: Feature to reward higher speed 
    f_velocity = obs_ego[2]

    feature_vector = np.array([f_sameLane, f_laneAbove, f_laneBelow, f_velocity, f_heading, f_collision])
    
    return feature_vector

def main(): 
  
    env = gym.make("highway-v0")
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
        "vehicles_count": 1,
        "screen_height": 150,
        "screen_width": 600,
    }
    env.configure(config)
    env.reset() 
    trajectory = utils.record_trajectories(env, max_timesteps = 10)
    print(trajectory)
    # for traj in trajectories:
    #     t = traj[:, 0, 0]
    #     print(t)
    # print(t)

    #theta = [0.66018253, -87.48911503, -44.98564138, -389.32662012, 42.51299763, 22.98094568]
    #r1 = [-137.24426340879288, -137.06469743390383, -136.9169573823888, -91.12354755570269, -78.98095179122116, 0.13490783178837873, 42.284887861431244, 57.25927346262358, 62.5791394422334, 64.46909816787173, 65.14053315948826, 65.51374878545731]
    # for state in trajectory1:
    #     feat = feature_func(state[0])
    #     r1.append(np.dot(feat, theta))
    # traj_reward1 = np.sum(r1, axis=0)
    # print(r1)

    # trajectory2 = utils.record_trajectories(env, max_timesteps = 10)
    #r2 = [-137.24426340879288, -137.06469743390383, -136.9169573823888, -136.801975092643, -136.72050108229703, -136.67308109704445, -121.66456878749999, -121.66456878749999, -121.66456878749999, -121.66456878749999, -121.66456878749999, -120.69125316542699]
    # for state in trajectory2:
    #     feat = feature_func(state[0])
    #     r2.append(np.dot(feat, theta))
    # traj_reward2 = np.sum(r2, axis=0)
    # print(r2)

    # np.savetxt("trajs.txt", trajectories, fmt='%s')


    # x = np.arange(0, len(r1), 1, dtype=int)

    # plt.plot(list(x), list(r1), color = "r", label = "Trajectory-crash")
    # plt.plot(list(x), list(r2), color = "b", label = "Trajectory-line")

    # plt.xlabel("Timesteps")
    # plt.ylabel("Reward R(s)")

    # plt.show()

    # print("Done")
    # np.savetxt("traj1.txt", trajectory1, fmt='%s')
    # for state in trajectory:
    #     print(feature_func(state))
    # filename = "highway_dqn/model2"
    # model = DQN('MlpPolicy', env, verbose=1)
    # model.learn(total_timesteps=25000)
    # model.save("highway_dqn/model2")
    # train_dqn(env, filename)

    # model = DQN.load(filename)

    # t = utils.record_trajectories(env, max_timesteps = 10)
    # r = []
    # for state in t:
    #     _, reward , _, _ = env.step(state[1])
    #     r.append(reward)
    #     print("Action and Reward", state[1], reward)

    # trajectory_sample = utils.record_trajectories(env, max_timesteps = 10)

    # while True:
    #     done = False
    #     obs = env.reset()
    #     while not done:
    #         action, _states = model.predict(obs, deterministic=True)
    #         action = int(action)
    #         obs, reward, done, info = env.step(action)
    #         print("Action, Reward: ", action, reward)
    #         env.render()

    ############### DTW TEST #########################
    # x1 = [[1.0,   0.0], [1.0,   0.0], [1.0,   0.0], [1.0,   0.0], [1.0,   0.0], [1.0,   0.0], [1.0,   0.0], [1.0,   0.0], [1.0,   0.0], [1.0,   0.0], [1.0,   0.0]]
    # x2 = [[1.0,   0.0], [1.0,   0.0], [1.0,   0.0], [1.0,   0.27], [1.0,   0.33], [1.0,   0.33], [1.0,   0.33], [1.0,   0.33], [1.0,   0.33], [1.0,   0.33], [1.0,   0.33]]
    # x = []
    # x.append(x1)
    # x.append(x2)
    # y = np.array(x2)
    # distance, path = fastdtw(x1, x2, dist=euclidean)
    # distance = frechet_distance(x)
    # distance = dtw(x1, x2)
    # print(distance)



if __name__ == "__main__":
    main()