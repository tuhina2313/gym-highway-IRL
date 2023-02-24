import sys
from cv2 import normalize
import gym
import highway_env
import pickle
# from highway_wrapper import ObservationWrapper, RewardWrapper
import numpy as np
import pygame
# from trajectory import Trajectory
# import maxent
# import deep_irl

import matplotlib.pyplot as plt

# from maxent_highway import N_LANE

N_LANE = 3

pygame.init()
fpsClock = pygame.time.Clock()
display = pygame.display.set_mode((150, 600))

def eucledian_distance(pair):
    v1 = pair[0][0]
    v2 = pair[0][1]
    return ((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2)

def eucledian_distance2(pair):
    v1 = pair[0][0]
    v2 = pair[0][1]
    return ((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2)

def eucledian_distance2(p1, p2):
    dist = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return dist

def generate_trajectories(env, n_traj):
    trajectories = []
    for _ in range(n_traj):
        obs = env.reset()
        stop = False

        trajectory = []
        while stop is not True:
            old_obs = obs
            #action, _ = model.predict(obs, deterministic=True)
            action = env.action_space.sample()
            obs, reward, done, info = env.step(int(action))
            trajectory.append((old_obs, action, obs, reward))
            print("sp: ", old_obs[0], "action: ", action, "s: ", obs[0] , "reward: ", reward)
            
            if done == True:
                trajectories.append(trajectory)
                stop = True
                break
        trajectories.append(trajectory)
    return np.array(trajectories)

# this implementation goes by observation [[EGO][OTHER]]
# def feature_func(state):
# # f_distance: Distance from the other vehicle
#     f_distance, f_velocity, f_sameLane, f_heading, f_collision = 0
#     obs_ego = state[0]
#     obs_other = state[1]
#     if np.all((obs_other == 0)) == False:
#         if obs_other[1] == 0:
#             # Dealing with the same lane
#             f_distance = obs_other[0]

#             f_sameLane = 1
            
#             if f_distance < 0.03:
#                 f_collision = 1
#         elif obs_other[1] < 0.33 or obs_other[1] > -0.33:
#             f_distance = obs_other[0]

#     # f_heading: Feature to penlize switching lanes (so that vehicle moves in stright line)
#     # A boolean indicating swtiching lanes
#     if obs_ego[4] != 0:
#         f_heading = 1

#     # f_maxS: Feature to reward higher speed 
#     f_velocity = obs_ego[2]

#     feature_vector = np.array([f_distance, f_velocity, f_heading, f_sameLane, f_collision])
#     return feature_vector/max(feature_vector)


def feature_func(state):
# f_distance: Distance from the other vehicle
# f_distance_sameL, f_dist_L, f_dist_Laneabove
    def get_vehicle_lane(ego_state):
        bins = np.linspace(0, 1.0, N_LANE)
        return np.digitize(ego_state[1], bins)


    f_velocity, f_heading, f_collision = 0.0, 0, 0
    f_sameLane_ahead, f_sameLane_behind, f_laneAbove_ahead, f_laneAbove_behind, f_laneBelow_ahead, f_laneBelow_behind = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    obs_ego = state[0]
    obs_other = state[1:]
    lane_offset = 1/N_LANE

    invalid_obs = np.all((obs_other == 0)) 
    
    if invalid_obs == False:

        lane_info = obs_other[:,1]

        same_lane = obs_other[np.where(np.around(lane_info, decimals=1) == 0.0)]
        x_info = same_lane[:,0]

        sameLane_ahead = x_info[np.where(x_info >= 0.0)]
        f_sameLane_ahead = sameLane_ahead[0] if sameLane_ahead.size != 0 else 0.0

        sameLane_behind = x_info[np.where(x_info <= 0.0)]
        f_sameLane_behind = abs(sameLane_behind[0]) if sameLane_behind.size != 0 else 0.0

        lane_below = obs_other[np.where(np.around(lane_offset - lane_info, decimals=1) == 0.0)]
        x_info = lane_below[:,0]

        laneBelow_ahead = x_info[np.where(x_info >= 0.0)]
        f_laneBelow_ahead = laneBelow_ahead[0] if laneBelow_ahead.size != 0 else 0.0

        laneBelow_behind = x_info[np.where(x_info <= 0.0)]
        f_laneBelow_behind = abs(laneBelow_behind[0]) if laneBelow_behind.size != 0 else 0.0

        lane_above = obs_other[np.where(np.around(lane_offset - lane_info, decimals=1) == np.around(lane_offset*2, decimals=1))]
        x_info = lane_above[:,0]

        laneAbove_ahead = x_info[np.where(x_info >= 0.0)]
        f_laneAbove_ahead = laneAbove_ahead[0] if laneAbove_ahead.size != 0 else 0.0

        laneAbove_behind = x_info[np.where(x_info <= 0.0)]
        f_laneAbove_behind = abs(laneAbove_behind[0]) if laneAbove_behind.size != 0 else 0.0
    # f_heading: Feature to penlize switching lanes (so that vehicle moves in stright line)
    # A boolean indicating swtiching lanes
    if obs_ego[4] != 0:
        f_heading = 1

    # f_velocity: Feature to reward higher speed 
    v_max = 0.4
    f_velocity = obs_ego[2]

    feature_vector = np.array([f_sameLane_ahead, f_sameLane_behind, f_laneAbove_ahead, f_laneAbove_behind, f_laneBelow_ahead, f_laneBelow_behind, f_velocity, f_heading, f_collision])
    # normalized_feat = (feature_vector-np.min(feature_vector))/(np.max(feature_vector)-np.min(feature_vector))
    return feature_vector


def create_bins(nbins): 

    bins = np.zeros((5,nbins))
    bins[0] = np.linspace(-100.0, 100,0, nbins)
    bins[1] = np.linspace(-100.0, 100,0, nbins)
    bins[2] = np.linspace(-20.0, 20.0, nbins)
    bins[3] = np.linspace(-20.0, 20.0, nbins)
    bins[4] = np.linspace(-20.0, 20.0, nbins)
    return bins

def assign_bins(observation, bins): 
    """
    discretizing the continuous observation space into state
    """
    state = np.zeros(4)
    for i in range(4):
        state[i] = np.digitize(observation[i], bins[i])
    return state

def record_trajectories(env, max_timesteps):
    timeStep = 0
    trajectory = []
    obs = []
    # old_obs = env.reset()
    while timeStep <= max_timesteps:
        action = 1
        env.render()
        keys = pygame.key.get_pressed()
        if(keys[pygame.K_DOWN]):
            action = 0
        if(keys[pygame.K_UP]):
            action = 2
        obs, reward, _, _ = env.step(action)
        observation_tuple = []
        observation_tuple.append(obs[0].tolist())
        observation_tuple.append(obs[1].tolist())
        feature_matrix = []
        # trajectory.append((np.around(obs, decimals=2), action))
        trajectory.append((obs, action))
        timeStep += 1
    observation_tuple.append(obs[0].tolist())
    observation_tuple.append(obs[1].tolist())
    trajectory.append((obs, None))
    return np.array(trajectory)


def configure_env(env):
    config = {
    "observation": {
    "vehicles_count": 1,
    "features": [ "x", "y", "vx", "vy", "heading"],
    "features_range": {
        "x": [-100, 100],
        "y": [-100, 100],
        "vx": [-20, 20],
        "vy": [-20, 20]
        },
    "absolute": False
    },
    "lanes_count": 3,
    "show_trajectories": True,
    "manual_control": True,
    "real_time_rendering": True
    }
    env.reset()
    return env


def save_trajs_nicely(trajectories):
    for traj in trajectories: 
        for obs in traj:
            observation_tuple = []
            observation_tuple.append(obs[0].tolist())
            observation_tuple.append(obs[1].tolist())


def save_trajectories(trajectories, filename):
    try:
        with open(filename, "wb") as f:
            pickle.dump(trajectories, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

def load_trajectories(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)

def get_reward_plot(reward):
    
    x = np.arange(0, 12, 1, dtype=int)

    label1 = "Trajectory Reward: "+ str(np.around(np.sum(reward[0]),decimals=2))
    label2 = "Trajectory Reward: "+ str(np.around(np.sum(reward[1]),decimals=2))
    label3 = "Trajectory Reward: "+ str(np.around(np.sum(reward[2]),decimals=2))

    plt.plot(list(x), list(reward[0]), color = "b", label = label1)
    # plt.plot(list(x), list(reward[1]), color = "g", label = label2)
    # plt.plot(list(x), list(reward[2]), color = "m", label = label3)

    plt.xlabel("Timesteps")
    plt.ylabel("Reward R(s)")
    # plt.title(label1+label2)
    
    plt.legend(loc='best')
    plt.show()


