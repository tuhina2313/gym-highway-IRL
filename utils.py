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
def feature_func(state):
# f_distance: Distance from the other vehicle
    f_distance, f_velocity, f_sameLane, f_heading, f_collision = 0
    obs_ego = state[0]
    obs_other = state[1]
    if np.all((obs_other == 0)) == False:
        if obs_other[1] == 0:
            # Dealing with the same lane
            f_distance = obs_other[0]

            f_sameLane = 1
            
            if f_distance < 0.03:
                f_collision = 1
        elif obs_other[1] < 0.33 or obs_other[1] > -0.33:
            f_distance = obs_other[0]

    # f_heading: Feature to penlize switching lanes (so that vehicle moves in stright line)
    # A boolean indicating swtiching lanes
    if obs_ego[4] != 0:
        f_heading = 1

    # f_maxS: Feature to reward higher speed 
    f_velocity = obs_ego[2]

    feature_vector = np.array([f_distance, f_velocity, f_heading, f_sameLane, f_collision])
    return feature_vector/max(feature_vector)


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
        if keys[pygame.K_RIGHT]:
            action = 3
        if(keys[pygame.K_LEFT]):
            action = 4
        if(keys[pygame.K_DOWN]):
            action = 2
        if(keys[pygame.K_UP]):
            action = 0
        obs, reward, _, _ = env.step(action)
        observation_tuple = []
        observation_tuple.append(obs[0].tolist())
        observation_tuple.append(obs[1].tolist())
        feature_matrix = []
        trajectory.append((np.around(obs, decimals=2), action))
        # trajectory.append((obs, action))
        timeStep += 1
    observation_tuple.append(obs[0].tolist())
    observation_tuple.append(obs[1].tolist())
    trajectory.append((np.around(obs, decimals=2), None))
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

    plt.plot(list(x), list(reward[0]), color = "r", label = "Trajectory 1")
    plt.plot(list(x), list(reward[1]), color = "g", label = "Trajectory 2")
    plt.plot(list(x), list(reward[2]), color = "b", label = "Trajectory 3")

    plt.xlabel("Timesteps")
    plt.ylabel("Reward R(s)")

    plt.show()


