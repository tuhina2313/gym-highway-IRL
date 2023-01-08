from asyncio.format_helpers import _format_callback_source
from time import time
from matplotlib.pyplot import axes
import numpy as np
# import ValueIteration
# from utils import eucledian_distance
import math
import gym
import highway_env
from stable_baselines3 import DQN

N_LANE = 3

def eucledian_distance(p1, p2):
    dist = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return dist

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

    # Distance vector to keep track of collision
    distance = np.zeros(len(obs_other))
    itr = 0
    for obs in obs_other:
        distance[itr] = eucledian_distance(obs_ego[:2], obs[:2])
        itr = itr +1
    if (all(i >= 0.1 for i in distance) == True):
        f_collision = 0
    else:
        f_collision = -1

    feature_vector = np.array([f_sameLane_ahead, f_sameLane_behind, f_laneAbove_ahead, f_laneAbove_behind, f_laneBelow_ahead, f_laneBelow_behind, f_velocity, f_heading, f_collision])
    # normalized_feat = (feature_vector-np.min(feature_vector))/(np.max(feature_vector)-np.min(feature_vector))
    return feature_vector

# THIS VERSION TAKES TRAJECTORIES
def calc_feature_expectations(traj):
    feature_vector = np.zeros(5)
    # f_distance: Distance from the other vehicle
    f_distance = [
        -1 * (eucledian_distance(tup))
        for tup in traj
    ]
    feature_vector[0] = f_distance
    # f_lane: Feature to penlize switching lanes
    v_pos = 0.04
    v_abs = [abs(v_pos - tup[0][1]) for tup in traj]
    for v in v_abs:
        if (v_abs > 0.01):
            f_lane = 1
        else:
            f_lane = 0
    
    # f_maxS: Feature to reward higher speed 
    v_max = 20
    f_maxS = [((tup[0][2] - v_max) ** 2) for tup in traj]

    # f_heading to minimise the heading angle so that vehicle moves in stright line
    f_heading = [tup[0][4] for tup in traj]

    # f_collision to penalise collision
    v_collision = [(eucledian_distance(tup)) for tup in traj]
    f_collision = np.array(len(v_collision))
    for i in range(v_collision):
        if v_collision[i] < 0.01:
            f_collision[i] = 5

    return np.array([f_distance, f_lane, f_maxS, f_heading, f_collision])

def irl(env, trajectories, feature_vector ,action_space, epochs, gamma, alpha):
  # tf.set_random_seed(1)   

  # initialise the neural network
  feature_dim = feature_vector.shape[0]
  theta = np.random.normal(0, 0.05, size=feature_dim)
  feature_exp = np.zeros([feature_dim])

  expert_traj_features = []
  expert_demo_feat = np.zeros(feature_dim)
  
  start_states = []
  for traj in trajectories:
    expert_features = []
    start_states.append(traj[0][0])
    for state in traj:
        state_features = np.array(feature_func(state[0]))
        expert_features.append(state_features)
    expert_traj_features.append(np.sum(expert_features, axis=0))

  for traj in expert_traj_features:
    expert_demo_feat += traj
    
  expert_demo_feat_n = expert_demo_feat/len(expert_traj_features)
  print(np.around(expert_demo_feat_n, decimals=3))

  timestep = 0
  max_timestep = 10

  single_trajectory_buffer = []
  trajectories_buffer =[]
  for state in start_states:
    single_traj_feat = np.array((len(start_states), 5))
    env.reset()
    while timestep <= max_timestep:
        s = state[0][0]
        state_features = np.array(feature_func(state))
        single_trajectory_buffer.append((state, state_features))

        best_action = 0
        best_value = float('-inf')
        for action in action_space:
            obs, reward, done, info = env.step(action)
            if reward > best_value:
                best_value = reward
                best_action = action
                state = obs
        #action, _states = model.predict(state, deterministic=True)

        # obs, reward, done, info = env.step(action)
        timestep = timestep+1
    
    trajectories_buffer.append(single_trajectory_buffer)
  
  for i in range(epochs):
    for traj in trajectories_buffer:
        scene_trajs = []
        for state in traj:
            reward = np.dot(state[1], theta)
            scene_trajs.append((reward, state[1]))

        rewards = [r[0] for r in scene_trajs]
        probability = [np.exp(r) for r in rewards]
        probability = probability /np.sum(probability)

        traj_features = np.array([t[1] for t in scene_trajs])
        feature_exp += np.dot(probability, traj_features)
   
   ################### PREVIOUS IMPLEMENTATION ###########################
    theta = 2 * 0.01 * theta
    grad = expert_demo_feat - feature_exp 
    grad = np.array(grad)

    if np.linalg.norm(grad) < 0.01:
        return theta

    theta += alpha * grad
    ###################### GRADIENT DESCENT ###############################
    # loss = (expert_demo_feat**2) - (feature_exp**2) + (2 * 0.01 * theta)
    # theta = theta - (alpha*loss)
    # print("Epoch: ", i, " Weights: ", np.around(theta, decimals=2))

  return theta, expert_demo_feat_n



