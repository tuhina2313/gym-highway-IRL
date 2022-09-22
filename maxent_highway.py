from asyncio.format_helpers import _format_callback_source
from time import time
from matplotlib.pyplot import axes
import numpy as np
# import ValueIteration
import tf_utils
# from utils import eucledian_distance
import math
import gym
import highway_env
from stable_baselines3 import DQN

N_LANE = 3

def eucledian_distance(pair):
    v1 = pair[0][0]
    v2 = pair[0][1]
    dist = ((v1 - v2) ** 2 + (v1 - v2) ** 2)
    return dist

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

    ego_lane = get_vehicle_lane(obs_ego)
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


# # FOR EACH OBSERVATION
# def feature_func(state):
# # f_distance: Distance from the other vehicle  
#     f_distance = (-1) * (eucledian_distance(state))

#     # f_lane: Feature to penlize switching lanes
#     v_pos = 0.04
#     v_abs = abs(v_pos - state[0][1])
#     if (v_abs > 0.01):
#         f_lane = 1
#     else:
#         f_lane = 0
    
#     # f_maxS: Feature to reward higher speed 
#     """
#     v_max = 20
#     f_maxS = (state[0][2] - v_max) ** 2
#     """

#     f_maxS = float(state[0][2])
#     # f_heading to minimise the heading angle so that vehicle moves in stright line
#     f_heading = float(state[0][4])

#     # f_collision to penalise collision
#     v_collision = eucledian_distance(state)
#     f_collision = 0
#     if v_collision < 0.01:
#         f_collision = 1

#     feature_vector = np.array([f_distance, f_lane, f_maxS, f_heading, f_collision])
#     return feature_vector/max(abs(feature_vector))

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
    
  # expert_demo_feat = expert_demo_feat/len(expert_traj_features)

  timestep = 0
  max_timestep = 10

  model = DQN.load("highway_dqn/model")

  feature_buffer = []
  single_trajectory_buffer = []
  trajectories_buffer =[]
  for state in start_states:
    single_traj_feat = np.array((len(start_states), 5))
    env.reset()
    while timestep <= max_timestep:
        s = state[0][0]
        state_features = np.array(feature_func(state))
        single_trajectory_buffer.append((state, state_features))

        action, _states = model.predict(state, deterministic=True)

        obs, reward, done, info = env.step(action)

        state = obs
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
    print("Epoch: ", i, " Weights: ", np.around(theta, decimals=2))

  return theta



