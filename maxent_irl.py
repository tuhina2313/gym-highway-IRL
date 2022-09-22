import numpy as np
import gym
import highway_env
# import ValueIteration
import value_iteration
from utils import eucledian_distance


def normalize(input):
  minimum = np.min(input)
  maximum = np.max(input)
  return (input - minimum) / (maximum - input)

def convert_to_vector(rewards):
    R = []
    for i in range(len(rewards)):
        R.append(rewards[i][0])
    return np.array(R)


def demo_svf(trajs, n_states):
  p = np.zeros(n_states)
  for traj in trajs:
    for state, _, _ in traj:
      p[int(state)] += 1
  p = p/len(trajs)
  return p

def calc_feature_expectations(traj):
    # f_distance: Distance from the other vehicle
    f_distance = [
        -1 * (eucledian_distance(tup))
        for tup in traj
    ]

    # f_lane: Feature to penlize switching lanes
    v_pos = 0.04
    v_abs = [abs(v_pos - tup[0][1]) for tup in traj]
    for v in v_abs:
        if (v_abs > 0.01):
            f_lane = 10
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

    return (np.array([f_distance, f_lane, f_maxS, f_heading, f_collision]))

def irl(env, trajectories, feature_vector, epochs, gamma, alpha):
  # tf.set_random_seed(1)  

  # initialise the neural network
  feature_dim = feature_vector.shape[0]
  # find feature values given demonstrations
  expert_features = []
  for traj in trajectories:
    expert_features.append(calc_feature_expectations(traj))
  expert_feature_expectations = np.sum(np.asarray(expert_features), axis=0)

  learner_feature_expectations = np.array(feature_dim)

  theta = []

  # training 
  for epoch in range(epochs):
    state = env.reset()

    if epoch % (epochs/10) == 0:
      print ('epochs: {}'.format(epoch))
    
    # compute the reward matrix from the human demonstrations
    # Uses a neural network with two FC layers (dimensions should be num_of_features: input for the NN)

    # compute policy using R coming from demonstrations
    
    # mu_exp comes from sampling in the environment. The initial state should be the same as the initial state for the given human trajectory. 
    # The objective function to be minimised can be seen as a difference of the feature expectations between human traj and the sampled traj (same s0)
    mu_exp = []
    
    # compute gradients on rewards:
    grad_r = mu_D - mu_exp

    # apply gradients to the neural network
    # Update the weights (need to add regularization on the weights)

    # grad_theta, l2_loss, grad_norm = nn_r.apply_grads(mu_D, grad_r)
    

  rewards = nn_r.get_rewards(mu_D)
  # return sigmoid(normalize(rewards))
  return normalize(rewards)





