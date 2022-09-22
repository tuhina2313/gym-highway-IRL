from asyncio.format_helpers import _format_callback_source
import numpy as np
# import ValueIteration
import tf_utils
from utils import eucledian_distance

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

def normalize(input):
  minimum = np.min(input)
  maximum = np.max(input)
  return (input - minimum) / (maximum - input)

class DeepIRL:
  def __init__(self, n_input, alpha, n_h1=400, n_h2=300, l2=10, name='deep_irl'):
    self.n_input = n_input
    self.lr = alpha
    self.n_h1 = n_h1
    self.n_h2 = n_h2
    self.name = name

    self.sess = tf.compat.v1.Session()
    self.input_s, self.reward, self.theta = self._build_network(self.name)
    self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(alpha)
    
    self.grad_r = tf.compat.v1.placeholder(tf.float32, [None, 1])
    self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.theta])
    self.grad_l2 = tf.gradients(self.l2_loss, self.theta)

    self.grad_theta = tf.gradients(self.reward, self.theta, -self.grad_r)
    # apply l2 loss gradients
    self.grad_theta = [tf.add(l2*self.grad_l2[i], self.grad_theta[i]) for i in range(len(self.grad_l2))]
    self.grad_theta, _ = tf.clip_by_global_norm(self.grad_theta, 100.0)

    self.grad_norms = tf.compat.v1.global_norm(self.grad_theta)
    self.optimize = self.optimizer.apply_gradients(zip(self.grad_theta, self.theta))
    self.sess.run(tf.compat.v1.global_variables_initializer())


  def _build_network(self, name):
    input_s = tf.compat.v1.placeholder(tf.float32, [None, self.n_input])
    with tf.compat.v1.variable_scope(name):
      # tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN")  
      fc1 = tf_utils.fc(input_s, self.n_h1, scope="fc1", activation_fn=tf.nn.elu,
        initializer = None)
      fc2 = tf_utils.fc(fc1, self.n_h2, scope="fc2", activation_fn=tf.nn.elu,
        initializer = None)
      reward = tf_utils.fc(fc2, 1, scope="reward")
    theta = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=name)
    return input_s, reward, theta


  def get_theta(self):
    return self.sess.run(self.theta)


  def get_rewards(self, states):
    rewards = self.sess.run(self.reward, feed_dict={self.input_s: states})
    return rewards


  def apply_grads(self, feat_map, grad_r):
    grad_r = np.reshape(grad_r, [-1, 1])
    feat_map = np.reshape(feat_map, [-1, self.n_input])
    _, grad_theta, l2_loss, grad_norms = self.sess.run([self.optimize, self.grad_theta, self.l2_loss, self.grad_norms], 
      feed_dict={self.grad_r: grad_r, self.input_s: feat_map})
    return grad_theta, l2_loss, grad_norms

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

# FOR EACH OBSERVATION
def feature_func(state):
# f_distance: Distance from the other vehicle  
    f_distance = (-1) * (eucledian_distance(state))

    # f_lane: Feature to penlize switching lanes
    v_pos = 0.04
    v_abs = abs(v_pos - state[0][0][1])
    if (v_abs > 0.01):
        f_lane = 10
    else:
        f_lane = 0
    
    # f_maxS: Feature to reward higher speed 
    v_max = 20
    f_maxS = (state[0][0][2] - v_max) ** 2

    # f_heading to minimise the heading angle so that vehicle moves in stright line
    f_heading = state[0][0][4]

    # f_collision to penalise collision
    v_collision = eucledian_distance(state)
    f_collision = 0
    if v_collision < 0.01:
        f_collision = 5

    return (np.array([f_distance, f_lane, f_maxS, f_heading, f_collision]))

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

    return np.array([f_distance, f_lane, f_maxS, f_heading, f_collision])

def irl(env, trajectories, feature_vector ,action_space, epochs, gamma, alpha):
  # tf.set_random_seed(1)   

  # number of features
  feature_dim = feature_vector.shape[0]

  # randomly initialise weights (theta) and feature_exp -- feature expectation for the learner
  theta = np.random.normal(0, 0.05, size=feature_dim)
  feature_exp = np.zeros([feature_dim])
  
  # initialise the neural network
  nn_r = DeepIRL(feature_dim, alpha, 3, 3)

  # initialise the human demonstration feature expectations
  # expert_features = np.array([0.0 , 0.0, 0.0, 0.0, 0.0])
  expert_features = []
  
  # find feature values given demonstrations
  for traj in trajectories:
    for state in traj:
      state_features = np.array(feature_func(state))
      expert_features.append(state_features)
  # expert_feature_expectations = np.sum(np.asarray(expert_features), axis=1)
    print(expert_features)
    #learner_feature_expectations = np.array([0.0 , 0.0, 0.0, 0.0, 0.0])
    learner_feature_expectations = []

  # training 
  for epoch in range(epochs):
    if epoch % (epochs/10) == 0:
      print ('epochs: {}'.format(epoch))
    
    # compute the reward matrix from the human demonstrations
    # Uses a neural network with two FC layers (dimensions should be num_of_features: input for the NN)
    # We are going to TUNE THIS neural network
    dummy_one_hot = np.zeros((5,5))
    rewards = nn_r.get_rewards(dummy_one_hot)

    # learner exp comes from sampling in the environment. The initial state should be the same as the initial state for the given human trajectory. 
    # The objective function to be minimised can be seen as a difference of the feature expectations between human traj and the sampled traj (same s0)
    for traj in trajectories:
      traj_length = len(traj)
      obs = traj[0][0]
      env.reset()
      for i in range(traj_length):
        max_q = 0
        next_step = obs
        for key in action_space:
          next_obs, reward, done, _ = env.step(key)
          if reward > max_q:
            next_step = next_obs
            max_q = reward
        
        # observation_tuple = []
        # observation_tuple.append(next_step[0].tolist())
        # observation_tuple.append(next_step[1].tolist())
        next_step_list = []
        for step in next_step:
          next_step_list.append(step.tolist())
        learner_feature_expectations += feature_func([next_step_list])
    
    # compute gradients on rewards:
      grad_r = expert_features - learner_feature_expectations

    # apply gradients to the neural network
    # Update the weights (need to add regularization on the weights)

      grad_theta, l2_loss, grad_norm = nn_r.apply_grads(feature_vector, grad_r)
    

  rewards = nn_r.get_rewards(feature_vector)
  # return sigmoid(normalize(rewards))
  return normalize(rewards)




