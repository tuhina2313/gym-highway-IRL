import numpy as np
import gym
import highway_env
from stable_baselines3 import DQN, A2C
import utils

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def train_a2c(env, model_file):
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save(model_file)