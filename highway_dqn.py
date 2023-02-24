from os import times
from time import time
import numpy as np
import gym
import highway_env
from stable_baselines3 import DQN
import utils
from copy import copy

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def train_new_dqn(env, model_file):
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20000, log_interval=4)
    model.save(model_file)



def train_dqn(env, filename):
    model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=5000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                verbose=1,
                tensorboard_log="highway_dqn/Model_with_R")
    model.learn(int(2e4))
    model.save(filename)

# Load and test saved model
def test_model(env, filename, max_timesteps):
    model = DQN.load(filename)
    while True:
        done = False
        obs = env.reset()
        timeStep = 0
        while timeStep <= max_timesteps:
            action, _states = model.predict(obs)
            print("Timestep: ", timeStep, "Action: ", action)
            obs, reward, done, info = env.step(int(action))
            env.render()
            timeStep = timeStep +1 

def get_stochastic_transition(env, obs, action_dict, delta):
    filename = "highway_dqn/groundtruth_continuous"
    model = DQN.load(filename)
    action, _states = model.predict(obs)
    print(action)
    cont_action = action_dict[int(action)]
    naction = tuple(map(lambda i, j: i + j, cont_action, delta))

    obs, reward, done, info = env.step(naction)
    return obs, reward, naction

def create_stochastic_transitions(env):
    filename = "highway_dqn/groundtruth_continuous"
    model = DQN.load(filename)
    
    action_dict = utils.get_action_dict()
    delta = 0.5
    action_delta = [(delta, 0), (-delta, 0) ,(0, delta), (0, -delta)]
    
    obs = env.reset()
    timestep = 0
    result = []
    while timestep < 15:
        obs_prob = []
        for adelta in action_delta:
            playout = copy(env) 
            nobs, reward, action = get_stochastic_transition(playout, obs, action_dict, adelta)
            obs_prob.append( (action, reward))
        result.append(obs_prob)
        action, _states = model.predict(obs)
        obs , reward, _ , _ = env.step(action_dict[int(action)])
        timestep = timestep + 1
    return result

def test_stochastic_transitions(env):
    filename = "highway_dqn/groundtruth_continuous"
    model = DQN.load(filename)
    
    action_dict = utils.get_action_dict()
    delta = 0.1
    action_delta = [(delta, 0), (-delta, 0) ,(0, delta), (0, -delta), (2*delta, 0), (-2*delta, 0) ,(0, 2*delta), (0, -2*delta), (5*delta, 0), (-5*delta, 0) ,(0, 5*delta), (0, -5*delta)]
    
    obs = env.reset()
    timestep = 0
    while timestep < 15:
        action, _states = model.predict(obs)
        obs , reward, _ , _ = env.step(action_dict[int(action)])
        obs_prob = []
        if timestep == 4:
        
            for adelta in action_delta:
                playout = copy(env) 
                nobs, reward, action = get_stochastic_transition(playout, obs, action_dict, adelta)
                obs_prob.append( (action, reward))
            return obs_prob
        timestep = timestep + 1
    return obs_prob

def get_action_probabilities(env, filename, max_timesteps):
    model = DQN.load(filename)
    action_probabilties = []
    obs = env.reset()
    timeStep = 0
    while timeStep <= max_timesteps:
        action, _states = model.predict(obs)
        action_rewards = []
        print("Timestep: ", timeStep)
        for i in range(5):
            obs, reward, done, info = env.step(i)
            env.render()
            action_rewards.append(reward)
        action_probabilties.append(np.around(action_rewards,decimals=2))
        min_reward = min(action_rewards)
        worst_action = action_rewards.index(min_reward)
        obs, reward, done, info = env.step(int(worst_action))
        timeStep = timeStep +1
        # env.render()
    np.savetxt("reward_worst.txt", action_probabilties, fmt='%s') 

def modify_action(action, action_dict):
    naction = action_dict[action]


def record_trajectories(env, model_file, max_timesteps):
    model = DQN.load(model_file)

    timeStep = 0
    trajectory = []
    obs = env.reset()
    while timeStep <= max_timesteps:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(int(action))
        trajectory.append((obs, action))
        timeStep += 1
        # env.render()
    trajectory.append((obs, None))
    return np.array(trajectory)