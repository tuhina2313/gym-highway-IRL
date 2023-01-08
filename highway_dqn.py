import numpy as np
import gym
import highway_env
from stable_baselines3 import DQN

def train_dqn(env, filename):
    model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
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
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(int(action))
            env.render()
            timeStep = timeStep +1 

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