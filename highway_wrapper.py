import numpy as np
import gym
from typing import TypeVar
import random


class highwayWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, obs):
        # return only state of the EGO Vehicle
        return obs[0]
    
    def reward(self, rew):
        # modify rew
        return rew