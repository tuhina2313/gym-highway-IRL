import numpy as np
from utils import eucledian_distance

class Trajectory:
    def __init__(self, trajectory):
        self.trajectory = trajectory
        self.features = feature_func(trajectory)
    
def feature_func(traj):
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