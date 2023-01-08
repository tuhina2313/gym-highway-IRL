import numpy as np
import gym
import utils
import maxent_highway
import matplotlib.pyplot as plt

def record_new_traj(n_traj):
   env = gym.make("highway-v0")
#    env = Monitor(gym.make('highway-v0'), './video', force=True)

   np.random.seed(50)
   env.seed(50)

   trajectories = []

   config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": [ "x", "y", "vx", "vy", "heading" ],
            },
        "absolute": False,
        "lanes_count": 3,
        "show_trajectories": True,
        "manual_control": True,
        "real_time_rendering": True,
        "vehicles_count": 5,
        "screen_height": 150,
        "screen_width": 600,
    }
   env.configure(config)

   for _ in range(n_traj):
    np.random.seed(50)
    env.seed(50)
    env.reset() 
    trajectory = utils.record_trajectories(env, max_timesteps=15)
    trajectories.append(trajectory)
   return trajectories

def analyse_R():
    #Loading trajectories
    t_ground = utils.load_trajectories("newData/Expert_vanilla.pickle")

    #Theta calculated from IRL
    # ground_theta = [0.13, 0.0,   0.02, 0.0,   0.06, 0.0,   0.20, 0.90, 1.0  ]
    ground_dqn = [0.1,  0.0,   0.1, 0.0,   0.05, 0.0,   0.46, 0.87, 1.0  ]

    # Record Preference trajectory 
    # trajectory = record_new_traj(4)
    # utils.save_trajectories(trajectory, filename="dqnData/final_test.pickle")

    t_pref = utils.load_trajectories("dqnData/final_test.pickle")

    test_run = []
    # test_run.append(t_ground[0])
    test_run.append(t_pref[0])
    test_run.append(t_pref[1])
    test_run.append(t_pref[2])
    test_run.append(t_pref[3])


    # Preparing Hybrid Trajectory
    # hybrid_traj = t_crash
    # test1 = hybrid_traj[0][0]
    # for i in range(5,len(t_ground[0])):
    #     hybrid_traj[i][0] = t_ground[0][i][0]

    # test_run.append(hybrid_traj)

    for state in test_run[3]:
        feat = maxent_highway.feature_func(state[0])

    reward = []
    for traj in test_run:
        r = []
        r_cumulative = 0
        for state in traj:
            feat = maxent_highway.feature_func(state[0])           
            r_cumulative = r_cumulative + np.dot(feat, ground_dqn)
            r.append(r_cumulative)
        reward.append(r)
    # utils.get_reward_plot(reward)
    x = np.arange(0, len(reward[0]), 1, dtype=int)
    plt.plot(list(x), list(reward[0]), color = "g")
    plt.plot(list(x), list(reward[1]), color = "b")
    plt.plot(list(x), list(reward[2]), color = "y")
    plt.plot(list(x), list(reward[3]), color = "r")

    # plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    # plt.yticks(np.arange(min(reward[0]), max(reward[0])+1, 0.5))
    plt.show()

if __name__ == "__main__":
    analyse_R()