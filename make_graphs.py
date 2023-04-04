# from lmfit.models import SkewedGaussianModel
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import skewnorm, norm
import statistics
from scipy.interpolate import make_interp_spline



import utils
import collections

retrieved_R = utils.load_trajectories(filename="resultData/rollout_rewards_expert")
agg_R = utils.load_trajectories(filename="resultData/rollout_rewards_agg")
un_R =  utils.load_trajectories(filename="resultData/low_human_rollouts")

retrieved_R = np.array(retrieved_R, dtype=float)
retrieved_R = np.around(retrieved_R, decimals=0)

# agg_R = np.around(agg_R, decimals=0)
# agg_R = [int(5 * round(float(x)/5)) for x in agg_R]

X_n = np.linspace(0, 40, 1000)
Y_n = np.linspace(0, 200, 1000)
# plt.plot(X_n, Y_n, color = "white")
#plt.show
# utils.save_trajectories(retrieved_R, filename="resultData/rollout_rewards_expert")
# utils.save_trajectories(agg_R, filename="resultData/rollout_rewards_agg")
############### Spline Curve for Optimal Policy distribution ##################
x_op , y_op = utils.get_sorted_frequency(retrieved_R)
print(x_op)
print(y_op)
X_Y_Spline_op = make_interp_spline(x_op, y_op)
X_ = np.linspace(min(x_op), max(x_op), 1000)
Y_ = X_Y_Spline_op(X_)

plt.plot(X_, Y_, color = "g")
plt.fill_between(X_,Y_,color='green',alpha=0.2)

############### Spline Curve for non-optimal Policy distribution ##################
# x_agg, y_agg = utils.get_sorted_frequency(agg_R)
# X_Y_Spline_agg = make_interp_spline(x_agg, y_agg)
# X_a = np.linspace(min(x_agg), max(x_agg), 1000)
# Y_a = X_Y_Spline_agg(X_a)

# plt.plot(X_a, Y_a, color = "b")
# plt.fill_between(X_a,Y_a,color='blue',alpha=0.2)

############### Spline Curve for rash Policy distribution ##################
x_un, y_un = utils.get_sorted_frequency(un_R)
X_Y_Spline_un = make_interp_spline(x_un, y_un)
X_u = np.linspace(min(x_un), max(x_un), 1000)
Y_u = X_Y_Spline_un(X_u)

plt.plot(X_u, Y_u, color = "r")
plt.fill_between(X_u,Y_u,color='red',alpha=0.2)


plt.show()

# x = np.linspace(0, 100, points)
# # plt.bar(frequency.keys(), frequency.values(), 10.0, color='b')
     
# # Varying positional arguments 
# y1 = skewnorm .pdf(x, 1, 60, 100) 
# plt.plot(x, y1, "--") 
# plt.show()