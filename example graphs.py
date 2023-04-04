# from lmfit.models import SkewedGaussianModel
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import skewnorm, norm
import statistics
from scipy.interpolate import make_interp_spline



import utils
import collections

X_n = np.linspace(0, 10, 100)
Y_n = np.linspace(0, 20, 100)

x = [3, 4, 5, 6, 7]
y_op = [10, 12, 13, 15, 17]
new = [9, 11, 12, 14, 16]
y_nop = [8, 10, 11, 13, 15]
low = [5, 6, 7, 9, 11]

X_Y_Spline_op = make_interp_spline(x, y_op)
X_ = np.linspace(min(x), max(x), 1000)
Y_ = X_Y_Spline_op(X_)

X_Y_Spline_nop = make_interp_spline(x, y_nop)
X_ = np.linspace(min(x), max(x), 1000)
Y_n = X_Y_Spline_nop(X_)

X_Y_Spline_new = make_interp_spline(x, new)
X_ = np.linspace(min(x), max(x), 1000)
Y_new = X_Y_Spline_new(X_)

X_Y_Spline_low = make_interp_spline(x, low)
X_ = np.linspace(min(x), max(x), 1000)
Y_low = X_Y_Spline_low(X_)


plt.plot(X_, Y_,color = "g", label = "Optimal")
# plt.fill_between(X_,Y_,color='green',alpha=0.2)

plt.plot(X_, Y_n, '--', color = "g")
plt.fill_between(X_,Y_, Y_n,color='green',alpha=0.2)

plt.plot(X_, Y_new, color = "b", label = "User-1")
plt.plot(X_, Y_low, color = "r", label = "User-2")




plt.show()