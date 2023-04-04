import numpy as np
import matplotlib.pyplot as plt

x_axis = np.arange(-5, 5, 0.1)
mu, sigma = 0, 0.1
mu2, sigma2 = 0, 0.5
X1 = np.random.normal(mu, sigma)
X2 = np.random.normal(mu2, sigma2, 50)
X = np.concatenate([np.random.normal(mu, sigma, 1), np.random.normal(mu2, sigma2, 1)])
print(X)
# plt.hist(X)
# plt.show() 