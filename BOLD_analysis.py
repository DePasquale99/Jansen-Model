
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import pearsonr
from time import time
import matplotlib.pyplot as plt
import py_compile
import cProfile
import seaborn as sns


def normalize_data(W):
    #takes the W matrix and normalizes the data by row:
    norm = np.sum(W, axis = 1)
    for idx, i in enumerate(norm):
        W[idx] /= i
    return W



#Loading and preprocessing data: cut the first 20 s due to fucked up initial conditions

real_data = np.load('Data/real_bold.npy').transpose()
simulated_data = np.load('Data/desikan_bold.npy')[:,2000:]


N = len(simulated_data)

real_R = np.zeros((N,N))
simulated_R = np.zeros((N,N))
#Calculating pearson correlation coefficient for both data
for i in range(N):
    for j in range(i):
        R = pearsonr(simulated_data[i], simulated_data[j]).statistic
        simulated_R[i, j] = R
        simulated_R[j, i] = R
        R = pearsonr(real_data[i], real_data[j]).statistic
        real_R[i, j] = R
        real_R[j, i] = R

print('Correlation coefficient for the real and simulated data:')
print(pearsonr(real_R.flatten(), simulated_R.flatten()))

W = np.load('Data/desikan.npy')
W = normalize_data(W)
sns.heatmap(W)
plt.show()

print('Correlation coefficient for the connectome and simulated data:')
print(pearsonr(W.flatten(), simulated_R.flatten()))


print('Correlation coefficient for the connectome and real data:')
print(pearsonr(W.flatten(), real_R.flatten()))



