
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
'''
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


'''
def data_analysis(path):
    '''
    analyzes the data contained in path, using pearson correlation for comparison with the connectome and the real BOLD data
    acquired experimentally. Returns a tuple containig the correlations of the three pairs of the data
    '''

    #Loading and preprocessing data: cut the first 20 s due to fucked up initial conditions

    real_data = np.load('Data/real_bold.npy').transpose()
    simulated_data = np.load(path + '.npy')[:,2000:]


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
    #Correlation between the acquired data and the simulated one
    
    correlationS =  pearsonr(real_R.flatten(), simulated_R.flatten()).statistic
    mseS = np.square(np.subtract(real_R.flatten(), simulated_R.flatten())).mean()

    W = np.load('Data/desikan.npy')
    W = normalize_data(W)
    #Correlation between the connectome and the simulated data
    correlationW = pearsonr(W.flatten(), simulated_R.flatten()).statistic
    mseW = np.square(np.subtract(W.flatten(), simulated_R.flatten())).mean()
    #Correlation between the acquired data and the connectome
    correlationC = pearsonr(W.flatten(), real_R.flatten()).statistic
    mseC = np.square(np.subtract(real_R.flatten(), W.flatten())).mean()

    return (correlationS, correlationW, correlationC, mseS, mseW, mseC)
