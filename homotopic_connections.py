import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

N = 68 #Number of regions


def normalize_data(W):
    #takes the W matrix and normalizes the data by row:
    norm_W = np.zeros(np.shape(W))
    norm = np.sum(W, axis = 1)
    for idx, i in enumerate(norm):
        norm_W[idx] = W[idx]/i
    return norm_W

def homotopic_connections(W, weight): 

    '''
Adds homotopic connections to a row normalized matrix:
    takes: 
    W - normalized matrix to be modified
    weight - percentage of homotopic connection over the overall input of the node
    '''
    homo_W = np.copy(W)
    weight *= weight + 1 #multiply for the normalization factor
    if len(W)%2 == 0:
        N = int(len(W)/2)
        for i in range(N):
            homo_W[i, i+N] += weight
            homo_W[i+N , i] += weight
    else:
        N = int((len(W)-1)/2)
        for i in range(N):
            homo_W[i, i+N] = weight
            homo_W[i+N , i] = weight
    
    homo_W = normalize_data(homo_W)
    return homo_W





#P1 connection matrix copied from Jansen network
#read_W('Data/data.dat')
W = np.load('Data/thalamus_matrix.npy')
W = normalize_data(W)

sns.heatmap(W)
plt.show()

#variable that swithces between using homotopic connections
homotopic = True
homotopic_weight = .15

if homotopic == True:
    W_homo = homotopic_connections(W, homotopic_weight)


sns.heatmap(W_homo)
plt.show()

sns.heatmap(W_homo-W)
plt.show()