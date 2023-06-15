import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eig

'''
Goal of this code is to analyze the topological properties of the network matrix; in particular:
-Calculate eigenvalues and eigenvectors
-Calculate degree distribution
-Calculate betweenness centrality
-Calculate clustering coefficient
'''

###Selecting the mdjacency matrix and building a weighted graph on it:

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



A = np.load('Data/desikan.npy')
A1 = normalize_data(A)
A2 = homotopic_connections(A1, .15)

Graph = nx.from_numpy_matrix(A)
Graph1 = nx.from_numpy_matrix(A1)
Graph2 = nx.from_numpy_matrix(A2)


def degree(Graph):
  #prints degree distribution of the graph
    n = len(Graph.nodes())

    degrees = list((d for n, d in Graph.degree()))
    nbins = max(i for i in degrees)

    return degrees 



def clustering(Graph):
  #prints clustering coefficient distribution for the graph
    n = len(Graph.nodes())

    clustering = list((nx.clustering(Graph)[c] for c in range(n)))
    return clustering




def betweenness_centrality(Graph):
  #prints the betweenness centrality distribution
  n = len(Graph.nodes())
  betweenness_centrality = list((nx.betweenness_centrality(Graph)[i] for i in range(n)))

  return betweenness_centrality




#Node degree comparison

plt.hist(degree(Graph), label='unprocessed graph', alpha = 0.5)
plt.hist(degree(Graph1), label='row normailzed graph', alpha = 0.5)
plt.hist(degree(Graph2), label='homotopically connected normailzed graph', alpha = 0.5)

plt.legend()
plt.show()

#clustering coefficient comparison

plt.hist(clustering(Graph), label='unprocessed graph', alpha = 0.5)
plt.hist(clustering(Graph1), label='row normailzed graph', alpha = 0.5)
plt.hist(clustering(Graph2), label='homotopically connected normailzed graph', alpha = 0.5)

plt.legend()
plt.show()


#betweenness centrality comparison


plt.hist(betweenness_centrality(Graph), label='unprocessed graph', alpha = 0.5)
plt.hist(betweenness_centrality(Graph1), label='row normailzed graph', alpha = 0.5)
plt.hist(betweenness_centrality(Graph2), label='homotopically connected normailzed graph', alpha = 0.5)

plt.legend()
plt.show()

values, vectors = eig(A)

values = [float(value) for value in values]




