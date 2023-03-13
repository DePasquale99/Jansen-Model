#Goal : to implement a network of n regions modelled by LaNMM modeland connect them via a matrix W

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from time import time
###Parameters for LaNMM model

e0 = 2.5 #half of the maximum firing rate (Hz)
v0 = 6 #potential at which half of the maximum firing rate is achieved (mV)
v0_p2 = 1
r = 0.56 #slope of the sigmoid at v = v0 (mV^-1)

def sigma(x, v0): 
    #Wave to pulse operator, transforms the average membrane potential of a population into average firing rate
    return 2*e0 / (1+ np.exp(r*(v0- x)))


A_AMPA, A_GABAs, A_GABAf =  3.25, -22, -30 #average synapitc gain (mV)
a_AMPA, a_GABAs, a_GABAf = 100, 50, 220 #excitatory synaptic time rate (s^-1)

C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12 = 108, 33.7, 1, 135, 33.75, 70, 550, 1, 200, 100, 80, 200, 30

#Input parameter of the model, sets the external input as an average firing rate
p1, p2 = 200, 150 #p1 is noisy in the original model, produced by N(200,30)
#In order to transform from presynaptic firing rate to PSP (fixed point of second order eqt for p1,p2)
I1, I2 = (A_AMPA/a_AMPA)*p1, (A_AMPA/a_AMPA)*p2


############################################### Actual network model
epsilon = 10 #cross region connectivity
N = 90 #Number of regions

def read_W(data_address):
    #ad hoc function for reading data
    data = np.genfromtxt(data_address,
                     skip_header=1,
                     names=True,
                     dtype=None,
                     delimiter=' ')
    
    print(len(data))
    W = np.zeros((N,N))
    for row in data:
        i, j, w = row
        W[i,j] = w
        
    return W


#P1 connection matrix copied from Jansen network
W = read_W('Data/data.dat')
#initial conditions:
X0 = np.append(np.ones((N, 5)),np.zeros((N, 5)), axis = 1 )
dx = np.zeros((N, 10))


def Network_LaNMM(x,t):
    #at each time calculate the input from other columns as matrix*input column (in this case P1->P1):
    x = np.reshape(x, (N, 10))
    ext_p1 = epsilon*np.dot(W, x[:, 0])

    for i in range(N):
        #for every region of the brain we calculate the update
        dx[i,0] = x[i,5] #P1 population
        dx[i,5] = A_AMPA*a_AMPA*(sigma(C10*x[i,3]+C1*x[i,2]+C0*x[i,1]+I1 +ext_p1[i], v0))-2*a_AMPA*x[i,5]-a_AMPA**2*x[i,0]

        dx[i,1] = x[i,6] # SS population
        dx[i,6] = A_AMPA*a_AMPA*(sigma(C3*x[i,0], v0))-2*a_AMPA*x[i,6]-a_AMPA**2*x[i,1]

        dx[i,2] = x[i,7] #SST population
        dx[i,7] = A_GABAs*a_GABAs*sigma(C4*x[i,0], v0) -2*a_GABAs*x[i,7] -a_GABAs**2*x[i,2]

        dx[i,3] = x[i,8] #P2 population
        dx[i,8] = A_AMPA*a_AMPA*sigma(C11*x[i,0]+ C5*x[i,3] + C6*x[i,4]+ I2, v0_p2) -2*a_AMPA*x[i,8] -a_AMPA**2*x[i,3]

        dx[i,4] = x[i,9] #PV population
        dx[i,9] = A_GABAf*a_GABAf*sigma(C12*x[i,0] + C8*x[i,3] + C9*x[i,4], v0) -2*a_GABAf*x[i,9] - a_GABAf**2*x[i,4]    

    return dx.flatten()

t0 = time()
t = np.arange(0, 10, 0.0005)
print(np.shape(X0))
result = odeint(Network_LaNMM,  X0.flatten(), t)
result = np.reshape(result, (len(t), N, 10))
print(np.shape(result))


t0 = time()-t0
print('exeution time: ', t0)
for i in range(N):
    plt.scatter(t, result[:,i,0], label = 'P1 cells of the second pop')
#plt.legend()
plt.show()

print('print time: ', time()-t0)