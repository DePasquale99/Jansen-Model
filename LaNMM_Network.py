#Goal : to implement a network of n regions modelled by LaNMM modeland connect them via a matrix W

import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from time import time
import matplotlib.pyplot as plt
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
    """
    Reads a data file that contains connectivity information between regions and returns the connectivity matrix.

    Parameters:
    data_file_path (str): the path to the data file.

    Returns:
    W (np.ndarray): a numpy array that represents the connectivity matrix between regions.
    """
    data = np.genfromtxt(data_address,
                     skip_header=1,
                     names=True,
                     dtype=None,
                     delimiter=' ')
    
    W = np.zeros((N,N))
    for row in data:
        i, j, w = row
        W[i,j] = w

    np.save('Data/data', W)
        
    return W


#P1 connection matrix copied from Jansen network
#read_W('Data/data.dat')
W = np.load('Data/data.npy')
#initial conditions: (0.2, 0) for every neural pop
X0 = np.append(np.ones((N, 5))*0.2,np.zeros((N, 5)), axis = 1 )
dx = np.zeros((N, 10))

def LaNMM(x, t=0):
    #Modified function for iteration in the network
    dx0 = x[5] #P1 population
    dx5 = A_AMPA*a_AMPA*(sigma(C10*x[3]+C1*x[2]+C0*x[1]+I1 + x[10], v0))-2*a_AMPA*x[5]-a_AMPA**2*x[0]

    dx1 = x[6] # SS population
    dx6 = A_AMPA*a_AMPA*(sigma(C3*x[0], v0))-2*a_AMPA*x[6]-a_AMPA**2*x[1]

    dx2 = x[7] #SST population
    dx7 = A_GABAs*a_GABAs*sigma(C4*x[0], v0) -2*a_GABAs*x[7] -a_GABAs**2*x[2]

    dx3 = x[8] #P2 population
    dx8 = A_AMPA*a_AMPA*sigma(C11*x[0]+ C5*x[3] + C6*x[4]+ I2, v0_p2) -2*a_AMPA*x[8] -a_AMPA**2*x[3]

    dx4 = x[9] #PV population
    dx9 = A_GABAf*a_GABAf*sigma(C12*x[0] + C8*x[3] + C9*x[4], v0) -2*a_GABAf*x[9] - a_GABAf**2*x[4]    

    dx = np.array([dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9])

    return dx



def Network_LaNMM(t,x):
    """
    Simulates the LaNMM network model.

    Parameters:
    x (np.ndarray): a numpy array that contains the initial conditions for the model.
    t (np.ndarray): a numpy array that contains the time points at which the simulation should be evaluated.

    Returns:
    dx.flatten() (np.ndarray): a flattened numpy array that contains the state of the model at each time point.
    """
    x = x.reshape((N, 10))
    ext_p1 = epsilon*np.dot(W, x[:, 0])
    x = np.append(x, np.transpose([ext_p1]), axis= 1) #add the input as 11th variable of the system, it should be automatically removed by odeint

    #iteration with numpy:
    dx = np.apply_along_axis(LaNMM, 1, x)

    return dx.flatten()

t0 = time()
timestep = 0.001
t_eval =np.arange(998, 1000, timestep)
result = solve_ivp(Network_LaNMM, [0, 1000], X0.flatten(), t_eval=t_eval)
t0 = time()-t0
print('exeution time: ', t0)

Y = np.reshape(result.y, ( N, 10,len(t_eval)))


plt.scatter(t_eval, Y[0,0,:])
plt.show()



np.save('Data/results', Y)#discard the initial condition point