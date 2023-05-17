
import numpy as np
from scipy.integrate import solve_ivp
from time import time
from numba import jit


###Parameters for LaNMM model

e0 = 2.5 #half of the maximum firing rate (Hz)
v0 = 6 #potential at which half of the maximum firing rate is achieved (mV)
v0_p2 = 1
r = 0.56 #slope of the sigmoid at v = v0 (mV^-1)

@jit
def sigma(x, v0) -> float: 
    #Wave to pulse operator, transforms the average membrane potential of a population into average firing rate
    return 2*e0 / (1+ np.exp(r*(v0- x)))


A_AMPA, A_GABAs, A_GABAf =  3.25, -22, -30 #average synapitc gain (mV)
a_AMPA, a_GABAs, a_GABAf = 100, 50, 220 #excitatory synaptic time rate (s^-1)

T0, T1, T2 = 10, 20, 30 #respectively connections TRN-> TC, TC->TRN, TC->TC

C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12 = 108, 33.7, 1, 135, 33.75, 70, 550, 1, 200, 100, 80, 200, 30

#Input parameter of the model, sets the external input as an average firing rate
p1, p2, th1 = 200, 150, 120 #p1 is noisy in the original model, produced by N(200,30)
#In order to transform from presynaptic firing rate to PSP (fixed point of second order eqt for p1,p2)
I1, I2, It = (A_AMPA/a_AMPA)*p1, (A_AMPA/a_AMPA)*p2, (A_AMPA/a_AMPA)*th1


############################################### Actual network model
epsilon = 50 #cross region connectivity
N = 68 #Number of regions


def normalize_data(W):
    #takes the W matrix and normalizes the data by row:
    norm = np.sum(W, axis = 1)
    for idx, i in enumerate(norm):
        W[idx] = W[idx]/i
    return W




#P1 connection matrix copied from Jansen network
#read_W('Data/data.dat')
W = np.load('Data/thalamus_matrix.npy')
W = normalize_data(W)


def ho_thalamus(x):
    '''
    Function that calculates the differential equations for HO thalamus
    asks for x: vector containing the three populations potentials and relative dervivatives, and input coming from p1s
    returns dx: vector containing increments for the 6 state variables of HO thalamus
    '''

    dx0 = x[3] #TC population
    dx3 = A_AMPA*a_AMPA*(sigma(T2*x[0]+T0*x[2]+T0*x[1]+x[6], v0))-2*a_AMPA*x[3]-a_AMPA**2*x[0]

    dx1 = x[4] #TRN2 population
    dx4 = A_GABAs*a_GABAs*sigma(T1*x[0], v0) -2*a_GABAs*x[4] -a_GABAs**2*x[1]

    dx2 = x[5] #TRN1 population
    dx5 = A_GABAf*a_GABAf*sigma(T1*x[0] , v0) -2*a_GABAf*x[5] - a_GABAf**2*x[2]    

    dx = np.array([dx0, dx1, dx2, dx3, dx4, dx5])



    return dx


@jit
def LaNMM(x, t=0):
    #Modified function for iteration in the network
    dx0 = x[5] #P1 population
    dx5 = A_AMPA*a_AMPA*(sigma(C10*x[3]+C1*x[2]+C0*x[1]+I1 + x[10], v0))-2*a_AMPA*x[5]-a_AMPA**2*x[0]

    dx1 = x[6] # SS population
    dx6 = A_AMPA*a_AMPA*(sigma(C3*x[0], v0))-2*a_AMPA*x[6]-a_AMPA**2*x[1]

    dx2 = x[7] #SST population
    dx7 = A_GABAs*a_GABAs*sigma(C4*x[0], v0) -2*a_GABAs*x[7] -a_GABAs**2*x[2]

    dx3 = x[8] #P2 population
    dx8 = A_AMPA*a_AMPA*sigma(C11*x[0]+ C5*x[3] + C6*x[4]+ I2 + x[11], v0_p2) -2*a_AMPA*x[8] -a_AMPA**2*x[3]

    dx4 = x[9] #PV population
    dx9 = A_GABAf*a_GABAf*sigma(C12*x[0] + C8*x[3] + C9*x[4], v0) -2*a_GABAf*x[9] - a_GABAf**2*x[4]    

    dx = np.array([dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9])

    return dx



def full_net(t, x):

    hot = x[:6] #High order thalamus variables
    cortex = x[6:].reshape((N,10)) #cortex variables


    D = W[-1] #weights to convert from p1 to hot input
    hot_input = np.dot(D[:-1], cortex[:,0])
    hot = np.append(hot, np.transpose([hot_input]), axis= 0) #add the input for high order thalamus as the seventh variable
    d_hot = ho_thalamus(hot) #calculate the increments for HO thalamus


    ext_p1 = .5*epsilon*np.dot(W[:68, :68], cortex[:, 0]) 
    ext_p2 = .5*epsilon*np.dot(W[:68, :68], cortex[:,0]) + epsilon*np.dot(W[:68, :68], cortex[:,3])

    cortex = np.append(cortex, np.transpose([ext_p1]), axis= 1) #add the input as 11th variable of the system
    cortex =np.append(cortex, np.transpose([ext_p2]), axis = 1) #same but for p2
    d_cortex = np.apply_along_axis(LaNMM, 1, cortex).flatten()
    #print(np.shape(d_cortex), np.shape(d_hot))


    #put everything toghether and returns
    return np.append(d_hot, d_cortex)



def main():
    # set the initial conditions:

    X0 = np.ones((N*10 + 6))*0.1
    t_eval = np.arange(900, 1000, 0.001)
    system = solve_ivp(full_net, [0, 1000], X0, t_eval=t_eval)

    data = system.y.reshape((N*10+6, len(t_eval)))
    

    np.save('Data/thalamus_output', data)


    return
t0 = time()
main()
print('integration finished, time used = ', time()- t0)