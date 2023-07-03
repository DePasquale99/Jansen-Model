import numpy as np
from sympy_functions import get_Jacobian, get_K, get_LaNMM
from scipy.linalg import eig
from time import time
from numba import jit
import matplotlib.pyplot as plt

#RK4 version that calculates one timepoint each time so I can integrate the two parts of the system separately
#Importing the functions from simpy: WRAPPER NEEDED TO WORK WITH INTEGRATOR

funct = get_LaNMM()
f_Jac =get_Jacobian()
f_K =get_K()

'''
@jit
def LaNMM(t, x):
    y = funct(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9])
    #print('LaNMM output size = ', np.shape(y))
    return y[:,0]'''

@jit
def Jacobian(t, x):
    return f_Jac(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9])

@jit
def K(t,x):
    return f_K(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9])

doog = time()

def main():
    timepoints = np.arange(0, t_end, timestep)
    system = np.zeros((iterations, 10)) #starting point = 0
    msf = np.zeros((lyap_iter, 10)) #setting the starting point = 1/10 (normalized unifom vector?)
    msf[0] = np.ones((1,10))/10
    lyap = 0
    lyap_idx = 0
    for idx, t in enumerate(timepoints[:-1]):
        if (idx%100000 == 0): print('Arrived to s = ', idx/10000, ' in time = ', time()- doog)
        system[idx+1] = rk_4(LaNMM, system[idx], timestep)
        if (idx>transient/timestep):
            #should be in a stable state, so we can start calculating lyapunov exp.
            if (idx%10 == 0):
                #every 10 timesteps, I actually calculate lyap
                d_msf = np.dot(Jacobian(t,system[idx+1]) + epsilon*eigenvalue*K(t,system[idx+1]), msf[lyap_idx])
                msf[lyap_idx+1] = msf[lyap_idx] + d_msf*timestep
                lyap += np.log(np.linalg.norm(msf[lyap_idx + 1]))
                msf[lyap_idx+1] = msf[lyap_idx+1]/ np.linalg.norm(msf[lyap_idx+1])
            else:
                #if not, I just integrate the equations that account for 
                d_msf = np.dot(Jacobian(t,system[idx+1]) + epsilon*eigenvalue*K(t,system[idx+1]), msf[lyap_idx])
                msf[lyap_idx+1] = msf[lyap_idx] + d_msf
                #msf[lyap_idx+1] = msf[lyap_idx+1]/ np.linalg.norm(msf[lyap_idx+1]) 
            lyap_idx += 1

    print('Lyapunov exponent is = ', lyap*10/lyap_iter)
    
    return


#Importing the connectivity matrix and calculating it's eigenvalues

def normalize_data(W):
    #takes the W matrix and normalizes the data by row:
    norm_W = np.zeros(np.shape(W))
    norm = np.sum(W, axis = 1)
    for idx, i in enumerate(norm):
        norm_W[idx] = W[idx]/i
    return norm_W

A = np.load('Data/desikan.npy')
A1 = normalize_data(A)

values, vectors = eig(A1)
values = [float(value) for value in values]
#print(values)
eigenvalue = values[0] #selecting the eigenvalue for MSF
epsilon = 50 #global connectivity parameter

t_end = 1000 #simulation duration
timestep = 0.0001 #integration timestep
tau = 0.001 #Lyapunov exponent timestep
transient = 500
iterations = int(t_end/timestep)
lyap_iter = int((t_end-transient)/timestep)


@jit
def rk_4(func, y0, timestep, args=()):
    h = timestep
    y = np.copy(y0)
    k1 = func(0, y0, *args)
    k2 = func(0+h/2, y0+k1*h/2, *args)
    k3 = func(0+ h/2, y0 +k2*h/2, *args)
    k4 = func(0+ h, y0+k3*h, *args)
    y += (h/6)*(k1 +2*k2 +2*k3 +k4)
    return y



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
p1, p2 = 200, 90 #p1 is noisy in the original model, produced by N(200,30)
#In order to transform from presynaptic firing rate to PSP (fixed point of second order eqt for p1,p2)
I1, I2 = (A_AMPA/a_AMPA)*p1, (A_AMPA/a_AMPA)*p2

def LaNMM(t , x):
    #implementation of the self coupled differential equation system
    dx0 = x[5] #P1 population
    dx5 = A_AMPA*a_AMPA*(sigma(C10*x[3]+C1*x[2]+C0*x[1]+I1 + epsilon*0.5*x[0] , v0))-2*a_AMPA*x[5]-a_AMPA**2*x[0]

    dx1 = x[6] # SS population
    dx6 = A_AMPA*a_AMPA*(sigma(C3*x[0], v0))-2*a_AMPA*x[6]-a_AMPA**2*x[1]

    dx2 = x[7] #SST population
    dx7 = A_GABAs*a_GABAs*sigma(C4*x[0], v0) -2*a_GABAs*x[7] -a_GABAs**2*x[2]

    dx3 = x[8] #P2 population (with self input added)
    dx8 = A_AMPA*a_AMPA*sigma(C11*x[0]+ C5*x[3] + C6*x[4]+ epsilon*.5*x[0] + epsilon*x[3] + I2, v0_p2) -2*a_AMPA*x[8] -a_AMPA**2*x[3]

    dx4 = x[9] #PV population
    dx9 = A_GABAf*a_GABAf*sigma(C12*x[0] + C8*x[3] + C9*x[4], v0) -2*a_GABAf*x[9] - a_GABAf**2*x[4]    

    dx = np.array([dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9])

    return dx

def main2():
        #using non numpy functions

    iterations = 100000
    timestep = .001
    results = np.zeros((iterations,10))
    results[0] = np.random.uniform(size=(10))
    print(type(epsilon))
    for i in range(iterations-1):
        results[i+1] = rk_4(LaNMM, np.array(results[i]), timestep)

    plt.plot(range(int(iterations/10)), results[-int(iterations/10):,3])
    plt.show()



    return


def full_system(t, x):
    #This function calculates both the self coupled system and lyapunov vector all toghether
    y, z = x[:10], x[10:]
    dy = LaNMM(t, y)
    J = Jacobian(t,y) + epsilon*eigenvalue*K(t,y)

    #print(np.shape(J), np.shape(z))
    dz = np.dot(J, z.T)
    #print(np.shape(dy),np.shape(dz))
    return np.append(dy, dz)


A = np.load('Data/desikan.npy')
A1 = normalize_data(A)

values, vectors = eig(A1)
values = [float(value) for value in values]
#print(values)
eigenvalue = values[0] #selecting the eigenvalue for MSF
epsilon = 50 #global connectivity parameter

t_end = 1000 #simulation duration
timestep = 0.001    #integration timestep
tau = 0.01 #Lyapunov exponent timestep
transient = 500
iterations = int(t_end/timestep)
lyap_iter = int((t_end-transient)/timestep)

lyap_period = int(tau/timestep)
#ÐO I NEED TO CODE A TRANSIENT TIME FOR THE SYSTEM TO STABILYZE BEFORE STARTING WITH MSF?


#Declaring gloablly the storage for the data so that the normalization can be done in a different function from the integration one

timepoints = np.arange(0, t_end, timestep)
system = np.zeros((iterations, 10)) #starting point = 0
lyap_vector = np.zeros((iterations, 10)) #setting the starting point = 1/10 (normalized unifom vector?)
lyap_vector[0] = np.ones((1,10))/10

def main3():
    '''
    The idea is to use K and J coming from sympy and LaNMM that works
    '''

    lyap = 0
    n_lyap = 0
    
    for idx in range(iterations-lyap_iter):
        #if (idx%100000 == 0): print('Arrived to s = ', idx*timestep, ' in time = ', time()- doog)
        y0 = np.append(system[idx], lyap_vector[idx])
        result  = rk_4(full_system, y0, timestep)
        system[idx+1], lyap_vector[idx+1] = result[:10], result[10:]
        if(idx%lyap_period == 0):
            lyap_vector[idx+1] = lyap_vector[idx+1]/ np.linalg.norm(lyap_vector[idx+1]) #NORMALIZATION 

    print('Transient time finished, starting to calculate Lyapunov exponents')    

    lyap_values = np.zeros((iterations))

    for idx in range(lyap_iter, iterations-1, 1):
        #if (idx%100000 == 0): print('Arrived to s = ', idx*timestep, ' in time = ', time()- doog)
        y0 = np.append(system[idx], lyap_vector[idx])
        result  = rk_4(full_system, y0, timestep)
        system[idx+1], lyap_vector[idx+1] = result[:10], result[10:]
        if(idx%lyap_period == 0):
            n_lyap +=1
            lyap += np.log(np.linalg.norm(lyap_vector[idx + 1]))
            lyap_values[n_lyap-1] = lyap/(n_lyap*tau)
            lyap_vector[idx+1] = lyap_vector[idx+1]/ np.linalg.norm(lyap_vector[idx+1])#NORMALIZATION 

    print('Lyapunov exponent is = ', lyap/(t_end -transient))
    print('Total time used = ', time() - doog)

    plt.plot(range(n_lyap), lyap_values[:n_lyap])
    plt.title('Lyapunov exponent convergence at tau = ' + str( tau) + ' and timestep = '+ str(timestep))
    plt.show()

    plt.plot(system[-10000:, 3], range(10000))
    plt.show()
    return


main3()

#NEXT: write J and K in a way that they workkkk

def d_sigma(v, v0):


    return 