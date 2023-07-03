import numpy as np
from sympy_functions import get_Jacobian, get_K, get_LaNMM
from scipy.linalg import eig
from time import time
from numba import jit
import matplotlib.pyplot as plt

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
timestep = 0.001    #integration timestep
tau = 0.01 #Lyapunov exponent timestep
transient = 800
iterations = int(t_end/timestep)
lyap_iter = int((t_end-transient)/timestep)

lyap_period = int(tau/timestep)
#ÐO I NEED TO CODE A TRANSIENT TIME FOR THE SYSTEM TO STABILYZE BEFORE STARTING WITH MSF?


#Declaring gloablly the storage for the data so that the normalization can be done in a different function from the integration one

timepoints = np.arange(0, t_end, timestep)
system = np.zeros((iterations, 10)) #starting point = 0
lyap_vector = np.zeros((iterations, 10)) #setting the starting point = 1/10 (normalized unifom vector?)
lyap_vector[0] = np.ones((1,10))/10



#RK4 version that calculates one timepoint each time so I can integrate the two parts of the system separately
@jit
def rk_4(func, y0, timestep, args=()):
    h = timestep
    y = np.copy(y0)
    k1 = func(0, y0, *args)
    k2 = func(0+h/2, y0+k1*h/2, *args)
    k3 = func(0+ h/2, y0 +k2*h/2, *args)
    k4 = func(0+ h, y0+k3*h, *args)
    y += (h/6)*(k1 +2*k2 +2*k3 +k4)
    #print(np.shape(y))
    return y

#Importing the functions from simpy: WRAPPER NEEDED TO WORK WITH INTEGRATOR

f_LaNMM = get_LaNMM()
f_Jac =get_Jacobian()
f_K =get_K()

def LaNMM(t, x):
    y =f_LaNMM(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9])[:,0]
    return y

def Jacobian(t, x):
    return f_Jac(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9])

def K(t,x):
    return f_K(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9])

@jit
def full_system(t, x):
    #This function calculates both the self coupled system and it's deviation
    y, z = x[:10], x[10:]
    dy = f_LaNMM(y[0],y[1],y[2],y[3],y[4],y[5],y[6],y[7],y[8],y[9])[:,0]
    J = Jacobian(t,y) + epsilon*eigenvalue*K(t,y)
    dz = np.dot(J, z)
    return np.append(dy, dz)

doog = time()



def main2():

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
                    #Tryina fix the different timesteps giving different convergences
    for idx in range(iterations-lyap_iter, iterations-1, 1):
        #if (idx%100000 == 0): print('Arrived to s = ', idx*timestep, ' in time = ', time()- doog)
        y0 = np.append(system[idx], lyap_vector[idx])
        result  = rk_4(full_system, y0, timestep)
        system[idx+1], lyap_vector[idx+1] = result[:10], result[10:]
        if(idx%lyap_period == 0):
            n_lyap +=1
            lyap += np.log(np.linalg.norm(lyap_vector[idx + 1]))
            lyap_values[n_lyap-1] = lyap/(n_lyap*tau)
            lyap_vector[idx+1] = lyap_vector[idx+1]/ np.linalg.norm(lyap_vector[idx+1])#NORMALIZATION 

    print((n_lyap*tau), ' = ', (t_end -transient))
    print('Lyapunov exponent is = ', lyap/(t_end -transient), ' = ', lyap_values[n_lyap-1])
    print('Total time used = ', time() - doog)

    plt.plot(range(n_lyap), lyap_values[:n_lyap])
    plt.title('Lyapunov exponent convergence at tau = ' + str( tau) + ' and timestep = '+ str(timestep))
    plt.show()


    plt.plot(np.arange(90, 100, 0.001), system[-10000:,3])
    plt.show()

from scipy.integrate import solve_ivp

def simulation():
    #checks if the self coupled model is working
    
    iterations = 100000
    timestep = .001
    results = np.zeros((iterations,10))
    results[0] = np.random.uniform(size=(10))
    
    for i in range(iterations-1):
        results[i+1] = rk_4(LaNMM, results[i], timestep)

    plt.plot(range(int(iterations/10)), results[-int(iterations/10):,3])
    plt.show()

    '''
    y0 = np.random.uniform(size=(10))

    results = solve_ivp(LaNMM, (0, 100), y0, t_eval=np.arange(90, 100, 0.001)).y
    results = np.reshape(results, (10, len(np.arange(90, 100, 0.001))))


    plt.plot(np.arange(90, 100, 0.001), results[0])
    plt.show()'''

    return

    
main2()
#simulation()






