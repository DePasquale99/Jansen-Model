import numpy as np
from sympy_functions import get_Jacobian, get_K, get_LaNMM
from scipy.linalg import eig


#RK4 version that calculates one timepoint each time so I can integrate the two parts of the system separately

def rk_4(func, y0, timestep, args=()):
    h = timestep
    y = np.zeros(np.shape(y0))
    k1 = func(0, y0, *args)
    k2 = func(0+h/2, y0+k1*h/2, *args)
    k3 = func(0+ h/2, y0 +k2*h/2, *args)
    k4 = func(0+ h, y0+k3*h, *args)
    y= y0 + (h/6)*(k1 +2*k2 +2*k3 +k4)
    #print(np.shape(y))
    return y

#Importing the functions from simpy: WRAPPER NEEDED TO WORK WITH INTEGRATOR
def LaNMM(t, x):
    funct = get_LaNMM()
    y = funct(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9])
    #print('LaNMM output size = ', np.shape(y))
    return y[:,0]

def Jacobian(t, x):
    return get_Jacobian()(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9])

def K(t,x):
    return get_K()(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9])

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
eigenvalue = values[0] #selecting the eigenvalue for MSF
epsilon = 50 #global connectivity parameter

t_end = 1000 #simulation duration
timestep = 0.0001 #integration timestep
tau = 0.001 #Lyapunov exponent timestep
transient = 900
iterations = int(t_end/timestep)
lyap_iter = int((t_end-transient)/timestep)
#ÐO I NEED TO CODE A TRANSIENT TIME FOR THE SYSTEM TO STABILYZE BEFORE STARTING WITH MSF?




def main():
    timepoints = np.arange(0, t_end, timestep)
    system = np.zeros((iterations, 10)) #starting point = 0
    msf = np.zeros((lyap_iter, 10)) #setting the starting point = 1/10 (normalized unifom vector?)
    msf[0] = np.ones((1,10))/10
    lyap = 0
    for idx, t in enumerate(timepoints):
        system[idx+1] = rk_4(LaNMM, system[idx], timestep)
        if (idx>transient/timestep):
            #should be in a stable state, so we can start calculating lyapunov exp.
            if (idx%10 == 0):
                #every 10 timesteps, I actually calculate lyap
                d_msf = np.dot(Jacobian(t,system[idx]) + epsilon*eigenvalue*K(t,system[idx]), msf[idx])
                msf[idx+1] = msf[idx] + d_msf
                lyap += np.log(np.linalg.norm(msf[idx + 1]))
                msf[idx+1] = msf[idx+1]/ np.linalg.norm(msf[idx+1])
            else:
                #if not, I just integrate the equations that account for 
                d_msf = np.dot(Jacobian(t,system[idx]) + epsilon*eigenvalue*K(t,system[idx]), msf[idx])
                msf[idx+1] = msf[idx] + d_msf
                msf[idx+1] = msf[idx+1]/ np.linalg.norm(msf[idx+1])

    print('Lyapunov exponent is = ', lyap*10/lyap_iter)
    
    return

main()