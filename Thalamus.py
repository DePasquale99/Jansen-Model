
import numpy as np
from scipy.integrate import solve_ivp
from time import time
import matplotlib.pyplot as plt
import py_compile
import cProfile
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

C0, C1, C2 = 10, 20, 30 #respectively connections TRN-> TC, TC->TRN, TC->TC

#Input parameter of the model, sets the external input as an average firing rate
p1= 120 #input to the thalamus main population
#In order to transform from presynaptic firing rate to PSP (fixed point of second order eqt for p1,p2)
I1= (A_AMPA/a_AMPA)*p1


@jit
def thalamus_NMM(t, x):
    dx0 = x[3] #TC population
    dx3 = A_AMPA*a_AMPA*(sigma(C2*x[0]+C0*x[2]+C0*x[1]+I1, v0))-2*a_AMPA*x[3]-a_AMPA**2*x[0]

    dx1 = x[4] #TRN2 population
    dx4 = A_GABAs*a_GABAs*sigma(C1*x[0], v0) -2*a_GABAs*x[4] -a_GABAs**2*x[1]

    dx2 = x[5] #TRN1 population
    dx5 = A_GABAf*a_GABAf*sigma(C1*x[0] , v0) -2*a_GABAf*x[5] - a_GABAf**2*x[2]    

    dx = np.array([dx0, dx1, dx2, dx3, dx4, dx5])

    return dx.flatten()



def main():
    #executes the iteration using solve ivp and RK45
    X0 = np.array([.1,.1,.1,0,0,0])
    t0 = time()
    timestep = 0.001
    t_eval =np.arange(990, 1000, timestep)
    result = solve_ivp(thalamus_NMM, [0, 1000], X0.flatten(), t_eval=t_eval)
    t0 = time()-t0
    print('execution time: ', t0)

    Y = np.reshape(result.y, ( 6,len(t_eval)))

    plt.plot(t_eval, Y[0,:])
    plt.show()
    
    plt.plot(t_eval, Y[1,:])
    plt.show()

    plt.plot(t_eval, Y[2,:])
    plt.show()
    np.save('Data/thalamus', Y)
    return

#cProfile.run('main()')

main()