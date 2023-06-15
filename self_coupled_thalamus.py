
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#Goal of this code is to implement self coupled LaNMM as it works in the homogeneus manifold
# => it still depends on epsilon

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

T0, T1, T2 = 10, 20, 30 #respectively connections TRN-> TC, TC->TRN, TC->TC

#Input parameter of the model, sets the external input as an average firing rate
p1, p2 = 200, 150 #p1 is noisy in the original model, produced by N(200,30)
#In order to transform from presynaptic firing rate to PSP (fixed point of second order eqt for p1,p2)
I1, I2 = (A_AMPA/a_AMPA)*p1, (A_AMPA/a_AMPA)*p2

epsilon = 50 #global connectivity
epsilon_t = 20 #connectivity between thalamus and cortex

def LaNMM(t, x, epsilon):
    #implementation of the actual differential equation system
    dx0 = x[5] #P1 population
    dx5 = A_AMPA*a_AMPA*(sigma(C10*x[3]+C1*x[2]+C0*x[1]+I1 + epsilon*0.5*x[0] , v0))-2*a_AMPA*x[5]-a_AMPA**2*x[0]

    dx1 = x[6] # SS population
    dx6 = A_AMPA*a_AMPA*(sigma(C3*x[0], v0))-2*a_AMPA*x[6]-a_AMPA**2*x[1]

    dx2 = x[7] #SST population
    dx7 = A_GABAs*a_GABAs*sigma(C4*x[0], v0) -2*a_GABAs*x[7] -a_GABAs**2*x[2]

    dx3 = x[8] #P2 population (with self input added)
    dx8 = A_AMPA*a_AMPA*sigma(C11*x[0]+ C5*x[3] + C6*x[4]+ epsilon*.5*x[0] + epsilon*x[3]+ I2, v0_p2) -2*a_AMPA*x[8] -a_AMPA**2*x[3]

    dx4 = x[9] #PV population
    dx9 = A_GABAf*a_GABAf*sigma(C12*x[0] + C8*x[3] + C9*x[4], v0) -2*a_GABAf*x[9] - a_GABAf**2*x[4]    

    dx = [dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9]

    return dx


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

def full_system(t, x, epsilon, epsilon_t):
    

    #separation of thevariables into the two subsystems

    hot = x[:6]
    cortex = x[6:]

    #adding the self coupled input for the thalamus as 7th variable of the system
    hot = np.append(hot, epsilon_t*cortex[0])

    x_t = ho_thalamus(hot)
    x_c = LaNMM(t, cortex, epsilon)

    return np.append(x_t, x_c)












    return

def main():
    t_eval = np.arange(900, 1000, 0.001)

    result = solve_ivp(full_system, (0, 1000) ,  [.1,0,0,.01,0,0,0,0,0,0, 0, 0, 0, 0, 0, 0], t_eval=t_eval, args=(epsilon, epsilon_t)).y

    print(np.shape(result))

    plt.scatter(t_eval, result[:,0], label = 'P1 potential')

    #plt.scatter(t, result[:,1], label = 'SS potential')

    #plt.scatter(t, result[:,2], label = 'SST potential')

    plt.scatter(t_eval, result[:,3], label = 'P2 potential')

    #plt.scatter(t, result[:,4], label = 'PV potential')
    #plt.title('LaNMM model with p1 = ', p1,', p2 = ', p2)
    plt.legend()
    plt.show()
    return 




def simulation(epsilon):
    '''
    This function simulates the homogeneus state of the system for a given value of the global connectivity epsilon:
    takes: global connectivity (epsilon)
    prints the psp for the two main piramidal populations
    '''
    t_eval = np.arange(900, 1000, 0.001)

    result = solve_ivp(full_system, (0, 1000) ,  [.1,0,0,.01,0,0,0,0,0,0, 0, 0, 0, 0, 0, 0], t_eval=t_eval, args=(epsilon, epsilon_t)).y

    print(np.shape(result))

    plt.scatter(t_eval, result[:,0], label = 'P1 potential')

    #plt.scatter(t, result[:,1], label = 'SS potential')

    #plt.scatter(t, result[:,2], label = 'SST potential')

    plt.scatter(t_eval, result[:,3], label = 'P2 potential')

    #plt.scatter(t, result[:,4], label = 'PV potential')
    #plt.title('LaNMM model with p1 = ', p1,', p2 = ', p2)
    plt.legend()
    plt.show()
    return 


