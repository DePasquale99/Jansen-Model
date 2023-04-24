import numpy as np
from scipy.integrate import solve_ivp
from numba import jit
import matplotlib.pyplot as plt
'''
At the moment I have 2000 points of u(t) for every brain area:
=> I can't have u(t) for every t, and this is a problem for the integrator
SOLUTION => print dense output for u(t) and then use it to integrate 
PROBLEM: I think I'm no more able to integrate the BOLD signal separately :(
OTHER SOLUTION: linearly approx any time t by weihting the two closest t's
'''

#Model parameters:

epsilon = .5 #efficacy with which a neural activity induces an increase in signal
tau_s =  .8 #signal decay time constant
tau_f = .4 #autoregolatory blood flow feedback 
tau_0 = 1 #mean transit time
alpha = .2 #outside flux regolatory parameter
E_0 = 0.8 #Resting net oxygen extraction fraction
V_0 = 0.02 #resting volume of the blood vessel

k1, k2, k3 = 7*E_0, 2, 2*E_0-0.2 #parameters for the BOLD signal

###LOADING OF THE DF (find another way to pass it to the integrator)

data = np.load('Data/results.npy')
N, variables, timepoints = np.shape(data)

#signal is formed by the sum of the PSP of every excitatory population in the column, namely 0,1 and 3
#Only one node is taken for the moment
node = 89
U = data[node,0,:] + data[node,1,:] + data[node,3,:]

print(np.shape(U))

def guess_the_U(t):
    #uses linear approx to find the value of t for a generical t(not multiple of timestep)
    x = U #selects the right cortical column
    t_approx = round(t)#based on the rounding, calculates the linear approx:
    if t_approx >= t:
        if t_approx<len(x)-1:
            frac = t_approx-t
            return x[t_approx]*(1-frac)+x[t_approx-1]*frac
        else:
            return (x[t_approx-1])
    else:

        if t_approx<len(x)-1:
            frac = t - t_approx
            return x[t_approx]*(frac)+x[t_approx+1]*(1-frac)
        else:
            return (x[t_approx])


#outwards flux 
f_out = lambda v: v**(1/alpha)


#effective oxygen extraction fraction
E = lambda f_in: 1 - (1-E_0)**(1/f_in)


#actual bold signal
bold = lambda v,q: V_0*(k1*(1-q) +k2*(1-q/v) +k3*(1-v))


def balloon(t, x):
    '''

    Function for integration of the BOLD signal model; takes:
    t: time of the previous step of integration
    x. state of the system at the previous step of interation
    Returns: dx, vector containing the finite differences between the previous and next step of integration

    '''
    s, f_in, v, q = x

    ds = epsilon*guess_the_U(t) -s/tau_s -(f_in -1)/tau_f

    df_in = s

    dv = f_in - f_out(v)

    dq = f_in*E(f_in)/(E_0*tau_0) - f_out(v)*q/(v*tau_0)

    return np.array([ds, df_in, dv, dq])


def BOLD(x):
    '''

    Non linear function that calculates effective BOLD signal; takes:
    x: (v, q) couples of vessel volume and deoxyemoglobine quantity
    Returns: the simulated BOLD signal

    '''
    v, q = x
    return V_0*(k1*(1-q) +k2*(1-q/v) +k3*(1-v))


def main():
    timestep = 0.01
    t =np.arange(900, 1000, timestep)
    X0 = np.ones((4))*0.01
    
    print(np.shape(X0))
    Y = solve_ivp(balloon, (0, timepoints), X0.flatten(), t_eval=t, vectorized=True ).y

    Y = np.reshape(Y, ( 4, len(t)))

    #extract p and q variables for bold conversion
    X = np.array([Y[2],Y[3]])
    signal = np.apply_along_axis(BOLD, 0, X)

    print(np.shape(signal))

    plt.plot(np.arange(len(t)), signal[:])
    plt.show()


    return

main()








