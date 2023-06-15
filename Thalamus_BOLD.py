
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
p1, p2, th1 = 200, 90, 120 #p1 is noisy in the original model, produced by N(200,30)
#In order to transform from presynaptic firing rate to PSP (fixed point of second order eqt for p1,p2)
I1, I2, It = (A_AMPA/a_AMPA)*p1, (A_AMPA/a_AMPA)*p2, (A_AMPA/a_AMPA)*th1


############################################### Actual network model
epsilon = 100 #cross region connectivity
epsilon_t = 50
N = 68 #Number of regions


def normalize_data(W):
    #takes the W matrix and normalizes the data by row:
    norm_W = np.zeros(np.shape(W))
    norm = np.sum(W, axis = 1)
    for idx, i in enumerate(norm):
        norm_W[idx] = W[idx]/i
    return norm_W




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
def LaNMM(x, I1, I2, t=0):
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



def full_net(t, x, I1, I2, epsilon, epsilon_t):
    '''
    Function that integrates the full network; divided in 2 subroutines:
        -HO Thalamus
        -Cortex
    It first separates the variables of the system into the two different systems, it htan calculates the inputs coming 
    from the other nodes and then calculates the finite differences.
    Parameters:
    t: integration time
    x: state variables of the system
    I1: input current for the pyramidal cell population P1
    I2: input current for the pyramidal cell population P2
    epsilon: global coupling parameter, it scales all of the connections in the network
    '''

    hot = x[:6] #High order thalamus variables
    cortex = x[6:].reshape((N,10)) #cortex variables


    D = W[-1] #weights to convert from p1 to hot input
    hot_input = epsilon*np.dot(D[:-1], cortex[:,0])
    hot = np.append(hot, np.transpose([hot_input]), axis= 0) #add the input for high order thalamus as the seventh variable
    d_hot = ho_thalamus(hot) #calculate the increments for HO thalamus


    ext_p1 = .5*epsilon*np.dot(W[:68, :68], cortex[:, 0]) + epsilon_t*hot[0]
    ext_p2 = .5*epsilon*np.dot(W[:68, :68], cortex[:,0]) + epsilon*np.dot(W[:68, :68], cortex[:,3])

    cortex = np.append(cortex, np.transpose([ext_p1]), axis= 1) #add the input as 11th variable of the system
    cortex =np.append(cortex, np.transpose([ext_p2]), axis = 1) #same but for p2
    d_cortex = np.apply_along_axis(LaNMM, 1, cortex, I1, I2).flatten()
    #print(np.shape(d_cortex), np.shape(d_hot))


    #put everything toghether and returns
    return np.append(d_hot, d_cortex)



def main():
    # set the initial conditions:

    X0 = np.ones((N*10 + 6))*0.1
    t_eval = np.arange(900, 1000, 0.001)
    system = solve_ivp(full_net, [0, 1000], X0, t_eval=t_eval, args=(I1, I2, epsilon, epsilon_t))

    data = system.y.reshape((N*10+6, len(t_eval)))
    

    np.save('Data/thalamus_output', data)


    return


#####BALLOON MODEL: directly converts the output to the expected BOLD signal

#Model parameters:

eff = .5 #efficacy with which a neural activity induces an increase in signal
tau_s =  .8 #signal decay time constant
tau_f = .4 #autoregolatory blood flow feedback 
tau_0 = 1 #mean transit time
alpha = .2 #outside flux regolatory parameter
E_0 = 0.8 #Resting net oxygen extraction fraction
V_0 = 0.02 #resting volume of the blood vessel

k1, k2, k3 = 7*E_0, 2, 2*E_0-0.2 #parameters for the BOLD signal



def f_out(v):
    #flux towards the outside of the blood vessel
    return v**(1/alpha)



def E(f_in): 
    #effective oxygen extraction fraction
    return 1 - (1-E_0)**(1/f_in)


def BOLD(x):
    '''

    Non linear function that calculates effective BOLD signal; takes:
    x: (v, q) couples of vessel volume and deoxyemoglobine quantity
    Returns: the simulated BOLD signal

    '''
    v, q = x
    return V_0*(k1*(1-q) +k2*(1-q/v) +k3*(1-v))


def balloon(x):
    '''

    Function for integration of the BOLD signal model; takes:
    t: time of the previous step of integration
    x. state of the system at the previous step of interation
    Returns: dx, vector containing the finite differences between the previous and next step of integration

    '''
    s, f_in, v, q, u = x

    ds = epsilon*u -s/tau_s -(f_in -1)/tau_f

    df_in = s

    dv = f_in - f_out(v)

    dq = f_in*E(f_in)/(E_0*tau_0) - f_out(v)*q/(v*tau_0)

    return np.array([ds, df_in, dv, dq])




def simulation(p1, p2, epsilon, epsilon_t):
    '''
    simulation function, takes as arguments the three analyzed variables for the system:
    the two neural columns external inputs, p1 and p2, and global coupling epsilon. 

    Saves the BOLD results in a folder, named after the specific parameters of the simulation.

    returns the relative path of the file in which results are stored
    '''
    filename = 'simulation_' + str(p1) + '_' + str(p2) + '_' + str(epsilon)

    path = 'Data/pipeline/' + filename

    I1, I2 = (A_AMPA/a_AMPA)*p1, (A_AMPA/a_AMPA)*p2
    X0 = np.ones((N*10 + 6))*0.1

    timestep = 0.01
    t_eval =np.arange(900, 1000, timestep)
    result = solve_ivp(full_net, [0, 1000], X0.flatten(), t_eval=t_eval, dense_output=True, args=(I1, I2, epsilon, epsilon_t))

    
    def PSP(t):
        #function that calculates the total piramidal PSP for a given time t
        full_sol = result.sol(t)
        full_sol = full_sol[6:] #discard thalamus for BOLD signaling
        x = np.zeros((N))
        for i in range(N):
            x[i] = full_sol[i*10]+full_sol[i*10+1] + full_sol[i*10+3]
        
        return x

    def balloon_continue(t, x):
        '''

        Function for integration of the BOLD signal model; takes:
        t: time of the previous step of integration
        x. state of the system at the previous step of interation
        Returns: dx, vector containing the finite differences between the previous and next step of integration
        '''
        U = PSP(t)
        X = np.reshape(x, (N,4))
        X = np.append(X, np.transpose([U]), axis= 1)

        dx = np.apply_along_axis(balloon, 1, X)

        return dx.flatten()
    
    x0 = np.ones((4,N))*0.01
    #integration using solve ivp:
    t_BOLD = np.arange(900, 1000, 0.01)
    solution = solve_ivp(balloon_continue, [900, 1000], x0.flatten(), t_eval = t_BOLD )

    loon = np.reshape(solution.y, ( N, 4,len(t_BOLD))) #this is all of the 4 variables
    
    print('Balloon integration done')

    X = np.array([loon[:,2,:], loon[:,3,:]])
    signal = np.apply_along_axis(BOLD, 0, X)
    print(np.shape(signal))

    np.save(path, signal)

    return path