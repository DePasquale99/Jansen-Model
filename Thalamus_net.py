
import numpy as np
from scipy.integrate import solve_ivp
from time import time
import matplotlib.pyplot as plt
import py_compile
import cProfile
from numba import jit



def ho_thalamus(x):


    return 


def lo_thalamus(x):

    return

def LaNMM_net():

    return


def full_net(t, x):
    hot = x[:6] #High order thalamus variables
    lot = x[6:N/2*6 + 6].reshape((N/2, 6)) #low order thalamus variables
    cortex = x[N/2*6 + 6:].reshape((N,10)) #cortex variables

    W = np.zeros((90, 90)) #weights to convert from p1 to hot input (this part has to be fixed)
    hot_input = np.dot(W, cortex[:,0])
    hot = np.append(hot, np.transpose([hot_input]), axis= 1) #add the input for high order thalamus as the seventh variable



    return



def main():


    return