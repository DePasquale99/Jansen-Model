import numpy as np
import sympy as smp
from numba import jit

'''
Goal of this code is to create the LaNMM model equations in sympy and export them in numpy
'''

###Parameters for LaNMM model

e0 = 2.5 #half of the maximum firing rate (Hz)
v0 = 6 #potential at which half of the maximum firing rate is achieved (mV)
v0_p2 = 1
r = 0.56 #slope of the sigmoid at v = v0 (mV^-1)

A_AMPA, A_GABAs, A_GABAf =  3.25, -22, -30 #average synapitc gain (mV)
a_AMPA, a_GABAs, a_GABAf = 100, 50, 220 #excitatory synaptic time rate (s^-1)

C0, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12 = 108, 33.7, 1, 135, 33.75, 70, 550, 1, 200, 100, 80, 200, 30

#Input parameter of the model, sets the external input as an average firing rate
p1, p2 = 200, 90 #p1 is noisy in the original model, produced by N(200,30)
#In order to transform from presynaptic firing rate to PSP (fixed point of second order eqt for p1,p2)
I1, I2 = (A_AMPA/a_AMPA)*p1, (A_AMPA/a_AMPA)*p2
epsilon = 50

#defining sigmoid function

v, v_0 = smp.symbols('v v_0')
sigma = smp.symbols('\sigma', cls = smp.Function)
sigma = sigma(v, v_0)
sigma = 2*e0/(1+smp.exp(r*(v_0-v)))


#Defining LaNMM equations


x0, x1, x2, x3, x4, x5, x6, x7, x8, x9 = smp.symbols('x:10')

#Writing the differential equations that describe the evolution of the system
y0 = x5
y1 = x6
y2 = x7
y3 = x8
y4 = x9

sigma_P1 = sigma.subs([(v, C10*x3+C1*x2+C0*x1+I1 + epsilon*0.5*x0), (v_0, v0)]) #SELF COUPLED VERSION
#sigma_P1 = sigma.subs([(v, C10*x3+C1*x2+C0*x1+I1 ), (v_0, v0)]) #UNCOUPLED
y5 = A_AMPA*a_AMPA*(sigma_P1)-2*a_AMPA*x5-a_AMPA**2*x0

sigma_rPN = sigma.subs([(v, C3*x0), (v_0, v0)])
y6 = A_AMPA*a_AMPA*(sigma_rPN)-2*a_AMPA*x6-a_AMPA**2*x1


sigma_IN = sigma.subs([(v, C4*x0), (v_0, v0)])
y7 = A_GABAs*a_GABAs*(sigma_IN)-2*a_GABAs*x7-a_GABAs**2*x2


sigma_P2 = sigma.subs([(v, C11*x0+ C5*x3 + C6*x4 + I2 +epsilon*0.5*x0 + epsilon*x3), (v_0, v0_p2)]) #SELF COUPLED VERSION
#sigma_P2 = sigma.subs([(v, C11*x0+ C5*x3 + C6*x4 + I2), (v_0, v0_p2)]) #STANDALONE VERSION
y8 = A_AMPA*a_AMPA*(sigma_P2)-2*a_AMPA*x8-a_AMPA**2*x3


sigma_PV = sigma.subs([(v, C12*x0 + C8*x3 + C9*x4), (v_0, v0)])
y9 = A_GABAf*a_GABAf*(sigma_PV)-2*a_GABAf*x9-a_GABAf**2*x4


#putting all of the equations in a vector
Y = smp.Matrix([y0,y1,y2,y3,y4,y5,y6,y7,y8,y9])

#Building the jacobian
Jac = smp.Matrix([Y.diff(x0).T, Y.diff(x1).T, Y.diff(x2).T, Y.diff(x3).T, Y.diff(x4).T, Y.diff(x5).T, Y.diff(x6).T, Y.diff(x7).T, Y.diff(x8).T, Y.diff(x9).T ]).T

def get_LaNMM():
    return smp.lambdify([x0,x1,x2,x3,x4,x5,x6,x7,x8,x9], Y)


def get_Jacobian():
    return smp.lambdify([x0,x1,x2,x3,x4,x5,x6,x7,x8,x9], Jac)


#Building K matrix
K = smp.zeros(10,10)
d_sigma = sigma.diff(v) 


K[5,0] = (1/2)*a_AMPA*A_AMPA/2*d_sigma.subs([(v, C10*x3+C1*x2+C0*x1+I1 + epsilon/2*x0), (v_0, v0)])


K[8,0] =  (1/2)*a_AMPA*A_AMPA/2*d_sigma.subs([(v, C11*x0+ C5*x3 + C6*x4 + I2 +epsilon/2*x0 + epsilon*x3), (v_0, v0_p2)])


K[8,3] =  a_AMPA*A_AMPA*d_sigma.subs([(v, C11*x0+ C5*x3 + C6*x4 + I2 +epsilon/2*x0 + epsilon*x3), (v_0, v0_p2)])

def get_K():
    return smp.lambdify([x0,x1,x2,x3,x4,x5,x6,x7,x8,x9], K)

