import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#Goal of this code is to implement Jansen model of a cortical column

e0 = 2.5 #half of the maximum firing rate
v0 = 6 #potential at which half of the maximum firing rate is achieved
r = 0.56 #slope of the sigmoid at v = v0


def sigma(x): 
    #Wave to pulse operator, transforms the average membrane potential of a population into average firing rate
    return 2*e0 / (1+ np.exp(r*(v0- x)))


A =  3.25 #average excitatory synapitc gain
B = 22 #average inhibirtory synaptic gain
C = 135 #model conductance
a = 100 #excitatory synaptic time rate
b = 50 #inhibitory synaptic time rate


#All of the different conductances in the model have been reduced to a single parameter C
C1, C2, C3, C4  = C , .8*C, .25*C, .25*C

#Input parameter of the model, sets the external input as an average firing rate
p = 50

def jansen_model(x,t):
    #implementation of the actual differential equation system
    dx0 = x[3]
    dx3 = A*a*sigma(x[1]-x[2])-2*a*x[3]-a**2*x[0]
    dx1 = x[4]
    dx4 = A*a*(p+C2*sigma(C1*x[0]))-2*a*x[4]-a**2*x[1]
    dx2 = x[5]
    dx5 = B*b*C4*sigma(C3*x[0]) -2*b*x[5] -b**2*x[2]

    dx = [dx0, dx1, dx2, dx3, dx4, dx5]

    return dx



t = np.arange(0, 1, 0.0001)

result = odeint(jansen_model, [0, 0, 0, 0, 0, 0], t)

print(np.shape(result))

Result = odeint(jansen_model, [0.1, 18.0, 12.0, 0, 0, 0], t)

y = result[:,1] - result[:,2]

Y = Result[:,1] - Result[:,2]
print(np.shape(y))
plt.scatter(t, y)
plt.title('Jansen model with p = 50')
plt.scatter(t, Y)
plt.show()