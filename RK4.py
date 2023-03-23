import numpy as np
import matplotlib.pyplot as plt


def RK4(func, y0, timepoints, args=()):
    y = np.zeros((len(timepoints), len(y0)))
    y[0] = y0
    for i in range(len(timepoints)-1):
        h = timepoints[i+1]-timepoints[i]
        k1 = func(t[i], y[i], *args)
        k2 = func(t[i]+h/2, y[i]+k1*h/2, *args)
        k3 = func(t[i]+ h/2, y[i] +k2*h/2, *args)
        k4 = func(t[i]+ h, y[i]+k3*h, *args)
        y[i+1] = y[i] + (h/6)*(k1 +2*k2 +2*k3 +k4)
    return y

def volterra_lotka(t, y, a=1.5, b=1, c=3, d=1):
    y1, y2 = y
    return np.array([a*y1 -b*y1*y2, -c*y2 + d*y1*y2])


t = np.arange(0, 10, 0.01)
y0 = [10, 5] 

sol = RK4(volterra_lotka, y0, t)

plt.scatter(t, sol[:,0], label= 'prey population')
plt.scatter(t, sol[:,1], label = 'predator population')
plt.legend()
plt.show()



