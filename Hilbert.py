from scipy.signal import hilbert
import numpy as np
import matplotlib.pyplot as plt


#Goal: understand the effect of hilbert transform, calculate Kuramoto parameter


data = np.load('Data/results.npy')
#at the moment I'm transforming signal from each neuron in a separate way
N, variables, timepoints = np.shape(data)


#Preprocessing: each time series is centered around zero (doesn't work so well)

mean = np.mean(data, 2)
print(np.shape(mean))

for i in range(N):
    for j in range(variables):
        data[i,j] -= mean[i,j]


transform = hilbert(data)
amplitude = np.abs(transform)


plt.subplot(211)
plt.plot(np.arange(0, 2, 0.001), data[5,0,:], label = 'original data')
plt.plot(np.arange(0, 2, 0.001), amplitude[5,0,:], label = 'hilbert amplitude')
plt.legend()
plt.subplot(212)
#plt.title('hilbert amplitude')
plt.plot(np.arange(0, 2, 0.001), data[5,3,:], label = 'hilbert amplitude')
plt.show()


def kuramoto_r(data):
    #calculates kuramoto phase synchronization parameter for each time step: takes 2D Hilbert transform as input,
    #in which first axis iterates through different nodes, and returns 1D array containing R for each timestep.
    N, timepoints = np.shape(data)
    #phase calculation
    instantaneous_phase = np.unwrap(np.angle(data))

    r = np.abs(np.sum(np.exp(instantaneous_phase*complex(0, 1)), axis = 0)/N)

    return r


plt.subplot(311)
plt.plot(np.arange(0, 2, 0.001), kuramoto_r(transform[:,0,:]))


plt.subplot(312)
plt.plot(np.arange(0, 2, 0.001), kuramoto_r(transform[:,1,:]))


plt.subplot(313)
plt.plot(np.arange(0, 2, 0.001), kuramoto_r(transform[:,3,:]))

plt.show()