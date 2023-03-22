import numpy as np
import matplotlib.pyplot as plt


data = np.load('Data/results.npy')

print(np.shape(data))

N, variables, timepoints = np.shape(data)
print(timepoints)
def time_series(data):
    P1 = np.zeros((N, timepoints))
    for i in range(N):
        for j in range(timepoints):
            P1[i,j] = data[i,0,j]
    plt.imshow(P1)
    plt.show()

    return


def plot(data):
    t = np.linspace(0, 2, timepoints)
    for i in range(0, N, 10):
        plt.scatter(t[:], data[i,0,:])
    plt.show()
    return 0

#plot(data)
time_series(data)