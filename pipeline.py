from LaNMM_BOLD import simulation
from BOLD_analysis import data_analysis
import numpy as np
import matplotlib.pyplot as plt


epsilon = 10
p1 = 200
p2 = 150

results = np.zeros((1, 6))

i = 0

path = simulation(p1, p2, epsilon)
results[i,0], results[i,1], results[i,2],results[i,3], results[i,4], results[i,5] = data_analysis(path)
#np.save('Data/pearsonR', results)
print('Done simulation with epsilon = ', epsilon)


print(results)
#np.save('Data/pearsonR', results)
#plt.plot(epsilons, results[:,0])
#plt.savefig('Assets/p1=200,p2=150.png')


