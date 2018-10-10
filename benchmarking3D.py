#Code to plot up benchmarking results


#Load in required libraries:
import numpy as np
from numpy import *
import pdb #to pause execution use pdb.set_trace()
import matplotlib.pyplot as plt


cores = np.array([4,8,16,32,64])
dt_out = 0.01
sim_time = np.array([1,2,4,3,2])*dt_out

plt.plot(cores,sim_time,'-o')
plt.show()





pdb.set_trace()
