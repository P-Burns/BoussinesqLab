#Code to post process dedalus results


#Load in required libraries:
import h5py
import numpy as np
from numpy import *
from scipy import *
from numpy import fft
from scipy import fftpack
from scipy.signal import welch
import pdb #to pause execution use pdb.set_trace()
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dedalus import public as de
import sys


#Get data from full State:
fnm = './Results/StateN2_02_25/' + 'State_s1' + '.h5'
hdf5obj = h5py.File(fnm,'r')
tmp_ = hdf5obj.get('tasks/'+'S')
data1 = np.array(tmp_)
hdf5obj.close()
print(data1.shape)
Nx = 80
Nz = 180

#Get single-point data:
fnm = './Results/StateN2_02_25_dt1/' + 'State_s1' + '.h5'
hdf5obj = h5py.File(fnm,'r')
tmp_ = hdf5obj.get('tasks/'+'S')
data2 = np.array(tmp_)
hdf5obj.close()
print(data2.shape)

Nt=600
dt2 = 0.1
t = np.arange(Nt)*dt2

plt.plot(t, data1[:,int(Nx/2.),int(Nz/2.)], '.k', markersize=5, alpha=1. )
plt.plot(t, data2.flatten(), '+r', markersize=10, alpha=0.5 )
plt.show()




