#Time series analysis of Gusto results


#Load in required libraries:
import numpy as np
from numpy import *
import numpy.ma as ma
from scipy import *
from numpy import fft
from scipy import fftpack
from scipy.signal import welch
from netCDF4 import Dataset
import netCDF4 as nc4
import pdb #to pause execution use pdb.set_trace()
import matplotlib.pyplot as plt


#data = Dataset('./tmp/diagnostics.nc', 'r')
#Ri = data.groups['RichardsonNumber']
#Ri_min = Ri.variables['min'][:]
#Ri_max = Ri.variables['max'][:]
#print(Ri_min.min(), Ri_max.max())
#pdb.set_trace()

#data = Dataset('./tmp/diagnostics.nc', 'r')
#gradU = data.groups['u_gradient']
#gradU_min = gradU.variables['min'][:]
#gradU_max = gradU.variables['max'][:]
#print(gradU_min.min(), gradU_max.max())
#pdb.set_trace()


#Read in Gusto output netCDF data:
dataPath = './point_data.nc'
f = nc4.Dataset(dataPath,'r')
bgrp = f.groups['b']
ts_masked = bgrp.variables['b'][:]
#f['time'][0:10] - to find time step of output = 0.004 (time step of linear solve)

dt = 0.004
t0 = 1*60
te = 4*60
Idx0 = int(t0/dt)
IdxE = int(te/dt)
ts = ts_masked.data[Idx0:IdxE]
ts = ts[:,0]

#Essentially this method divides signal into a number of segments that are nperseg points long. 
#A form of spectral analysis (periodogram) is performed on each segment.
#Then an average periodogram is computed from the set of periodograms.
#By default the segments overlap and the overlap is nperseg/2.
#So if nperseg is longer then we have higher frequency resolution but possibly more noise.

Nt = len(ts)
dnmntr = 4.
nperseg = int(Nt/dnmntr)
signal_f = 1./dt

#pdb.set_trace()
freqvec, psd = welch( ts,
                      fs=signal_f,              # sampling rate
                      window='hanning',         # apply a Hanning window before taking the DFT
                      nperseg=nperseg,          # compute periodograms of nperseg-long segments of ts
                      detrend='constant')       # detrend ts by subtracting the mean


#Set general plotting parameters assuming A4 page size:
A4Width = 8.27
MarginWidth = 1
width = A4Width-2*MarginWidth
height = 4
#For scaling the A4 plot dimensions:
ScaleFactor = 1
#ScaleFactor = 0.7
width = width*ScaleFactor
height - height*ScaleFactor

fig = plt.figure(1, figsize=(width,height))
grid = plt.GridSpec(1, 2, wspace=0.5, hspace=0.)

ax1 = fig.add_subplot(grid[0,0])
ax1.set_xlabel(r'$t$ (s)')
ax1.set_ylabel(r'$b$ (m/s/s)')
ax1.set_ylim(-.9,-.85)
t = np.arange(Nt)*dt + t0
ax1.plot(t,ts)

xlim = 2.5
ax2 = fig.add_subplot(grid[0,1])
ax2.set_xlabel(r'$f$' ' (Hz)')
ax2.set_ylabel('PSD')
ax2.set_xlim(0,xlim)
ax2.plot(freqvec,psd)

#ax3 = fig.add_subplot(grid[0,2])
#ax3.set_xlabel(r'$f$' ' (Hz)')
#ax3.set_xlim(0,xlim)
#ax3.semilogy(freqvec,psd)

plt.show()





pdb.set_trace()
