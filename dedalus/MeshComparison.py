#Code to compare results from different grid resolutions


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


#Read in data:
dir_state = './Results/meshTest/State_mesh0/'
fnm = dir_state + 'State_s2' + '.h5'
hdf5obj = h5py.File(fnm,'r')
tmp_ = hdf5obj.get('tasks/S')
S0 = np.array(tmp_)

dir_state = './Results/meshTest/State_mesh1/'
fnm = dir_state + 'State_s2' + '.h5'
hdf5obj = h5py.File(fnm,'r')
tmp_ = hdf5obj.get('tasks/S')
S1 = np.array(tmp_)

dir_state = './Results/meshTest/State_mesh2/'
fnm = dir_state + 'State_s2' + '.h5'
hdf5obj = h5py.File(fnm,'r')
tmp_ = hdf5obj.get('tasks/S')
S2 = np.array(tmp_)

dir_state = './Results/meshTest/State_mesh3/'
fnm = dir_state + 'State_s2' + '.h5'
hdf5obj = h5py.File(fnm,'r')
tmp_ = hdf5obj.get('tasks/S')
S3 = np.array(tmp_)

dir_state = './Results/meshTest/State_mesh4/'
fnm = dir_state + 'State_s2' + '.h5'
hdf5obj = h5py.File(fnm,'r')
tmp_ = hdf5obj.get('tasks/S')
S4 = np.array(tmp_)


#Set up grids:
Lx,Lz = (0.2,0.45)

Nx0 = 20
Nx1 = 40
Nx2 = 80
Nx3 = 160
Nx4 = 320

Nz0 = 45 + 1
Nz1 = 90
Nz2 = 180
Nz3 = 360
Nz4 = 720

x_basis0 = de.Fourier('x', Nx0, interval=(0, Lx), dealias=3/2)
z_basis0 = de.SinCos('z', Nz0, interval=(0, Lz), dealias=3/2)
domain0 = de.Domain([x_basis0, z_basis0], grid_dtype=np.float64)
x0=domain0.grid(0)[:,0]
z0=domain0.grid(1)[0,:]

x_basis1 = de.Fourier('x', Nx1, interval=(0, Lx), dealias=3/2)
z_basis1 = de.SinCos('z', Nz1, interval=(0, Lz), dealias=3/2)
domain1 = de.Domain([x_basis1, z_basis1], grid_dtype=np.float64)
x1=domain1.grid(0)[:,0]
z1=domain1.grid(1)[0,:]

x_basis2 = de.Fourier('x', Nx2, interval=(0, Lx), dealias=3/2)
z_basis2 = de.SinCos('z', Nz2, interval=(0, Lz), dealias=3/2)
domain2 = de.Domain([x_basis2, z_basis2], grid_dtype=np.float64)
x2=domain2.grid(0)[:,0]
z2=domain2.grid(1)[0,:]

x_basis3 = de.Fourier('x', Nx3, interval=(0, Lx), dealias=3/2)
z_basis3 = de.SinCos('z', Nz3, interval=(0, Lz), dealias=3/2)
domain3 = de.Domain([x_basis3, z_basis3], grid_dtype=np.float64)
x3=domain3.grid(0)[:,0]
z3=domain3.grid(1)[0,:]

x_basis4 = de.Fourier('x', Nx4, interval=(0, Lx), dealias=3/2)
z_basis4 = de.SinCos('z', Nz4, interval=(0, Lz), dealias=3/2)
domain4 = de.Domain([x_basis4, z_basis4], grid_dtype=np.float64)
x4=domain4.grid(0)[:,0]
z4=domain4.grid(1)[0,:]


#Plot results:
plt.plot(S0[0,int(Nx0/2.),:], z0, ls=':',  lw=2, c='gray', label=r'4$\Delta x$')
plt.plot(S1[0,int(Nx1/2.),:], z1, ls='-',  lw=2, c='gray', label=r'2$\Delta x$')
plt.plot(S2[0,int(Nx2/2.),:], z2, ls='-',  lw=3, c='k',    label=r'$\Delta x$=2.5 mm')
plt.plot(S3[0,int(Nx3/2.),:], z3, ls='-',  lw=2, c='k',    label=r'$\Delta x$/2')
plt.plot(S4[0,int(Nx4/2.),:], z4, ls=':',  lw=1, c='k',    label=r'$\Delta x$/4')
plt.xlabel(r'$S^{\prime}$ (g/kg)')
plt.ylim(0,Lz)
plt.ylabel(r'$z$ (m)')
plt.legend(frameon=False)
plt.show()









