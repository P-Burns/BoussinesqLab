"""
This is an example code for using a function called InterpolateFromAegir to use Aegir spectral coefficients compute the function at any set of (x,z) grid points.  This is an expensive calculation as the interpolation is not accomplished with FFTs, but with direct summation. We assume that the domain size (Lx, Lz) is the same for Aegir as the new grid.

The structure of the code is as follows:

1. We define a Dedalus Grid and Dedalus functions as an example (but one could generate points from firedrake too.

2. As an example we define a function that we wish to evalute on the new grid, and takes its transform using Aegir spectral transform routines. This could be replaced by the computation of the random phase field.  Then we take its transoform.

3. We pass the (new) alternative grid and Aegir spectral coefficients to the interpolation routine InterpolateFromAegir, which returns the new function.

4. We test to see if the interpolation is correct by defining a new Dedalus function and doing a point-to-point comparison.  This step is only for debugging and not needed for doing interpolations once we're sure it works.


"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import fft

# Import the Aegir transform module
import CosineSineTransforms as cst

# Import key dedalus modules
from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh, pad_limits

import logging
logger = logging.getLogger(__name__)


#
# Define the domain lengths
#

Lx = 1.
Lz = 2.


#
# This Nx and Nz is for the 'new' or alternative grid (in this case Dedalus)
#


NxAlt = 8
NzAlt = 12

# Bases and domain for Dedalus
xBasis = de.Fourier('x', NxAlt, interval=(0., Lx), dealias=3/2)
zBasis = de.SinCos('z', NzAlt, interval=(0, Lz), dealias=3/2)
domain 	= de.Domain([xBasis, zBasis], grid_dtype=np.float64)

xgridAlt = domain.grid(0)
zgridAlt = domain.grid(1)

#
# We define a function here so that we can use it to check our interpolation
# later
#
uWaveDedalus = domain.new_field(name='uWaveDedalus')
uWaveDedalus.meta['z']['parity'] = -1
uWaveDedalusInterp = domain.new_field(name='uWaveDedalus')
uWaveDedalusInterp.meta['z']['parity'] = -1



#
# Basis and domain for Aegir
#


#
# These are the resolution for the Aegir function that we'll interpolate
#
Nx = 8
Nz = 16

xGridNumpy = np.linspace(0.,Lx, Nx, endpoint=False)

dz = float(Lz)/(Nz-1)
zGridNumpy = np.arange(Nz)*dz

#
# Create a test function on the Aegir grid
# This function could be computed by the random phase intial conditon
# code, for example.
#
# We shall need its transformed coefficients for the routine
# below, that interpolates Aegir functions onto other grids
#

kk = np.fft.fftfreq(Nx,Lx/Nx)*2.*np.pi # wave numbers used in Aegir
kk_cosine = np.arange((Nz))*np.pi/Lz   # wave numbers used in Aegir

#
# This defines the numpy function we wish to interpolate onto a new grid
# You could replace this with the calculation for the random phase initial
# condition, which should give you grid point values.

uWaveNumpy = np.zeros((Nx, Nz))
for i in range(Nx):
    for j in range(Nz):
        uWaveNumpy[i,j] = np.cos(2.*np.pi*xGridNumpy[i]/Lx)*np.sin(3.*np.pi*zGridNumpy[j]/Lz)

# Compute the Aegir spectral coefficients

# Remember that to compare the coefficients from Aegir and Numpy you need
# to divide by Nx.
#
uWaveNumpyHat =  cst.FFT_FST(Nx, Nz, uWaveNumpy)/(np.float_(Nx))


def InterpolateFromAegir(Nx, Nz, xgridAlt,  zgridAlt, kk, kk_cosine, uHatNumpy):
    """
    
    Nx and Nz are the size of the Aegir arrays
    xgrid and zgrid are the grids from Dedalus and can be any size
    kk and kk_cosine are from Aegir
    uHatNumpy is the spectral coefficents as computed with Aegir
    
    """
    fOutput = np.zeros((len(xgridAlt),len(zgridAlt)))*1j
    for ix in range(len(xgridAlt)): # Loops over the xgrid
        for iz in range(len(zgridAlt)):# Loops over the zgrid
            fOutput[ix,iz] = 0.
            for i in range(Nx):
                for j in range(1,Nz-1):
                    fOutput[ix, iz] += np.sin(kk_cosine[j]*zgridAlt[iz])*np.exp(-1j*kk[i]*xgridAlt[ix])*uHatNumpy[i,j]
    return fOutput.real

uWaveDedalusInterp['g'] = InterpolateFromAegir(Nx, Nz, xgridAlt[:,0], zgridAlt[0,:], kk, kk_cosine, uWaveNumpyHat)

#
#print('Can the function repeat its own grid point values',max(np.abs(np.reshape(fOutputCheck-uWaveNumpy,-1))))
#

#
# Recall that to check, we really need the function on the Dedalus grid
# so we'll define the exact same function n the Dedalus grid

for i in range(NxAlt):
   for j in range(NzAlt):
       uWaveDedalus['g'][i,j] =  np.cos(2.*np.pi*xgridAlt[i,0]/Lx)*np.sin(3.*np.pi*zgridAlt[0,j]/Lz)

uWaveDedalus['c']
uWaveDedalusInterp['c']

eps = 1.e-9
waveErrorMax = eps
for i in range(NxAlt):
    for j in range(NzAlt): 
#        waveError = abs(uWaveDedalus['g'][i,j] - fOutputCheck[i,j])
        waveError = abs(uWaveDedalus['g'][i,j] - uWaveDedalusInterp['g'][i,j])
        if waveError > waveErrorMax:
            waveErrorMax = waveError

print('The max error betwteen the values on Aegir and Dedalus grid is:', waveErrorMax)

#print('function in Dedalus ', uWaveDedalus['g'])
#print('function Interpolated to Dedalus', fOutputCheck)

plt.plot(uWaveDedalus['g'][:,1],'-bo')
plt.plot(uWaveDedalusInterp['g'][]

