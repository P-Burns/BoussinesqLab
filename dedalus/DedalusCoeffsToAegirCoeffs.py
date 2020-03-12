"""
This is an example code that 

1. Creates a function using the Dedalus framework, and returns Dedalus spectrial coefficients.
2. Compares the Dedalus spectral coefficients to Aegirs
3. Provides a function that inputs Dedalus spectral coefficients and outputs Aegir spectral coefficients.
4. Checks/Tests to make sure the function works.

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

Lx = 1.
Lz = 2.
Nx = 8
Nz = 8

# Bases and domain for Dedalus
xBasis = de.Fourier('x', Nx, interval=(0., Lx), dealias=3/2)
zBasis = de.SinCos('z', Nz, interval=(0, Lz), dealias=3/2)
domain 	= de.Domain([xBasis, zBasis], grid_dtype=np.float64)

xgrid = domain.grid(0)
zgrid = domain.grid(1)

uWaveDedalus = domain.new_field(name='uWaveDedalus')
uWaveDedalus.meta['z']['parity'] = -1


#
# Basis and domain for Aegir. Because we wanted to be able to set boundary conditions with Aegir we
# chose spectral cofficients to also exist on the boundaries for the cosine series. This means
# that a direct comparision isn't really possible between the two grids. But we should get similar spectral coefficients.
#

xGridNumpy = np.linspace(0.,Lx, Nx, endpoint=False)
NzAegir = Nz+2                     # This is only required if you want to
                                   # Compare with Dedalus. Otherwise leave
                                   # it as is.

                                   
dz = float(Lz)/(NzAegir-1)
zGridNumpy = np.arange(NzAegir)*dz

#
# Create a test function on the Aegir grid
# This function could be computed by the random phase intial conditon
# code, for example.
#
# We shall need its transformed coefficients for the routine
# below, that interpolates Aegir functions onto other grids
#

kk = np.fft.fftfreq(Nx,Lx/Nx)*2.*np.pi # wave numbers used in Aegir
kk_cosine = np.arange((NzAegir))*np.pi/Lz # wave numbers used in Aegir

#
# This defines the numpy function we wish to find on the Aegir grid

uWaveNumpy = np.zeros((Nx, NzAegir))*1j
for i in range(Nx):
    for j in range(NzAegir):
        #uWaveNumpy[i,j] = np.sin(np.pi*zGridNumpy[j]/Lz)*np.sin(2.*np.pi*xGridNumpy[i]/Lx)
        uWaveNumpy[i,j] = np.cos(2.*np.pi*xGridNumpy[i]/Lx)*np.sin(3.*np.pi*zGridNumpy[j]/Lz)

# Compute the Aegir spectral coefficients

# Remember that to compare the coefficients from Aegir and Numpy you need
# to divide by Nx. However, to actually use the Aegir coeffients, use
# them as they are. 
#
uWaveNumpyHat =  cst.FFT_FST(Nx, NzAegir, uWaveNumpy)
uWaveNumpyHatCompare = np.copy(uWaveNumpyHat)/(np.float_(NzAegir)-2.)


#
# This loop checks to make sure that u_hat_k_transpose = u_hat_(-k).
# Recall that the first element is the 0 mode and has no negative counterpart
#
eps = 1.e-9
MaxRealityError = eps
for i in range(int(Nx/2)+1):
    for j in range(NzAegir):
        if i == 0:
            RealityError = 0.
        else:
            RealityError = np.abs(uWaveNumpyHatCompare[i,j]-np.conj(uWaveNumpyHatCompare[Nx-i,j]))

        if RealityError > MaxRealityError:
            MaxRealityError = RealityError
            
if np.abs(MaxRealityError > eps):
    print('Checking the Reality.....',i,j,MaxRealityError)
else:
    print('Checking to make sure the symmetry conditions are met ....and they are.')



def StoreDedalusCoefficientsAsNumpyCoeffs(uHatDedalus):
    """
    
    uHatDedalus is the spectral coefficents as computed with Dedalus
    NewArray has the dimensions and normalization required to pass these coefficients to Aegir arrays
    
    """
    nxDedalus = len(uHatDedalus)
    nzDedalus = len(uHatDedalus[1])
    NzAegir = nzDedalus + 2
    NxAegir = nxDedalus*2
    print('aegir versus dedalus',NxAegir, nxDedalus)
    NewArray = np.zeros((NxAegir,NzAegir))*1j
    NewArray[0:nxDedalus,0:nzDedalus] = uHatDedalus[0:nxDedalus,0:nzDedalus]   # Assign the positive wave numbers
    for i in range(np.int(nxDedalus/2+1)):
        for j in range(nzDedalus):
            if i > 0: # Do not do the zero mode because it does not have a corresponding negative wave number
                NewArray[NxAegir-i, j] = np.conj(uHatDedalus[i,j])
    return NewArray*(np.float_(NzAegir)-2.)  # This normalizes the spectrum with the protocol that Aegir expects.

#
# Recall that to check, we really need the function on the Dedalus grid
# so we'll define the exact same function n the Dedalus grid

for i in range(Nx):
   for j in range(Nz):
       #uWaveDedalus['g'][i,j] =  np.sin(np.pi*zgrid[0,j]/Lz)*np.sin(2.*np.pi*xgrid[i,0]/Lx)
       uWaveDedalus['g'][i,j] =  np.cos(2.*np.pi*xgrid[i,0]/Lx)*np.sin(3.*np.pi*zgrid[0,j]/Lz)

uWaveDedalus['c'] # Compute the coefficients

#
# Call the routine to fill out the full spectrum required for Aegir using the Dedalus spectrum
#
NewAegirCeofficients = StoreDedalusCoefficientsAsNumpyCoeffs(uWaveDedalus['c'])

print("NewAegirCoefficients Computed")

#
# Compute the difference in the spectrum computed from dedalus and mapped onto the aegir arrays
# versus what Aegir computed itself
#

eps = 1.e-9
coeffsErrorMax = eps
for i in range(np.int(Nx)):
    for j in range(Nz):
        coeffsError = np.abs(uWaveNumpyHat[i,j]-NewAegirCeofficients[i,j])
        if coeffsError > coeffsErrorMax:
            coeffsErrorMax = coeffsError

print("The maximum error between the computed numpy/aegir coeffs and those of Dedalus is",coeffsErrorMax)


