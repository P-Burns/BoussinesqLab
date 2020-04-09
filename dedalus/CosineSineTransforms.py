import numpy as np
from numpy import *
from scipy import *
from numpy import fft
from scipy import fftpack
import pdb #to pause execution for debugging use pdb.set_trace()


I = complex(0.,1.)

#
# FFT in X, Cosine FFT in Z (cosine transform)
#
def FFT_FCT(N1,N2,F):
    F_hat = complex(0.,0.)*np.zeros((N1,N2))
    #
    #   Cosine Transform first
    #
    for l in range(N1):
        F_hat[l,:] = fftpack.dct(F[l,:],type=1)/(np.float_(N2)-1.)
    #
    #  Then Fourier Transform
    #
    for q in range(N2):
        F_hat[:,q] = np.fft.fft(F_hat[:,q])    
    #
    #  No normalization in this configuration
    #
    return F_hat

#
# Inverse of FFT in X, Cosine FFT in Z
#

def iFFT_FCT(N1,N2,F_hat):
    F = np.zeros((N1,N2))
    for k in range(N2):
        F[:,k] = np.fft.ifft(F_hat[:,k]).real
    #
    #   The F's are reals once we take the ifft, but they have complex parts that won't work
    #   with the cosine transform, so we set them all real
    #
    #    F = real(F)
    
    #
    # Now that you have Reals take the inverse cosine transform
    #
    for l in range(N1):
        F[l,:] = fftpack.idct(F[l,:],type=1)*.5
    #
    # Normalize for the inverse cosine transform
    #
    
    return F
#
# FFT in X, SINE FFT in Z (sine transform)
#
def FFT_FST(N1,N2,F):
    F_hat = complex(0.,0.)*np.zeros((N1,N2))
    #
    #   Sine Transform first
    #
    for l in range(N1):
        F_hat[l,1:N2-1] = fftpack.dst(F[l,1:N2-1],type=1)/(np.float_(N2)-1.)
    #
    #  Then Fourier Transform
    #
    for q in range(N2):
        F_hat[:,q] = np.fft.fft(F_hat[:,q])    
    #
    #  No normalization in this configuration
    #
    return F_hat

#
# Inverse of FFT in X, SINE FFT in Z
#

def iFFT_FST(N1,N2,F_hat):
    F = np.zeros((N1,N2))
    for k in range(N2):
        F[:,k] = np.fft.ifft(F_hat[:,k]).real
    #
    #   The F's are reals once we take the ifft, but they have complex parts that won't work
    #   with the sine transform, so we set them all real
    #
    #    F = real(F)
    
    #
    # Now that you have Reals take the inverse sine transform
    #
    for l in range(N1):
        F[l,1:N2-1] = fftpack.idst(F[l,1:N2-1],type=1)*.5
    #
    # Normalize for the inverse cosine transform
    #
    #    F = F/2./N2

    return F

#
# The following are for computing first derivatives
#


def iFFT_FST_X(N1,N2,F_hat,kk):
    F = np.zeros((N1,N2))
    for k in range(N2):
        F[:,k] = np.fft.ifft(1j*kk*F_hat[:,k]).real
    
    for ll in range(N1):
        F[ll,1:N2-1] = fftpack.idst(F[ll,1:N2-1],type=1)*.5
        F[ll,0]=0.
        F[ll,N2-1] = 0.
        
    return F

def iFFT_FCT_X(N1,N2,F_hat,kk):
    F = np.zeros((N1,N2))
    for k in range(N2):
        F[:,k] = np.fft.ifft(1j*kk*F_hat[:,k]).real
    
    for ll in range(N1):
        F[ll,:] = fftpack.idct(F[ll,:],type=1)*.5
        
    return F

def iFFT_FST_Z(N1,N2,F_hat,kk_cosine):
    F = np.zeros((N1,N2))
    for k in range(N2):
        F[:,k] = np.fft.ifft(F_hat[:,k]).real
    for ll in range(N1):
        F[ll,1:N2-1] = fftpack.idst(kk_cosine[1:N2-1]*F[ll,1:N2-1],type=1)*.5
        F[ll,0]=0.
        F[ll,N2-1] = 0.
        
    return F


def iFFT_FCT_Z(N1,N2,F_hat,kk_cosine):
    F = np.zeros((N1,N2))
    
    for k in range(N2):
        F[:,k] = np.fft.ifft(F_hat[:,k]).real
        
    for l in range(N1):
        F[l,:] = fftpack.idct(kk_cosine[:]*F[l,:],type=1)*.5
        
    return F

#
# The following are for the second derivatives
#


def iFFT_FST_XX(N1,N2,F_hat,kk):
    F = np.zeros((N1,N2))
    for k in range(N2):
        F[:,k] = np.fft.ifft(-kk*kk*F_hat[:,k]).real
    
    for ll in range(N1):
        F[ll,1:N2-1] = fftpack.idst(F[ll,1:N2-1],type=1)*.5
        F[ll,0]=0.
        F[ll,N2-1] = 0.
        
    return F

def iFFT_FCT_XX(N1,N2,F_hat,kk):
    F = np.zeros((N1,N2))
    for k in range(N2):
        F[:,k] = np.fft.ifft(-kk*kk*F_hat[:,k]).real
    
    for ll in range(N1):
        F[ll,:] = fftpack.idct(F[ll,:],type=1)*.5
        
    return F

def iFFT_FST_ZZ(N1,N2,F_hat,kk_cosine):
    F = np.zeros((N1,N2))
    for k in range(N2):
        F[:,k] = np.fft.ifft(F_hat[:,k]).real
    for ll in range(N1):
        F[ll,1:N2-1] = fftpack.idst(-kk_cosine[1:N2-1]*kk_cosine[1:N2-1]*F[ll,1:N2-1],type=1)*.5
        F[ll,0]=0.
        F[ll,N2-1] = 0.
        
    return F


def iFFT_FCT_ZZ(N1,N2,F_hat,kk_cosine):
    F = np.zeros((N1,N2))
    
    for k in range(N2):
        F[:,k] = np.fft.ifft(F_hat[:,k]).real
        
    for l in range(N1):
        F[l,:] = fftpack.idct(-kk_cosine[1:N2-1]*kk_cosine[:]*F[l,:],type=1)*.5
        
    return F

def iFFT_FST_Helm(N1, N2, F_hat, kk, kk_cosine):
    """ This function solves the helmholtz equation for problems
        with the sin handedness """
    fac = np.zeros((N1,N2))*complex(0,1)

##---------------------------------------------------------------------------##
## 
##---------------------------------------------------------------------------##

    for i in range(N1):
        for j in range(N2):
            if (kk[i] == 0) and (kk_cosine[j] == 0):
                fac[i,j] = 0.
            else:
                fac[i,j] = -F_hat[i,j]/(kk[i]*kk[i]+kk_cosine[j]*kk_cosine[j])

    return iFFT_FST(N1, N2, fac)

#
# The following are for higher even derivatives
#


def iFFT_FST_XXXX(N1,N2,F_hat,kk):
    F = np.zeros((N1,N2))
    for k in range(N2):
        F[:,k] = np.fft.ifft(kk*kk*kk*kk*F_hat[:,k]).real
    
    for ll in range(N1):
        F[ll,1:N2-1] = fftpack.idst(F[ll,1:N2-1],type=1)*.5
        F[ll,0]=0.
        F[ll,N2-1] = 0.
        
    return F

def iFFT_FCT_XXXX(N1,N2,F_hat,kk):
    F = np.zeros((N1,N2))
    for k in range(N2):
        F[:,k] = np.fft.ifft(kk*kk*kk*kk*F_hat[:,k]).real
    
    for ll in range(N1):
        F[ll,:] = fftpack.idct(F[ll,:],type=1)*.5
        
    return F

def iFFT_FST_XX6(N1,N2,F_hat,kk):
    F = np.zeros((N1,N2))
    for k in range(N2):
        F[:,k] = np.fft.ifft(-kk*kk*kk*kk*kk*kk*F_hat[:,k]).real
    
    for ll in range(N1):
        F[ll,1:N2-1] = fftpack.idst(F[ll,1:N2-1],type=1)*.5
        F[ll,0]=0.
        F[ll,N2-1] = 0.
        
    return F

def iFFT_FCT_XX6(N1,N2,F_hat,kk):
    F = np.zeros((N1,N2))
    for k in range(N2):
        F[:,k] = np.fft.ifft(-kk*kk*kk*kk*kk*kk*F_hat[:,k]).real
    
    for ll in range(N1):
        F[ll,:] = fftpack.idct(F[ll,:],type=1)*.5
        
    return F

#   Yes, all these kk are a little excessive.

def iFFT_FST_XX8(N1,N2,F_hat,kk):
    F = np.zeros((N1,N2))
    for k in range(N2):
        F[:,k] = np.fft.ifft(kk**8*F_hat[:,k]).real
    
    for ll in range(N1):
        F[ll,1:N2-1] = fftpack.idst(F[ll,1:N2-1],type=1)*.5
        F[ll,0]=0.
        F[ll,N2-1] = 0.
        
    return F

def iFFT_FCT_XX8(N1,N2,F_hat,kk):
    F = np.zeros((N1,N2))
    for k in range(N2):
        F[:,k] = np.fft.ifft(kk**8*F_hat[:,k]).real
    
    for ll in range(N1):
        F[ll,:] = fftpack.idct(F[ll,:],type=1)*.5
        
    return F

def iFFT_FST_ZZZZ(N1,N2,F_hat,kk_cosine):
    F = np.zeros((N1,N2))
    for k in range(N2):
        F[:,k] = np.fft.ifft(F_hat[:,k]).real
    for ll in range(N1):
        fac = kk_cosine[1:N2-1]*kk_cosine[1:N2-1]*kk_cosine[1:N2-1]*kk_cosine[1:N2-1]
        F[ll,1:N2-1] = fftpack.idst(fac*F[ll,1:N2-1],type=1)*.5
        F[ll,0]=0.
        F[ll,N2-1] = 0.
        
    return F

def iFFT_FCT_ZZZZ(N1,N2,F_hat,kk_cosine):
    F = np.zeros((N1,N2))
    
    for k in range(N2):
        F[:,k] = np.fft.ifft(F_hat[:,k]).real
        
    for l in range(N1):
        fac = kk_cosine[1:N2-1]*kk_cosine[1:N2-1]*kk_cosine[1:N2-1]*kk_cosine[1:N2-1]
        F[l,:] = fftpack.idct(fac*F[l,:],type=1)*.5
        
    return F

def iFFT_FST_ZZ6(N1,N2,F_hat,kk_cosine):
    F = np.zeros((N1,N2))
    for k in range(N2):
        F[:,k] = np.fft.ifft(F_hat[:,k]).real
    for ll in range(N1):
        fac = kk_cosine[1:N2-1]**6
        F[ll,1:N2-1] = fftpack.idst(-fac*F[ll,1:N2-1],type=1)*.5
        F[ll,0]=0.
        F[ll,N2-1] = 0.
        
    return F


def iFFT_FCT_ZZ6(N1,N2,F_hat,kk_cosine):
    F = np.zeros((N1,N2))
    
    for k in range(N2):
        F[:,k] = np.fft.ifft(F_hat[:,k]).real
        
    for l in range(N1):
        fac = kk_cosine[1:N2-1]**6
        F[l,:] = fftpack.idct(-fac*F[l,:],type=1)*.5
        
    return F

def iFFT_FST_ZZ8(N1,N2,F_hat,kk_cosine):
    F = np.zeros((N1,N2))
    for k in range(N2):
        F[:,k] = np.fft.ifft(F_hat[:,k]).real
    for ll in range(N1):
        fac = kk_cosine[1:N2-1]**8
        F[ll,1:N2-1] = fftpack.idst(fac*F[ll,1:N2-1],type=1)*.5
        F[ll,0]=0.
        F[ll,N2-1] = 0.
        
    return F


def iFFT_FCT_ZZ8(N1,N2,F_hat,kk_cosine):
    F = np.zeros((N1,N2))
    
    for k in range(N2):
        F[:,k] = np.fft.ifft(F_hat[:,k]).real
        
    for l in range(N1):
        fac = kk_cosine[1:N2-1]**8
        F[l,:] = fftpack.idct(fac*F[l,:],type=1)*.5
        
    return F
