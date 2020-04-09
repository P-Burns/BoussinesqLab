import sys
import numpy as np
import time as tme
from numpy import *
from numpy import fft
from scipy import fftpack
import matplotlib.pyplot as plt


def randomPhaseICs(rhoHat,Lx,Lz,kkx,kkz,kx_0,kz_0,m,plot=False):

    Nx=len(kkx)
    Nz=len(kkz)

    #Below, renormalize the pe spectrum, so that it has a shape where the amplitudes (PE energy) have a shape that varies more slowly.
    BigPEkTarget = np.zeros((Nx,Nz))
    BigPEk = np.zeros((Nx,Nz), dtype = 'float')
    ixwSet = np.zeros(int(max(Nx,Nz)))
    ikx_count = 0
    for ikx in kkx:
        iikx = np.sqrt(ikx*ikx/4./np.pi/np.pi*Lx*Lx)
        #iikx = int(ikx/2./np.pi*Lx)
        ikz_count = 0
        for ikz in kkz:
            iikz = int(ikz/np.pi*Lz)
            xwj = np.sqrt(iikx*iikx + iikz*iikz)
            xw = np.sqrt(ikx*ikx + ikz*ikz)
            ixw = int(np.sqrt(iikx*iikx + iikz*iikz))
            #ixw = int(round(np.sqrt(iikx*iikx + iiky*iiky),0))
            absrhoHat = rhoHat[ikx_count, ikz_count]*rhoHat[ikx_count, ikz_count].conj()
            BigPEk[ikx_count, ikz_count] = BigPEk[ikx_count,ikz_count] + .5*xw*xw*absrhoHat.real
            #print(ikx_count, ikz_count, iikx, iikz, m, kx_0, kz_0)
            BigPEkTarget[ikx_count,ikz_count] = (iikx**(m/2)/(iikx+kx_0)**m)*(iikz**(m/2)/(iikz+kz_0)**m)
            ikz_count += 1
            #print(iikx, iikz, xwj, ixw)
        ikx_count += 1

    #
    # Renormalize BigETarget so that we see what the actual shape of potential energy is when it is
    # binned according to the shell wave number, ix
    #
    BigPEkTarget = BigPEkTarget/max(np.reshape(BigPEkTarget, -1))
    BigPEk = BigPEk/max(np.reshape(BigPEk,-1))

    #We renormalize the amplitudes of the original random initial condition so the amplitdues are slowly varrying.
    #So, renormalize.
    #First loop over every wave number. 
    #Find out which 'shell' belong to each wave number and then divide the spectral coefficient by the max energy in that shell.
    #Next multiply it by the target energy.

    rhoHatNew = np.copy(rhoHat)
    ikx_count = 0
    for ikx in kkx:
        ikz_count = 0
        iikx = np.sqrt(ikx*ikx/4./np.pi/np.pi*Lx*Lx)
        #iikx = int(ikx/2./np.pi*Lx)
        for ikz in kkz:
            iikz = int(ikz/np.pi*Lz)
            xwj = np.sqrt(iikx*iikx + iikz*iikz)
            xw = np.sqrt(ikx*ikx + ikz*ikz)
            ixw = int(np.sqrt(iikx*iikx + iikz*iikz))
            if iikz > 0 and iikx > 0 and ikz_count < Nz-1:
                rhoHatNew[ikx_count, ikz_count] = rhoHat[ikx_count,ikz_count]/BigPEk[ikx_count, ikz_count]
                rhoHatNew[ikx_count, ikz_count] = rhoHatNew[ikx_count, ikz_count]*BigPEkTarget[ikx_count, ikz_count]
            else:
                rhoHatNew[ikx_count, ikz_count] = 0.
            ikz_count += 1
        ikx_count += 1

    #rhoNew = rhoHatNew['g']
    return rhoHatNew



    # Section of code for making plots and printing numbers to check operation of above method:
    if plot == True:
        print(kx_0,kz_0,m)

        PE2DTarget = np.zeros((Nx,Nz))
        ikx_count = 0
        for ikx in kkx:
            iikx = np.sqrt(ikx*ikx/4./np.pi/np.pi*Lx*Lx)
            #iikx = int(ikx/2./np.pi*Lx)
            ikz_count = 0
            for ikz in kkz:
                iikz = int(ikz/np.pi*Lz)
                PE2DTarget[ikx_count, ikz_count] = BigPEK(ikx, ikz, Lx, Lz, kx_0, kz_0, m)
                ikz_count += 1
            ikx_count += 1

        plt.figure(1)
        plt.contourf(kkz/np.pi*Lz,kkx/2./np.pi*Lx,PE2DTarget) # We want this to be low wave numbers in x and z

        # This cell makes a contour plot of the real part of the density
        plt.figure(2)
        plt.contourf(rho.real)
        plt.colorbar()

        print('Centroid of original data is at wave number: ',PeCentroid(rhoHat, kkx, kkz))

        #Compute the PE energy for each vertical and horizontal wave number
        PEkshape = np.zeros((Nx,Nz), dtype = 'float')
        ikx_count = 0
        for ikx in kkx:
            ikz_count = 0
            for ikz in kkz:
                xw = sqrt(ikx*ikx/4./np.pi/np.pi*Lx*Lx + ikz*ikz/np.pi/np.pi*Lz*Lz)
                energyValue = .5*xw*xw*rhoHat[ikx_count, ikz_count]*rhoHat[ikx_count, ikz_count].conj()
                PEkshape[ikx_count, ikz_count] = energyValue.real
                ikz_count += 1
            ikx_count += 1

        PEarray = np.zeros((Nx,Nz))
        ikx_count = 0
        for ikx in kkx:
            ikz_count = 0
            for ikz in kkz:
                xw = sqrt(ikx*ikx/4./np.pi/np.pi*Lx*Lx)
                zw = ikz/np.pi*Lz
                ixw = int(xw)
                #print(ikx_count, xw, ikz_count, zw)
                energyValue = .5*xw*zw*rhoHat[ikx_count, ikz_count]*rhoHat[ikx_count, ikz_count].conj()
                PEarray[ikx_count, ikz_count] = energyValue.real
                ikz_count += 1
            ikx_count += 1

        plt.figure(3)
        plt.contourf(PEarray)

        plt.figure(4)
        plt.plot(kkz/np.pi*Lz,PEarray[1,:])

        # Here we plot the potential energy as a function of the shell wave number. 
        # We can see there is lots of noise  in the high wave numbers
        #print(PEarray[4,4])
        plt.figure(5)
        plt.loglog(PEarray[3,:])

        print(max(np.reshape(BigPEkTarget,-1)))
        print(max(np.reshape(BigPEk,-1)))

        #Now that we have brand new, renormalized spectral coefficients, compute the new PEk! This should have a more smoothly varying shape.
        BigPEkNew = np.zeros((Nx,Nz), dtype = 'float')
        ikx_count = 0
        for ikx in kkx:
            #iikx = int(ikx/2./np.pi*Lx)
            iikx = np.sqrt(ikx*ikx/4./np.pi/np.pi*Lx*Lx)
            ikz_count = 0
            for iky in kkz:
                iikz = int(ikz/np.pi*Lz)
                xwj = np.sqrt(iikx*iikx + iikz*iikz)
                xw = np.sqrt(ikx*ikx + ikz*ikz)
                ixw = int(np.sqrt(iikx*iikx + iikz*iikz))
                absrhoHat = rhoHatNew[ikx_count, ikz_count]*rhoHatNew[ikx_count, ikz_count].conj()
                BigPEkNew[ikx_count, ikz_count] = BigPEkNew[ikx_count, ikz_count] + .5*xw*xw*absrhoHat.real
                ikz_count += 1
            ikx_count += 1

        logBigPEk = np.log(BigPEk)
        plt.figure(6)
        plt.contourf(logBigPEk)
        plt.figure(7)
        plt.loglog(BigPEk[20,:])
        plt.title('This is the PE energy vs the z wave number.')
        plt.figure(8)
        plt.loglog(BigPEk[:,80])
        plt.title('This is the PE energy vs the x wave number.')

        logBigPEkTarget = np.log(BigPEkTarget)
        plt.figure(9)
        plt.contourf(logBigPEkTarget)
        plt.title('Here is our target distribution')
        
        plt.figure(10)
        plt.semilogy(BigPEkTarget[20,:])
        plt.title('This is the TARGET PE energy vs the z wave number.')
        
        plt.figure(11)
        plt.semilogy(BigPEkTarget[:,80])
        plt.title('This is the TARGET PE energy vs the x wave number.')
        plt.show()

    def BigPEK(kx, kz, Lx, Lz, kx_0, kz_0, m):
        """This is the type of function that is typically used to define amplitudes 
        for random phase initial conditions. For our periodic lab tests we cannot sensibly
        use a 'circle' in wave number space to compute an effective radius as we need to be
        able to have higher wave numbers in the vertical than in the horizontal. Therefore we
        define a two-d output.
    
        ikx = the x wave number (horizontal)
        ikz = the z wave number (vertical)
    
        kx_0 is the horizontal wave number where we'd like the energy to peak
        kz_0 is the vertical wave number where we'd like the energy to peak
    
        """
        ikx = np.sqrt(kx*kx/4./np.pi/np.pi*Lx*Lx)
        ikz = kz/np.pi*Lz
        return (ikx**(m/2.)/(ikx+kx_0)**m )* (ikz**(m/2.)/(ikz+kz_0)**m)

    def PeCentroid(rhohat, kkx, kkz):
        """This function computes the 'centroid' of the potential energy. It is the wave numbers
        at which the PE is centered.
        """
        pe_top = 0
        pe_bot = 0
        ikx_count = 0
        for ikx in kkx:
            iikx = np.sqrt(ikx*ikx/4./np.pi/np.pi*Lx*Lx)
            #iikx = int(ikx/2./np.pi*Lx)
            ikz_count = 0
            for ikz in kkz:
                iikz = int(ikz/np.pi*Lz)
                xwj = np.sqrt(iikx*iikx + iikz*iikz)
                pe_top = pe_top + xwj*rhohat[ikx_count, ikz_count]*rhohat[ikx_count, ikz_count].conj() 
                pe_bot = pe_bot + rhohat[ikx_count, ikz_count]*rhohat[ikx_count, ikz_count].conj()
                ikz_count += 1
            ikx_count += 1
        #print (rhohat[40,0],rhohat[40,Nz-1])
        #print(rhohat[:,80])
        return pe_top.real/pe_bot.real
