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


#Passed variables:
if len(sys.argv) > 1:
    tIdx = int(sys.argv[1])


#Program control:
ProblemType 	= 'Layers'
#ProblemType 	= 'KelvinHelmholtz'
VaryN		= 1
#ParkRun 	= 14
#ParkRun 	= 18
ParkRun 	= -1
scalePert	= 0
#N2		= 0.09
#N2		= 0.25
#N2 		= 1
N2		= 2.25		
#N2 		= 3.83
#N2		= 6.25
#N2 		= 9
#N2		= 16
#N2		= 25

#User must make sure correct data is read in for some analysis:
#var_nms = ['psi']
var_nms = ['S']
#var_nms = ['psi','S']
Nvars = len(var_nms)

#Choose which diagnostics to compute:
#n.b. code has been written to make each switch/variable 
#largely independent of the others. This makes it easier for the
#user and helped to make the code more object orientated/modular to 
#minimise repetition.
FullFields              = 1
StatePsi                = 0
StateS                  = 0
PE                      = 0
Wind                    = 0
dSdz                    = 0
dUdz                    = 0
Richardson              = 0
SpectralAnalysisT       = 0
PSDoverviewPlot		= 0
TimeSeriesAnalysisPE    = 0
FourierSinCosBasis      = 0
Vorticity               = 0
KE                      = 0
PE                      = 0
TotalEnergy             = 0
SpectralAnalysisZ       = 0
TrackSteps		= 1

NaturalBasis            = 0
nvars	            	= 2
BasisCheck1             = 0
PlotState2		= 0
BasisCheck2             = 0


#General statistical processing:
xMean = 0
tMean = 0
SlidingMean = 0

#Choose type of plot:
MakePlot = 1
PlotXZ = 0
PlotTZ = 0
PlotProfileZ = 0
MakeMovie = 0

#Write analysis to file
w2f_analysis = 0

if VaryN == 0:
    #Options when reading data:
    dir_state = './Results/State/'
    #dir_state = './Results/State_mesh0/'
    #dir_state = './Results/State_mesh1/'
    #dir_state = './Results/State_mesh2/'
    #dir_state = './Results/State_mesh3/'
    #dir_state = './Results/State_mesh4/'
    #dir_state = './Results/State18_01Spert0/'
    #dir_state = './Results/State18_05Spert0/'
    #dir_state = './Results/State18_10Spert0/'
if VaryN == 1:
    if N2 == 0.09: 	RunName = 'StateN2_00_09/'
    if N2 == 0.25: 	RunName = 'StateN2_00_25/'
    if N2 == 1: 	RunName = 'StateN2_01/'
    if N2 == 2.25: 	RunName = 'StateN2_02_25/'
    if N2 == 3.83:	RunName = 'State18/'
    if N2 == 6.25:	RunName = 'StateN2_06_25/'
    if N2 == 9:		RunName = 'StateN2_09/'
    if N2 == 16:	RunName = 'StateN2_16/'
    if N2 == 25:	RunName = 'StateN2_25/'
    dir_state = './Results/' + RunName

StartMin = 1
#StartMin = 31
#nfiles = 95
nfiles = 1

#Model output/write timestep:
dt = 1./10

#Analysis timestep:
#(This is important for computations involving numerous large arrays - 
#it avoids exceeding system memory limits).
#Effectively we use a subset of the model output data for the analysis:
#dt2 = dt
#dt2 = 1/5.
#dt2 = 1/2.
dt2 = 1.
#dt2 = 2.

secPerFile = 60.

#Choose sliding-window length for averaging.
#N.B. some data will be lost from start and end of time period.
#N.B. Choose an odd length to make window symmetric around some t point:
#Nt_mean = 1801
Nt_mean = 3001
wing = Nt_mean//2


#Set up grid and related objects:
Lx = 0.2
Lz = 0.45
#factor = 1./4
#factor = 1./2
factor = 1
#factor = 2
#factor = 4
Nx = 80
Nz = 180
Nx = int(Nx*factor)
Nz = int(Nz*factor)
if factor == 1./4: Nz += 1

x_basis = de.Fourier('x', Nx, interval=(0, Lx), dealias=3/2)
z_basis = de.SinCos('z', Nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
x=domain.grid(0)[:,0]
z=domain.grid(1)[0,:]

dx = x[1]-x[0]
dz = z[1]-z[0]

#Construct some general arrays for contour plots:
ntPerFile = secPerFile/dt
tq = dt2/dt
Nt = ntPerFile*nfiles/tq
Nt = int(Nt)

x2d = np.tile(x,(Nz,1))
z2d = np.tile(z,(Nx,1)).transpose()
t = np.arange(Nt)*dt2 + (StartMin-1)*secPerFile

t2d_z = np.tile(t,(Nz,1))
z2d_t = np.tile(z,(Nt,1)).transpose()


#Set physical constants and related objects:
g = 9.81
ct = 2*10**(-4.)
cs = 7.6*10**(-4.)

if ParkRun == 18:               #for lab run 18 of Park et al
    N2 = 3.83
    drho0_dz = -425.9		
    rho0 = -g/N2*drho0_dz
    scalePert = 1
if ParkRun == 14:               #for lab run 14 of Park et al
    N2 = 0.35
    drho0_dz = -31.976
    rho0 = -g/N2*drho0_dz
    scalePert = 1
if ParkRun < 0:
    #Set ref density constant:
    N2_18 = 3.83
    drho0_dz_18 = -425.9
    rho0 = -g/N2_18*drho0_dz_18

    #Find initial background density field given some 
    #user defined N^2 and the constant ref density: 
    drho0_dz = -N2*rho0/g

if ProblemType == 'Layers':
    bt = 0.
    bs = -1./(rho0*cs)*drho0_dz
if ProblemType == 'KelvinHelmholtz':
    bt = 0.
    bs = 0.

if FullFields == 1:
    drho0_dz13 = -122.09
    dgamma = 100./3
    dz_b = 2./100
    a0 = 100.
    z_a = Lz/2
    rhoprime13 = dgamma*z_a + a0*dz_b + dgamma/2*dz_b

    if scalePert == 1:
        rhoprime = rhoprime13 * drho0_dz/drho0_dz13
        Spert0 = rhoprime * 1./(rho0*cs)
    if scalePert == 0:
        rhoprime = rhoprime13 * drho0_dz_18/drho0_dz13
        Spert0 = rhoprime * 1./(rho0*cs)

    TTop = 0.
    STop = Spert0
    Tbase = np.zeros(Nz) + TTop
    Sbase = -bs*(z-Lz) + STop


#Set general plotting parameters assuming A4 page size:
A4Width = 8.27
MarginWidth = 1
width = A4Width-2*MarginWidth
height = 4
#For scaling the A4 plot dimensions:
ScaleFactor = 1
#ScaleFactor = 0.7
width = width*ScaleFactor
height = height*ScaleFactor


#Read in Dedalus results:
fileIdx = np.arange(nfiles) 

if 'psi' in var_nms:
    Psi = np.zeros((Nt,Nx,Nz))
if 'T' in var_nms:
    T = np.zeros((Nt,Nx,Nz))
if 'S' in var_nms:
    S = np.zeros((Nt,Nx,Nz))

for jj in range(0,Nvars):
    for ii in fileIdx:
        fnm = dir_state + 'State_s' + str(ii+StartMin) + '.h5'
        hdf5obj = h5py.File(fnm,'r')
        tmp_ = hdf5obj.get('tasks/'+var_nms[jj])
        idxS = ii*ntPerFile/tq
        idxE = (ii+1)*ntPerFile/tq
        idxS = int(idxS)
        idxE = int(idxE)
        print(idxS,idxE)

        if var_nms[jj] == 'psi':
            Psi[idxS:idxE,:,:] = np.array(tmp_)[::int(tq),:,:]
        if var_nms[jj] == 'S':
            S[idxS:idxE,:,:] = np.array(tmp_)[::int(tq),:,:]

        hdf5obj.close()


#Define some useful functions:
def x_mean(f):
    data = np.mean(f,1)
    data = data.transpose()
    return data

def t_mean(f):
    #Average across full time period:
    data = np.mean(f,0)
    data = data.transpose()
    return data

def sliding_mean(f,Nt,Nx,Nz,wing):
    data = np.zeros((Nt,Nx,Nz))
    for tt in range(wing,Nt-wing):
        data[tt,:,:] = np.mean(f[tt-wing:tt+wing,:,:],0)
    return data

def d_dz(f,Nt,Nx,Nz,z):
    fz = np.zeros((Nt,Nx,Nz))
    for jj in range(1,Nz-1):
        dz = z[jj+1] - z[jj-1]
        for ii in range(1,Nx-1):
            df = f[:,ii,jj+1] - f[:,ii,jj-1]
            fz[:,ii,jj] = df/dz
    return fz

def d_dx(f,Nt,Nx,Nz,x):
    fx = np.zeros((Nt,Nx,Nz))
    for jj in range(1,Nz-1):
        for ii in range(1,Nx-1):
            dx = x[ii+1]-x[ii-1]
            df = f[:,ii+1,jj] - f[:,ii-1,jj]
            fx[:,ii,jj] = df/dx
    return fx


if FullFields == 1:
    #add base state:
    S += Sbase


if StatePsi == 1:
    data = Psi
    if MakePlot == 1:
        PlotTitle = ''
        FigNmBase = 'psi'
        clevels = 50
        cmap = 'PRGn'


if StateS == 1:
    data = S
    if MakePlot == 1:
        PlotTitle = 'S'
        FigNmBase = 'S'
        clevels = 50
        cmap = 'PRGn'


if Wind == 1:

    #tmp1 = domain.new_field()
    #tmp1.meta['z']['parity'] = -1
    #tmp2 = domain.new_field()
    #tmp2.meta['z']['parity'] = -1
    #u = domain.new_field()
    #u.meta['z']['parity'] = 1
    #w = domain.new_field()
    #w.meta['z']['parity'] = -1

    #tmp1['g'] = data[tt,:,:]
    #tmp1.differentiate('z',out=u)
    #tmp2['g'] = data[tt,:,:]
    #tmp2.differentiate('x',out=w)
    #w['g'] *= -1

    data = d_dz(Psi,Nt,Nx,Nz,z)
    data2 = -d_dx(Psi,Nt,Nx,Nz,x)

    if MakePlot == 1:

        PlotTitle = r'$u$'
        PlotTitle2 = r'$w$'
        FigNmBase = 'Wind'
        cmap = 'bwr'

        nlevs = 41
        scaleU = 1./1000
        u_max = 1.0*scaleU
        u_min = -1.0*scaleU
        
        du = (u_max-u_min)/(nlevs-1)
        clevels = np.arange(nlevs)*du + u_min        

        nlevs = 41
        scaleW = 1./1000
        w_max = 1.*scaleW
        w_min = -1.*scaleW
        dw = (w_max-w_min)/(nlevs-1)
        clevels2 = np.arange(nlevs)*dw + w_min

        #clevels = 50
        #clevels2= 50


if Vorticity == 1:

    Psi_x = d_dx(Psi,Nt,Nx,Nz,x)
    Psi_z = d_dz(Psi,Nt,Nx,Nz,z)
    Psi_xx = d_dx(Psi_x,Nt,Nx,Nz,x)
    Psi_zz = d_dz(Psi_z,Nt,Nx,Nz,z)
    data = -(Psi_xx + Psi_zz)

    clevels = 40
    clevels2= 40

    nlevs = 41
    scaleVort = 1./1
    Vort_max = 2.0*scaleVort
    Vort_min = -2.0*scaleVort

    dVort = (Vort_max-Vort_min)/(nlevs-1)
    clevels = np.arange(nlevs)*dVort + Vort_min
 
    cmap = 'bwr'
    PlotTitle = ''

if KE == 1:

    u = d_dz(Psi,Nt,Nx,Nz,z)
    w = -d_dx(Psi,Nt,Nx,Nz,x)
    data = 1./2*( u**2 + v**2 )


if PE == 1:
    #N.B. we need to convert from perturbation S to full S.
    #Compute a mean full salinity using base state (assuming S'<<Sbase):
    S0 = mean(Sbase)
    #Compute density from salinity using equation of state for full fields:
    rho = rho0*(1 + cs*(S-S0))

    #Compute a new z array to reduce following loop
    #There are a large number of time points - we wish to avoid looping 
    #over the time axis:
    z_arr = np.tile(z,(Nt,1))

    #Compute potential energy per unit volume for each grid point:
    PE = np.zeros((Nt,Nx,Nz))
    for ii in range(0,Nx):
        PE[:,ii,:] = S[:,ii,:]*g*z_arr

    #Find total PE in each column at each time:
    PEcolumn = np.zeros((Nt,Nx))
    for ii in range(0,Nx):
        PEcolumn[:,ii] = sum(PE[:,ii,:],1)


if dSdz == 1:    
    data = d_dz(S,Nt,Nx,Nz,z)

    if MakePlot == 1:

        PlotTitle = r'$\partial S/\partial z$ (g/kg/m)'
        FigNmBase = 'dSdz'
        cmap = 'PRGn'

        nlevs = 41
        if ParkRun == 18: 
            SzMin = -1000
            SzMax = 1000
        if ParkRun == 14:
            SzMin = -100
            SzMax = 100
        if ParkRun < 0:
            if N2==0.1 or N2==0.25 or N2==0.09:
                SzMin = -100
                SzMax = 100
            if N2==1:
                SzMin = -300
                SzMax = 300
            if N2==2.25:
                SzMin = -500
                SzMax = 500
            if N2==6.25:
                SzMin = -1500
                SzMax = 1500
            if N2==9:
                SzMin = -2000
                SzMax = 2000
            if N2==16:
                SzMin = -3000
                SzMax = 3000
            if N2==25:
                #dS = (Sbase[1]-Spert0/2) - (Sbase[0]+Spert0/2)
                #dz = Lz/(Nz-1)
                #SzMin = round(dS/dz,-2)
                #SzMax = -SzMin
                SzMin = -5000
                SzMax = 5000
        dSz = (SzMax - SzMin)/(nlevs-1)
        clevels = np.arange(nlevs)*dSz + SzMin

if dUdz == 1:

    u = d_dz(Psi,Nt,Nx,Nz,z)
    w = -d_dx(Psi,Nt,Nx,Nz,x)
    data = d_dz(u,Nt,Nx,Nz,z)
    data2 = d_dz(w,Nt,Nx,Nz,z)

    if MakePlot == 1:

        PlotTitle = r'$\partial u/\partial z$'
        PlotTitle2 = r'$\partial w/\partial z$'
        FigNmBase = 'dUdz' 
        cmap = 'bwr'

        nlevs = 41
        uz_max = 20.
        uz_min = -20.
        duz = (uz_max-uz_min)/(nlevs-1)
        clevels = np.arange(nlevs)*duz + uz_min

        nlevs = 41
        wz_max = 20.
        wz_min = -20.
        dwz = (wz_max-wz_min)/(nlevs-1)
        clevels2 = np.arange(nlevs)*dwz + wz_min


if Richardson == 1:
    Sz = d_dz(S,Nt,Nx,Nz,z)
    u = d_dz(Psi,Nt,Nx,Nz,z)
    uz = d_dz(u,Nt,Nx,Nz,z)

    N2_local = np.zeros((Nt,Nx,Nz)) 
    Ri_local = np.zeros((Nt,Nx,Nz))
    for jj in range(1,Nz-1):
        for ii in range(1,Nx-1):
            N2_local[:,ii,jj] = -g*cs*Sz[:,ii,jj] 
            Ri_local[:,ii,jj] = N2_local[:,ii,jj]/uz[:,ii,jj]**2
            
            #Set inf points to arbitrary value for plotting purposes 
            #Python contour routine will not colour inf regions
            #LogicalInf = np.isinf( Ri_local[:,ii,jj] )
            #idxsInf = np.where( LogicalInf == 1 ) 
            #if len(idxsInf[0]) != 0:
            #    Ri_local[idxsInf[0],ii,jj] = -999
            #N.B. there are only a very few inf points so this code section is not essential - it has minimal impact on plots.

    data = Ri_local

    if MakePlot == 1:
        PlotTitle = r'$Ri$'
        FigNmBase = 'Ri'
        cmap = 'PRGn'

        nlevs = 41
        RiC = 1/4.
        RiRange = 2.
        dRi = RiRange/(nlevs-1)
        clevels = np.arange(nlevs)*dRi - (RiRange/2.-RiC)


if FourierSinCosBasis == 1:

    data = S
    #data = Psi

    #Compute Fourier-SinCos basis coefficients:    
    tmp = domain.new_field()
    tmp.meta['z']['parity'] = -1
    data_hat = np.zeros( (Nt,Nx,Nz), dtype=complex )

    for tt in range(0,Nt):
        tmp['g'] = data[tt,:,:]
        #Add coefficients for positive wavenumbers given by Dedalus:
        data_hat[tt,0:int(Nx/2.),:] = tmp['c']

        #Define coefficients for negative wavenumbers in x using
        #Fourier transform symmetry relation for Real functions: 
        for j in range(0,Nz):
            for i in range(0,int(Nx/2.)):
                if i > 0:
                    #Apply symmetry property of real-valued functions:
                    data_hat[tt,Nx-i,j] = np.conj(tmp['c'][i,j])

    #check symmetry of coefficients:
    #plt.contourf(data_hat[0,:,:],50)   
    #plt.show()
    #pdb.set_trace()
    
    if MakePlot == 1:
        
        count1 = 0
        #linewidthvec = [3,3,3,3,3,3,3,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
        #linecolorvec = ['k','b','c','g','y','m','r','k','b','c','g','y','m','r'] 
        #linecolorvec = ['black','silver','blue','cyan','lightblue','green','orange','gold','red','m'] 
        linecolorvec = ['black','silver','m','red','blue','cyan','lightblue','green','orange','gold'] 
        
        for jj in range(0,Nz):
            for ii in range(0,int(Nx/2.)):
            
                #f = abs(data_hat[:,ii,jj])
                f = abs(data_hat[:,ii,jj])**2
                 
                #limit = 0
                #PowerLimit = 10**(2)
                #PowerLimit = 5*10**(2)
                PowerLimit = 10**(3)
                #PowerLimit = 5*10**(4)
                #PowerLimit = 5*10**(5)
                if sum(f) > PowerLimit: 

                    label = str(ii) + ',' + str(jj)
                    color = linecolorvec[count1]

                    #plt.semilogy(f, linewidth=linewidth, color=color, label=label)
                    plt.semilogy(t, f, linewidth=2, color=color, label=label)
                    #plt.semilogy(f)
         
                    count1 = count1 + 1
        
        print(count1)
        plt.xlabel('t (s)')
        plt.ylim(10**(-4),10**(2))
        plt.legend()
        plt.show()
        

if SpectralAnalysisT == 1:
   
    data = S
    #data = Psi

    #Look at time series for point(s) in domain:
    Welch = 1

    signal_f = 1./dt				#Sampling frequency (needs to be units 1/s for Welch method)
    BigFreqArr = np.zeros((Nx,Nz))   
    f0 = 0.1

    if Welch == 0:
        SpectralCoef = np.zeros((int(Nt/2.)+1,Nx,Nz))
        freqvec = np.arange(Nt/2.+1)*1./Nt      #assumes dt=1
                                                #Note array length is Nt/2 and so max freq. will be Nyquist freq.
        freqvec         = freqvec*signal_f      #uses actual signal frequency (Dedalus timestep)
        #freqvec        = freqvec*2*np.pi       #converts to angular frequency (rad/s) - but less intuitive
    
    if Welch == 1:
        #Essentially this method divides signal into a number of segments that are nperseg points long. 
        #A form of spectral analysis (periodogram) is performed on each segment.
        #Then an average periodogram is computed from the set of periodograms.
        #By default the segments overlap and the overlap is nperseg/2.
        #So if nperseg is longer then we have higher frequency resolution but possibly more noise.
        dnmntr = 10
        #dnmntr = 50
        nperseg = int(Nt/dnmntr)
        SpectralCoef = np.zeros((int(nperseg/2.)+1,Nx,Nz))

    #Perform spectral analysis on time series at each grid point:
    for jj in range(0,Nz):
        for ii in range(0,Nx):

            ts = data[:,ii,jj]

            if Welch == 0: 
                #ts = ts*np.hanning(Nt)		#apply Hanning window function to make periodic signal
                ts_hat = abs( np.fft.fft(ts) )
                #ts_power = ts_hat**2
                SpectralCoef[:,ii,jj] = ts_hat[0:int(Nt/2.)+1]

                #Remove slow flows:
                fIdxVec = np.where(freqvec <= f0)
                fIdx = max(fIdxVec[0])

                ts_hat2 = ts_hat[fIdx:int(Nt/2.)+1]
                idx = np.where( ts_hat2 == max(ts_hat2) )
                BigFreqArr[ii,jj] = freqvec[idx[0] + fIdx]  
            
            if Welch == 1:
                freqvec, psd = welch( ts,
                                      fs=signal_f,		# sampling rate
                                      window='hanning',		# apply a Hanning window before taking the DFT
                                      nperseg=nperseg,		# compute periodograms of nperseg-long segments of ts
                                      detrend='constant')	# detrend ts by subtracting the mean
                
                SpectralCoef[:,ii,jj] = psd
                #Remove slow flows:
                fIdxVec = np.where(freqvec <= f0)
                fIdx = max(fIdxVec[0])
                
                ts_hat2 = psd[fIdx:int(nperseg/2.)+1]
                #Find dominant frequencies:
                idx = np.where( ts_hat2 == max(ts_hat2) )
                BigFreqArr[ii,jj] = freqvec[idx[0] + fIdx]
    
    if w2f_analysis == 1:
         xIdx = int(Nx/2.)
         zIdx = int(Nz/2.)
         f = SpectralCoef[:,xIdx,zIdx]
         if N2 == 25:	fnm = './psd_N2_25' + '_' + str(nperseg) + '.txt'
         if N2 == 9:	fnm = './psd_N2_09' + '_' + str(nperseg) + '.txt'
         if N2 == 3.83:	fnm = './psd18' + '_' + str(nperseg) + '.txt'
         if N2 == 1:	fnm = './psd_N2_01' + '_' + str(nperseg) + '.txt'
         if N2 == 0.09:	fnm = './psd_N2_00_09' + '_' + str(nperseg) + '.txt'
         np.savetxt(fnm,f)

    if MakePlot == 1:

        if PSDoverviewPlot == 0:
            fig = plt.figure(1, figsize=(width,height))
            #grid = plt.GridSpec(1, 4, wspace=0.5, hspace=0.)
            grid = plt.GridSpec(1, 2, wspace=0.5, hspace=0.)
 
            #ax1 = fig.add_subplot(grid[0,:2])
            #ax1.set_xlim(0,Lx)
            #ax1.set_ylim(0,Lz)

            #rangeF = 5.
            #nf = 51 
            #df = rangeF/(nf-1)
            #clevels = np.arange(nf)*df
            #i1 = ax1.contourf(x2d,z2d,BigFreqArr.transpose(),clevels,cmap='seismic',extend="both")
            #plt.contourf(x2d,z2d,BigFreqArr.transpose(),cmap='seismic')
            #fig.colorbar(i1, orientation="horizontal")
            #plt.colorbar(i1)

            xIdx = int(Nx/2.)
            nz = 6
            dz = 0.05
            zmin = 0.1   #avoid boundary layers
            zpnts = np.arange(nz)*dz + zmin
            zIdxs = zpnts/(Lz/Nz)
            zIdxs = zIdxs
            xlim = 4

            colorvec = ('k','grey','g','b','c','gold')

            for ii in range(0,nz):

                f = SpectralCoef[:,xIdx,int(zIdxs[ii])]
                label = str(zpnts[ii])

                if ii == 0:
                    #ax2 = fig.add_subplot(grid[0,2])
                    ax2 = fig.add_subplot(grid[0,0])
                    ax2.set_xlabel(r'$f$' ' (Hz)')
                    ax2.set_xlim(0,xlim)
                    ax2.set_xlabel(r'$f$' ' (Hz)')
                    ax2.set_ylabel('PSD')
                ax2.plot(freqvec,f, color=colorvec[ii], label=label)
                ax2.legend()

                if ii == 0:
                    #ax3 = fig.add_subplot(grid[0,3])
                    ax3 = fig.add_subplot(grid[0,1])
                    ax3.set_xlim(0,xlim)
                    ax3.set_xlabel(r'$f$' ' (Hz)')
                    ax3.set_ylabel('PSD')
                ax3.semilogy(freqvec,f, color=colorvec[ii], label=label)
                ax3.legend()
                #Show position of analysis points on contour plot:
                #ax1.plot( [x[xIdx],x[xIdx]], [z[zIdxs[ii]],z[zIdxs[ii]]], '+' )

            plt.show()

        if PSDoverviewPlot == 1:
            #npersegVal = '180'
            npersegVal = '900'

            s_N2_00_09 = np.loadtxt('./psd_N2_00_09' + '_' + npersegVal + '.txt')
            s_N2_01 = np.loadtxt('./psd_N2_01' + '_' + npersegVal + '.txt')
            s18 = np.loadtxt('./psd18' + '_' + npersegVal + '.txt')
            s_N2_09 = np.loadtxt('./psd_N2_09' + '_' + npersegVal + '.txt')
            s_N2_25 = np.loadtxt('./psd_N2_25' + '_' + npersegVal + '.txt')

            colorvec = ('k','grey','g','b','c')
            labelvec = ('N=0.03','N=1', 'N=1.96', 'N=3', 'N=5')
            plt.xlim(0,1.5)
            plt.xlabel(r'$f$ (Hz)')
            #plt.ylim(0,1)
            plt.ylabel('PSD')
            plt.semilogy(freqvec,s_N2_00_09, color=colorvec[0], label=labelvec[0])
            plt.semilogy(freqvec,s_N2_01, color=colorvec[1], label=labelvec[1])
            plt.semilogy(freqvec,s18, color=colorvec[2], label=labelvec[2])
            plt.semilogy(freqvec,s_N2_09, color=colorvec[3], label=labelvec[3])
            plt.semilogy(freqvec,s_N2_25, color=colorvec[4], label=labelvec[4])
            plt.legend()
            plt.show()

if TimeSeriesAnalysisPE == 1:    
    #Look at time series of change of total PE in each column
    Welch = 1
    signal_f = 1./dt                             #Sampling frequency (needs to be units 1/s for Welch method)
 
    if Welch == 1:
        dnmntr = 50
        nperseg = int(Nt/dnmntr)
        SpectralCoef = np.zeros((int(nperseg/2.)+1,Nx))

    #Perform spectral analysis on time series of change of total PE in each column:
    for ii in range(0,Nx):
        ts = PEcolumn[:,ii]
        if Welch == 1:
            freqvec, psd = welch( ts,
                                  fs=signal_f,              # sampling rate
                                  window='hanning',         # apply a Hanning window before taking the DFT
                                  nperseg=nperseg,          # compute periodograms of nperseg-long segments of ts
                                  detrend='constant')       # detrend ts by subtracting the mean

            SpectralCoef[:,ii] = psd

    if MakePlot == 1:
        xIdx = 20

        plt.figure(1)
        plt.subplot(121) 
        plt.plot(freqvec,SpectralCoef[:,xIdx])
        plt.xlabel(r'$f$' ' (Hz)')
        plt.subplot(122)
        plt.semilogy(freqvec,SpectralCoef[:,xIdx])
        plt.xlabel(r'$f$' ' (Hz)')

        plt.show()

if TrackSteps == 1:

    Sz = d_dz(S,Nt,Nx,Nz,z)
    step_mask = np.zeros((Nt,Nx,Nz),dtype=bool)
    step_count = np.zeros((Nt))
    step_dz = np.zeros((Nt,50))
    step_dS = np.zeros((Nt,50))

    #Exclude boundary layer effects:
    if (N2 == 0.09) or (N2 == 0.25): zIdx_offset = int(.1/dz)
    if (N2 != 0.09) and (N2 != 0.25): zIdx_offset = int(.05/dz)

    #Exclude initial chaos (depends on N2):
    if N2 == 0.09: tIdx_offset = int(15./dt2)
    if N2 == 0.25: tIdx_offset = int(20./dt2)
    if N2 == 1: tIdx_offset = int(10./dt2)
    if N2 == 2.25: tIdx_offset = int(7./dt2)
    if N2 == 3.83: tIdx_offset = int(6./dt2)
    if N2 == 6.25: tIdx_offset = int(6./dt2)
    if N2 == 9: tIdx_offset = int(3./dt2)
    if N2 == 16: tIdx_offset = int(2./dt2)
    if N2 == 25: tIdx_offset = int(1./dt2)

    #Choose x point for analysis
    xIdx = int(Nx/2.)
   
    #Search for steps by searching for relatively small 
    #vertical gradients of salinity field. An iterative approach is 
    #used where we start searching for gradients less than some initial value
    #and continue iterating until we find the maximum number of steps for each t.

    for tt in range(0+tIdx_offset,Nt):

        #epsilon = 1.
        epsilon = np.min(abs(Sz[tt,xIdx,:]))

        MaxSteps0 = 0
        MaxSteps1 = 1
        count0 = 0

        while MaxSteps1 >= MaxSteps0: 
            print('iteration: ',count0)
            if count0 != 0: MaxSteps0 = MaxSteps1

            tmp1 = np.zeros((Nx,Nz),dtype=bool)
            tmp3 = np.zeros((50))
            tmp4 = np.zeros((50))

            #Define step regions for given epsilon(t):
            for jj in range(0+zIdx_offset,Nz-zIdx_offset):
                if (Sz[tt,xIdx,jj] <= 0):         
                    if abs(Sz[tt,xIdx,jj]) <= epsilon:
                        tmp1[xIdx,jj] = True
                    else:
                        tmp1[xIdx,jj] = False                 
                else:
                    tmp1[xIdx,jj] = False
       
            #Count number of steps and compute step quantities: 
            count = 0
            flag = 0
            for jj in range(0+zIdx_offset,Nz-zIdx_offset):
                if (tmp1[xIdx,jj] == True) and (flag == 0):
                    count += 1
                    flag = 1
                    j0 = jj
                if (tmp1[xIdx,jj] == False):
                    flag = 0
                    if count != 0:
                        dz = z[jj]-z[j0]
                        Sz_mean = np.mean(Sz[tt,xIdx,j0:jj])
                        if (MaxSteps1 > MaxSteps0):
                            tmp3[count-1]=dz
                            tmp4[count-1]=Sz_mean
                if (count==1) and (flag==1) and (jj==Nz-zIdx_offset-1):
                    dz = z[jj]-z[j0]
                    Sz_mean = np.mean(Sz[tt,xIdx,j0:jj])
                    if (MaxSteps1 > MaxSteps0):
                        tmp3[count-1]=dz
                        tmp4[count-1]=Sz_mean
 
            MaxSteps1 = count
            print('max # steps: ', MaxSteps1, MaxSteps0, ' epsilon: ', epsilon)

            if (MaxSteps1 >= MaxSteps0):
                if epsilon != 0: epsilon = epsilon + epsilon*0.001
                else: epsilon = epsilon + (epsilon+0.1)*0.001

        #    if (MaxSteps1 > MaxSteps0):
        #        step_mask[tt,:,:] = tmp1
        #        step_count[tt] = count
        #        step_dz[tt,:] = tmp3
        #        step_dS[tt,:] = tmp4
        #        print('array update: ', count0)

            #If no steps found on 1st iteration, then reset ICs and search using new epsilon(t):
            #if (count0 == 0) and (MaxSteps1==MaxSteps0): MaxSteps1 = 1

            count0 += 1

        step_mask[tt,:,:] = tmp1
        step_count[tt] = count
        step_dz[tt,:] = tmp3
        step_dS[tt,:] = tmp4
        #print('array update: ', count0)





    if w2f_analysis == 1:
        dir_TrackSteps = './Results/' + RunName + '/TrackSteps/'
        #Create directory if it doesn't exist:
        if not os.path.exists(dir_TrackSteps):
            os.makedirs(dir_TrackSteps)

        fnm1 = dir_TrackSteps + 'steps_t.txt'
        fnm2 = dir_TrackSteps + 'steps_dz.txt'
        fnm3 = dir_TrackSteps + 'steps_dS.txt'
        np.savetxt(fnm1,step_count)
        np.savetxt(fnm2,step_dz)
        np.savetxt(fnm3,step_dS)


    if MakePlot == 1:

        fig=plt.figure(figsize=(width,height))
        grid = plt.GridSpec(1, 1, wspace=0.4, hspace=0.4)
        ax1 = fig.add_subplot(grid[0,0])

        plot_data = step_mask[:,xIdx,:].transpose().astype(int)
        xgrid = t2d_z
        ygrid = z2d_t
        i1 = ax1.contourf(xgrid,ygrid,plot_data, 1, colors=['white','black'])
        ax1.set_xlabel(r'$t$ (s)')
        ax1.set_ylabel(r'$z$ (m)')
        cb = plt.colorbar(i1, ticks=[0,1])
 
        #ax2 = fig.add_subplot(grid[0,1])
        #ax2.plot(t,step_count)
        #ax2.set_xlabel(r'$t$ (s)')
        #ax2.set_ylabel('Number of steps')

        #ax3 = fig.add_subplot(grid[1,1])
        #ax3.hist(step_count, bins=np.arange(20)+1)
        #ax3.set_xlabel('Number of steps')
        #ax3.set_ylabel('Count')

        plt.show()
        pdb.set_trace()


if NaturalBasis == 1:

    # Eigenvectors are not a fnc of time.
    # There is one eigenvector for each alpha.
    # Eigenvectors are assumed to be a fnc of k and m wavenumbers 
    # (see structure of NaturalFreq.py).
    # An eigenvector, for some k and m, has nvar elements.
    ivec_1 = np.zeros((Nx,Nz,nvars))
    if nvars == 3: ivec0 = np.zeros((Nx,Nz,nvars))
    ivec1 = np.zeros((Nx,Nz,nvars))

    if BasisCheck1 == 1:
        State2 = np.zeros((Nx,Nz,Nt,nvars))

    if BasisCheck2 == 1:
        #Store wavenumber-vector magnitude array to transform back to original State 
        #in NaturalFreq.py 
        kmag_arr = np.zeros((Nx,Nz))

    #Initialise Dedalus objects and arrays for computing Fourier-SinCos coefficients:
    tmp_Psi = domain.new_field()
    tmp_Psi.meta['z']['parity'] = -1
    if nvars == 3: 
        tmp_T = domain.new_field()
        tmp_T.meta['z']['parity'] = -1
    tmp_S = domain.new_field()
    tmp_S.meta['z']['parity'] = -1

    Psi_hat = np.zeros((Nx,Nz))*1j
    if nvars == 3: T_hat = np.zeros((Nx,Nz))*1j
    S_hat = np.zeros((Nx,Nz))*1j

    #Set-up wavenumbers for Dedalus grid:
    kk = np.zeros(Nx)
    kkx = x_basis.wavenumbers
    kk[0:int(Nx/2.)] = kkx
    dk = kkx[1]-kkx[0]
    nyquist_f = -(np.max(kkx) + dk)
    kk[int(Nx/2.)] = nyquist_f
    kkx_neg = np.flipud(kkx[1:]*(-1))
    kk[int(Nx/2.)+1:] = kkx_neg

    #plt.plot(kk,'-+')
    #plt.show()
    #print(kk) 
    #pdb.set_trace()

    kk_cosine = z_basis.wavenumbers

    for tt in range(0,Nt):

        print(tt)

        tmp_Psi['g'] = Psi[tt,:,:]
        if nvars == 3: tmp_T['g'] = T[tt,:,:]
        tmp_S['g'] = S[tt,:,:]

        #Add coefficients for positive wavenumbers given by Dedalus:
        Psi_hat[0:int(Nx/2.),:] = tmp_Psi['c']
        if nvars == 3: T_hat[0:int(Nx/2.),:] = tmp_T['c']
        S_hat[0:int(Nx/2.),:] = tmp_S['c']

        #Define coefficients for negative wavenumbers in x using
        #Fourier transform symmetry relation for Real functions: 
        for j in range(0,Nz):
            for i in range(0,int(Nx/2.)):
                if i > 0:
                    #Apply symmetry property of real-valued functions:
                    Psi_hat[Nx-i,j] = np.conj(tmp_Psi['c'][i,j])
                    if nvars == 3: T_hat[Nx-i,j] = np.conj(tmp_T['c'][i,j])
                    S_hat[Nx-i,j] = np.conj(tmp_S['c'][i,j])

        # There is one sigma for each alpha.
        # Sigmas are complex and depend on k and m wavenumbers, and vary through time, 
        # so here we re-define the arrays for each time point in the loop. 
        # n.b. _1: alpha=-1, 0: alpha=0, 1: alpha=+1 
        sigma_1 = np.zeros((Nx,Nz))*1j
        if nvars == 3: sigma0 = np.zeros((Nx,Nz))*1j
        sigma1 = np.zeros((Nx,Nz))*1j

        #Loop over wavenumbers:
        for jj in range(1,Nz):
            for ii in range(0,Nx):

                k = kk[ii]
                n = kk_cosine[jj]

                kvec    = np.array([k,n])
                kmag    = np.linalg.norm(kvec)
                if (BasisCheck2==1) and (tt==0): kmag_arr[ii,jj] = kmag

                if nvars == 3: c1 = sqrt(-ct/bt)
                c2 	= sqrt(cs/bs)
                c3      = abs(k)/kmag*sqrt(-(ct*bt-cs*bs))          #N.B. omega = c3*sqrt(g)

                #Transform spectral coefficients - required to make 
                #linear operator skew hermitian
                Psi_hat[ii,jj]        	= Psi_hat[ii,jj]*kmag
                if nvars == 3:
                    T_hat[ii,jj]       	= T_hat[ii,jj]*sqrt(g)*c1
                S_hat[ii,jj]           	= S_hat[ii,jj]*sqrt(g)*c2

                if k != 0:
                #k=0 is a special case with different eigenvectors.

                    if nvars == 3:
                        #Construct eigenvectors:
                        r0      = np.array([0,c2/c1,1]).real
                        r0mag   = np.linalg.norm(r0)
                        r0      = r0/r0mag

                        r1      = np.array([-c3/c2,-c1/c2,1]).real
                        r1mag   = np.linalg.norm(r1)
                        r1      = r1/r1mag

                        r_1     = np.array([c3/c2,-c1/c2,1]).real
                        r_1mag  = np.linalg.norm(r_1)
                        r_1     = r_1/r_1mag

                    if nvars == 2:
                        r1      = np.array([-abs(k)/k,1])
                        r1mag   = np.linalg.norm(r1)
                        r1      = r1/r1mag

                        r_1     = np.array([abs(k)/k,1])
                        r_1mag  = np.linalg.norm(r_1)
                        r_1     = r_1/r_1mag

                if k == 0:

                    if nvars == 3:
                        #Construct eigenvectors:
                        r0      = np.array([0,1,0]).real
                        r0mag   = np.linalg.norm(r0)
                        r0      = r0/r0mag

                        r1      = np.array([-1,0,1]).real
                        r1mag   = np.linalg.norm(r1)
                        r1      = r1/r1mag

                        r_1     = np.array([1,0,1]).real
                        r_1mag  = np.linalg.norm(r_1)
                        r_1     = r_1/r_1mag

                    if nvars == 2:
                        r1      = np.array([0,1])
                        r1mag   = np.linalg.norm(r1)
                        r1      = r1/r1mag

                        r_1     = np.array([1,0])
                        r_1mag  = np.linalg.norm(r_1)
                        r_1     = r_1/r_1mag

                #Store eigenvectors for printing later:
                #There are n eigenvectors, each with nvar components, where each component is assumed to 
                #depend on k and m wavenumbers (see structure of NaturalFreq.py). 
                #Eigenvectors are time invariant. 
                #There are many ways you could print the eigenvectors, each with shape (Nx,Nz,nvar).
                #I looped over the index with nvar elements and printed out each 2D array of (Nx,Nz) numbers. 
                #Python doesn't like printing 3D arrays.
                #I simply reverse this procedure when reading in my eigenvectors in NaturalFreq.py.
                if tt == 0:
                    ivec_1[ii,jj,:] = r_1
                    if nvars == 3: ivec0[ii,jj,:] = r0
                    ivec1[ii,jj,:] = r1

                #construct matrix of eigenvectors for some k and n, and
                #construct vector of spectral coefficients for some k,n 
                if nvars == 3:
                    EigenVecs = np.array([r_1,r0,r1]).transpose()
                    upsilon = np.array([Psi_hat[ii,jj],T_hat[ii,jj],S_hat[ii,jj]])
                else:
                    EigenVecs = np.array([r_1,r1]).transpose()
                    upsilon = np.array([Psi_hat[ii,jj],S_hat[ii,jj]])

                #Make transformation to find amplitudes of Natural basis 
                EigenVecs_inv = linalg.inv(EigenVecs)
                sigma_kn = np.mat(EigenVecs_inv) * np.mat(upsilon).transpose()

                if nvars == 3:
                    sigma_1[ii,jj] = sigma_kn[0,0]
                    sigma0[ii,jj] = sigma_kn[1,0]
                    sigma1[ii,jj] = sigma_kn[2,0]
                if nvars == 2:
                    sigma_1[ii,jj] = sigma_kn[0,0]
                    sigma1[ii,jj] = sigma_kn[1,0]

        if BasisCheck1 == 1:
            #Reverse transform back to the new State:
            if nvars == 3:
                tmp_Psi = domain.new_field()
                tmp_T = domain.new_field()
                tmp_S = domain.new_field()
                tmp_Psi.meta['z']['parity'] = -1
                tmp_T.meta['z']['parity'] = -1
                tmp_S.meta['z']['parity'] = -1

                tmp_Psi['c'] = Psi_hat[0:int(Nx/2.),:]
                tmp_T['c'] = T_hat[0:int(Nx/2.),:]
                tmp_S['c'] = S_hat[0:int(Nx/2.),:]

                State2[:,:,tt,0] = tmp_Psi['g']
                State2[:,:,tt,1] = tmp_T['g']
                State2[:,:,tt,2] = tmp_S['g']
            if nvars == 2:
                tmp_Psi = domain.new_field()
                tmp_S = domain.new_field()
                tmp_Psi.meta['z']['parity'] = -1
                tmp_S.meta['z']['parity'] = -1

                tmp_Psi['c'] = Psi_hat[0:int(Nx/2.),:]
                tmp_S['c'] = S_hat[0:int(Nx/2.),:]

                State2[:,:,tt,0] = tmp_Psi['g']
                State2[:,:,tt,1] = tmp_S['g']


        # For writing natural basis to a file:
        if w2f_analysis == 1:

            #First write the coefficients (sigmas) of the natural basis:
            if tt == 0:
                dir_sigma = './Results/' + RunName + '/NaturalBasis/'
                #Create directory if it doesn't exist:
                if not os.path.exists(dir_sigma):
                    os.makedirs(dir_sigma)

            fnm_sigma_1 = dir_sigma + 'sigma_1_' + str(tt) + '.txt'
            fnm_sigma1 = dir_sigma + 'sigma1_' + str(tt) + '.txt'
            np.savetxt(fnm_sigma_1,sigma_1.view(float))
            np.savetxt(fnm_sigma1,sigma1.view(float))
            if nvars == 3:
                fnm_sigma0 = dir_sigma + 'sigma0_' + str(tt) + '.txt'
                np.savetxt(fnm_sigma0,sigma0.view(float))

            #Then write out the eigenvectors (as explained above): 
            if tt == 0:
                dir_ivec = './Results/' + RunName + '/NaturalBasis/'
                #Create directory if it doesn't exist:
                if not os.path.exists(dir_ivec):
                    os.makedirs(dir_ivec)

                for ll in range(0,nvars):
                    #For complex eigenvectors:
                    #for nn in range(0,2):
                        #fnm_ivec_1 = dir_ivec + 'ivec_1_' + str(ll+1) + str(nn+1) + '.txt'
                        #fnm_ivec0 = dir_ivec + 'ivec0_' + str(ll+1) + str(nn+1) + '.txt'
                        #fnm_ivec1 = dir_ivec + 'ivec1_' + str(ll+1) + str(nn+1) + '.txt'
                        #pdb.set_trace()
                        #ivec_1_float = ivec_1.view(float)
                        #ivec0_float = ivec0.view(float)
                        #ivec1_float = ivec1.view(float)
                        #pdb.set_trace()
                    fnm_ivec_1 = dir_ivec + 'ivec_1_' + str(ll+1) + '.txt'
                    fnm_ivec1 = dir_ivec + 'ivec1_' + str(ll+1) + '.txt'
                    np.savetxt(fnm_ivec_1,ivec_1[:,:,ll])
                    np.savetxt(fnm_ivec1,ivec1[:,:,ll])
                    if nvars == 3:
                        fnm_ivec0 = dir_ivec + 'ivec0_' + str(ll+1) + '.txt'
                        np.savetxt(fnm_ivec0,ivec0[:,:,ll])

            if (BasisCheck2==1) and (tt==0):
                fnm_kmag = dir_ivec + 'kmag_arr.txt'
                np.savetxt(fnm_kmag,kmag_arr)


if PlotState2 == 1:
    
    data = State2[:,:,:,0]   

    PlotTitle = ''
    FigNmBase = 'psi'
    clevels = 50
    cmap = 'PRGn'
    tIdx = 10
   
    plt.contourf(data[:,:,tIdx].transpose(),clevels,cmap=cmap)
    plt.colorbar()
    plt.show()



#Generic stastical processing:
if xMean == 1:
    data = x_mean(data)
    if 'data2' in locals(): data2 = x_mean(data2)
    #data has shape (Nt,Nz)
if tMean == 1:
    data = t_mean(data)
    if 'data2' in locals(): data2 = t_mean(data2)
    #data has shape (Nx,Nz)
if SlidingMean == 1:
    data = sliding_mean(data,Nt,Nx,Nz,wing)
    if 'data2' in locals(): data2 = sliding_mean(data2,Nt,Nx,Nz,wing)
    #data has shape (Nt,Nx,Nz)


# For writing analysis to a file:
if w2f_analysis == 1:

    dir_analysis = './Analysis/'
    #Create directory if it doesn't exist:
    if not os.path.exists(dir_analysis):
        os.makedirs(dir_analysis)

    #incomplete:
    #fnm_analysis = dir_analysis + 'dSdz.txt'
    #xIdx = 20
    #data = Sz[:,xIdx,:]
    #np.savetxt(fnm_analysis,data)


#Plotting section:
#The intention was to try and avoid repeating the bulk of the code below 
#numerous times for different variables. Repitition has certainly been 
#significantly reduced, however, it is very hard to make a fully general
#plotting program for research purposes, which are exploratory. Currently 
#this section excludes the time-series analysis and spectral methods sections.
if (MakePlot==1 and PlotXZ==1) or (MakePlot==1 and PlotTZ==1):
    if PlotXZ == 1:
        #We define 3 options here:
        #1) consider spatial field at some t.
        #2) consider spatial field averaged across all t.
        #3) apply a sliding time-mean to data and then consider field at some t. 
        #If an average is desired then the correct switch should be used.
        if tMean != 1:
            if MakeMovie != 1:
                try:
                    tIdx
                except NameError:
                    Nhr = 0
                    Nmin = StartMin
                    Nsec = 0
                    time = Nhr*(60*60) + Nmin*60 + Nsec
                    tIdx = int((time-t[0])/dt/tq)

                data = data[tIdx,:,:].transpose()
                if 'data2' in locals():
                    data2 = data2[tIdx,:,:].transpose()
     
        #Set arrays for contouring:
        xgrid = x2d
        ygrid = z2d

    if PlotTZ == 1:
        #We define 3 options here:
        #1) consider how a single column evolves over time.
        #2) consider how the average column evolves over time.
        #3) apply a sliding time-mean to the data and then consider a single column.
        #If an average is desired then the correct switch should be used.
        if xMean != 1:
            xIdx = int(Nx/2.)
            data = data[:,xIdx,:].transpose()
            if 'data2' in locals():
                data2 = data2[:,xIdx,:].transpose()

        #Set arrays for contouring:
        xgrid = t2d_z
        ygrid = z2d_t

    if MakeMovie != 1:
            
        fig=plt.figure(figsize=(width,height))
        if 'data2' in locals():
            grid = plt.GridSpec(1, 2, wspace=0.4, hspace=0.0)
            ax1 = fig.add_subplot(grid[0,0])
        if PlotProfileZ == 1:
            grid = plt.GridSpec(1, 2, wspace=0.4, hspace=0.0)
            ax1 = fig.add_subplot(grid[0,0])
        if 'data2' not in locals() and PlotProfileZ != 1:
            grid = plt.GridSpec(1, 1, wspace=0.0, hspace=0.0)
            ax1 = fig.add_subplot(grid[0,0])

        i1=ax1.contourf(xgrid,ygrid,data,clevels,cmap=cmap,extend="both")
        fig.colorbar(i1)
        ax1.set_ylim(0,Lz)
        ax1.set_ylabel(r'$z$ (m)')
        if PlotXZ == 1: 
            ax1.set_xlim(0,Lx)
            ax1.set_xlabel(r'$x$ (m)')
            #ax1.set_title( r'$' + PlotTitle + '$' + ", " + str("%5.1f" % t[tIdx]) + " s" )
            ax1.set_title( r'$' + PlotTitle + '$' )
            start, end = ax1.get_xlim()
            ax1.xaxis.set_ticks((0,0.05,0.1,0.15,0.2))
        if PlotTZ == 1:
            ax1.set_xlabel(r'$t$ (s)')
            ax1.set_title(PlotTitle)

        if PlotProfileZ == 1:

            xIdx1 = int(Nx/2.)
            #xIdx2 = int(Nx/2.)/2
            #xIdx3 = int(Nx/2.)*3/2
            #xIdx4 = 2
            #xIdx5 = Nx-3

            #ax1.plot([x[xIdx1],x[xIdx1]],[z[0],z[Nz-1]], 'b')
            #ax1.plot([x[xIdx2],x[xIdx2]],[z[0],z[Nz-1]], 'g')
            #ax1.plot([x[xIdx3],x[xIdx3]],[z[0],z[Nz-1]], 'c')
            #ax1.plot([x[xIdx4],x[xIdx4]],[z[0],z[Nz-1]], 'm')
            #ax1.plot([x[xIdx5],x[xIdx5]],[z[0],z[Nz-1]], 'y')

            ax2 = fig.add_subplot(grid[0,1])
            ax2.set_title( r'$t$ =' + str("%5.1f" % t[tIdx]) + " s" )
            ax2.set_ylim(0,Lz)
            ax2.plot(S[tIdx,xIdx1,:],z, 'k')
            #ax2.plot(S[tIdx,xIdx2,:],z, 'g')
            #ax2.plot(S[tIdx,xIdx3,:],z, 'c')
            #ax2.plot(S[tIdx,xIdx4,:],z, 'm')
            #ax2.plot(S[tIdx,xIdx5,:],z, 'y')

            ax2.set_xlabel(r'$S$ (g/kg)')
            #ax2.set_xlim(0,300)

        if 'data2' in locals():
            ax2 = fig.add_subplot(grid[0,1])
            i2=ax2.contourf(xgrid,ygrid,data2,clevels2,cmap=cmap,extend="both")
            fig.colorbar(i2)
            ax2.set_ylim(0,Lz)
            if PlotXZ == 1: 
                ax2.set_xlim(0,Lx)
                ax2.set_title( PlotTitle2 + ", " + str("%5.1f" % t[tIdx]) + " s" )
            if PlotTZ == 1:
                ax2.set_title(PlotTitle2)
                ax2.set_xlabel(r'$t$ (s)')

        plt.show()

    if MakeMovie == 1:
        #If you wish to read in all model outputs, make a sliding average on full output,
        #but then only plot a sub-sample of the data, then you can define a plot 
        #interval here:
        if dt == dt2:
            dt_plot = 2.
            dplot = int(dt_plot/dt)
        else:
            dplot = 1

        FigPath = dir_state + 'images/'
        for tt in range(0,Nt,dplot):
            fig=plt.figure(figsize=(width,height))
            if 'data2' in locals():
                grid = plt.GridSpec(1, 2, wspace=0.4, hspace=0.0)
                ax1 = fig.add_subplot(grid[0,0])
            if PlotProfileZ == 1:
                grid = plt.GridSpec(1, 2, wspace=0.4, hspace=0.0)
                ax1 = fig.add_subplot(grid[0,0])
            if 'data2' not in locals() and PlotProfileZ != 1:
                grid = plt.GridSpec(1, 1, wspace=0.0, hspace=0.0)
                ax1 = fig.add_subplot(grid[0,0])

            i1=ax1.contourf(xgrid,ygrid,data[tt,:,:].transpose(),clevels,cmap=cmap,extend="both")
            fig.colorbar(i1)
            ax1.set_xlim(0,Lx)
            ax1.set_ylim(0,Lz)
            ax1.set_xlabel(r'$x$ (m)')
            ax1.set_ylabel(r'$z$ (m)')
            ax1.set_title(PlotTitle + ", " + str("%5.1f" % t[tt]) + " s")


            if PlotProfileZ == 1:
                xIdx1 = int(Nx/2.)
                #xIdx2 = int(Nx/2.)/2
                #xIdx3 = int(Nx/2.)*3/2
                #xIdx4 = 2
                #xIdx5 = Nx-3

                #ax1.plot([x[xIdx1],x[xIdx1]],[z[0],z[Nz-1]], 'b')
                #ax1.plot([x[xIdx2],x[xIdx2]],[z[0],z[Nz-1]], 'g')
                #ax1.plot([x[xIdx3],x[xIdx3]],[z[0],z[Nz-1]], 'c')
                #ax1.plot([x[xIdx4],x[xIdx4]],[z[0],z[Nz-1]], 'm')
                #ax1.plot([x[xIdx5],x[xIdx5]],[z[0],z[Nz-1]], 'y')

                ax2 = fig.add_subplot(grid[0,1])
                ax2.set_ylim(0,Lz)
                ax2.plot(S[tt,xIdx1,:],z, 'k')
                #ax2.plot(S[tIdx,xIdx2,:],z, 'g')
                #ax2.plot(S[tIdx,xIdx3,:],z, 'c')
                #ax2.plot(S[tIdx,xIdx4,:],z, 'm')
                #ax2.plot(S[tIdx,xIdx5,:],z, 'y')
                ax2.set_xlabel(r'$S$ (g/kg)')
                ax2.set_xlim(0,300)

            if 'data2' in locals():
                ax2 = fig.add_subplot(grid[0,1])
                i2=ax2.contourf(xgrid,ygrid,data2[tt,:,:].transpose(),clevels2,cmap=cmap,extend="both")
                fig.colorbar(i2)
                ax2.set_ylim(0,Lz)
                ax2.set_ylabel(r'$z$ (m)')
                if PlotXZ == 1: 
                    ax2.set_xlim(0,Lx)
                    ax2.set_xlabel(r'$x$ (m)')
                    ax2.set_title(PlotTitle2 + ", " + str("%5.1f" % t[tt]) + " s")
                if PlotTZ == 1:
                    ax2.set_xlabel(r'$t$ (s)')
                    ax2.set_title(PlotTitle2)

            FigNm = FigNmBase + '_' + str("%04d" % tt) + '.png'
            fig.savefig(FigPath+FigNm)
            plt.close(fig)


if BasisCheck1==1 and MakePlot==1:
    plt.contourf(State2[:,:,10,0].transpose(),50)
    plt.colorbar()
    plt.show()












pdb.set_trace()
