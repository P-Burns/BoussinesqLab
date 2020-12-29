#Code to compute State from the Natural basis and for Frequency Averaging


#Load in required libraries:
import numpy as np
from numpy import *
from scipy import *
from numpy import fft
from scipy import fftpack
import CosineSineTransforms as cst
import pdb #to pause execution use pdb.set_trace()
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.signal import welch
from dedalus import public as de
from scipy.integrate import simps
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import BoundaryNorm


#Program control:
ProblemType			= 'Layers'

ParkRun 			= -1
#N2				= 1
#N2				= 2.25
#N2				= 4
#N2				= 6.25
N2				= 7.5625
#N2				= 9
#N2				= 10.5625
#N2				= 12.25
#N2				= 14.0625
#N2				= 16
#N2				= 20.25
#N2				= 25

DiffusionFactor			= 100
FullFields 			= 0
Modulated			= 1
BasisCheck1			= 0
BasisCheck2 			= 0 
nvars 				= 2
sigma3D 			= 1

FindMainBasisParts 		= 1
PlotMainBasisParts 		= 0
PlotState_MainBasis		= 0
PlotFastMode1			= 0
PlotFastMode2			= 0
PlotSpectrum			= 0
PlotSigmaSums                   = 0
ConvergePowerLimit 		= 1
DedalusTransforms		= 1
AnalyseMainBasisParts		= 0
SpectralAnalysis		= 0
DispersionRelation		= 0

Resonances			= 1
LinearNonLinear			= 0
NearResonanceSearch		= 1

AnalyseCapOmega			= 1
listOmega			= 0
listOmegaGrad			= 0
listOmegaPhaseSpace		= 0
IntegrateCapOmega		= 1
PlotWeightFnc			= 0

contourOmega			= 0
PlotMainTriads                  = 0
FindUniqueSet                   = 0
FindMaxModes                    = 0
FindOneAlphaSet                 = 1
keyNonWaveModes                 = 0
keyWaveModes                    = 1
xAxis_k                         = 0
xAxis_n                         = 1
xAxis_r                         = 0
xAxis_alt                       = 0
xAxis_angle                     = 0
xAxis_c                         = 0
xAxis_cg_x                      = 0
xAxis_cg_z                      = 0
ManualColors			= 0
WaveSpeeds			= 0
ComputePDF			= 1
ComputeCumulativeDistFnc	= 1
InteractionCoef			= 1
PlotInteractCoef		= 0
SigmaSigma			= 0
CombineSigmaSigmaWithC 		= 0
FrequencyAve			= 1

PlotCDFsVsN			= 0


MakePlot			= 1
PlotXZ				= 0
PlotTZ				= 1
PlotT				= 0
Plot3D 				= 0

if N2 == 0.25: RunName = 'StateN2_00_25'
if N2 == 1: RunName = 'StateN2_01'
if N2 == 2.25: RunName = 'StateN2_02_25'
if N2 == 4: RunName = 'StateN2_04'
if N2 == 6.25: RunName = 'StateN2_06_25'
if N2 == 7.5625: RunName = 'StateN2_07_5625'
if N2 == 9: RunName = 'StateN2_09'
if N2 == 10.5625: RunName = 'StateN2_10_5625'
if N2 == 12.25: RunName = 'StateN2_12_25'
if N2 == 14.0625: RunName = 'StateN2_14_0625'
if N2 == 16: RunName = 'StateN2_16'
if N2 == 20.25: RunName = 'StateN2_20_25'
if N2 == 25: RunName = 'StateN2_25'
if Modulated == 1:
    RunName = RunName + '_R'


#Set up grid and related objects:
Lx = 0.2
Lz = 0.45
factor = 1
Nx = 80
Nz = 180
Nx = Nx*factor
Nz = Nz*factor

x_basis = de.Fourier('x', Nx, interval=(0, Lx), dealias=3/2)
z_basis = de.SinCos('z', Nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
x=domain.grid(0)[:,0]
z=domain.grid(1)[0,:]

dx = x[1]-x[0]
dz = z[1]-z[0]
dt = .1

#make analysis during period of step existance:
dat = np.loadtxt('/home/ubuntu/BoussinesqLab/dedalus/stairStartEnd.txt')
N_vec = dat[0,:]
stairStart = dat[1,:]
stairEnd = dat[2,:]
Nidx = np.where(N_vec == np.sqrt(N2))
t0 = int( stairStart[int(Nidx[0])]/dt )
te = int( stairEnd[int(Nidx[0])]/dt )
print(t0*dt,te*dt)
#pdb.set_trace()

#if N2 == 2.25:
#    t0 = 100
#    te = 400
Nt = te-t0+1

t = np.arange(Nt)*dt + t0*dt

#Construct some general arrays for contour plots:
x2d = np.tile(x,(Nz,1))
z2d = np.tile(z,(Nx,1)).transpose()

t2d_z = np.tile(t,(Nz,1))
z2d_t = np.tile(z,(Nt,1)).transpose()


#Set-up wavenumbers for Dedalus grid:
kk = np.zeros(Nx)
kkx = x_basis.wavenumbers
kk[0:int(Nx/2.)] = kkx
dk = kkx[1]-kkx[0]
nyquist_f = -(np.max(kkx) + dk)
kk[int(Nx/2.)] = nyquist_f
kkx_neg = np.flipud(kkx[1:]*(-1))
kk[int(Nx/2.)+1:] = kkx_neg
kk_cosine = z_basis.wavenumbers

#Check wavenumbers:
#plt.plot(kk)
#plt.plot([0,Nx],[0,0])
#plt.show()
#print(kk)
#pdb.set_trace()
#plt.plot(kk_cosine)
#plt.show()

#Set physical constants and related objects:
g = 9.81
ct = 2*10**(-4.)
cs = 7.6*10**(-4.)

if ParkRun == 18:               #for lab run 18 of Park et al
    N2 = 3.83
    drho0_dz = -425.9
    rho0 = -g/N2*drho0_dz
if ParkRun == 14:               #for lab run 14 of Park et al
    N2 = 0.35
    drho0_dz = -31.976
    rho0 = -g/N2*drho0_dz
if ParkRun < 0:
    #Set ref density constant:
    N2_18 = 3.83
    drho0_dz_18 = -425.9
    rho0 = -g/N2_18*drho0_dz_18

    #Find initial background density field given some 
    #user defined N^2 and the constant ref density: 
    drho0_dz = -N2*rho0/g

if ProblemType == 'Layers':
    bs = -1./(rho0*cs)*drho0_dz
    if nvars == 3: bt = -bs*0.001
    if nvars == 2: bt = 0
if ProblemType == 'KelvinHelmholtz':
    bt = 0.
    bs = 0.

# Physical Diffusion:
nu = 1.*10**(-6.)
kappat = 1.4*10**(-7.)
kappas = 1.4*10**(-7.)
nu = nu * DiffusionFactor
kappat = kappat * DiffusionFactor
kappas = kappas * DiffusionFactor

#Define background salinity field
if FullFields == 1:
    drho0_dz13 = -122.09
    dgamma = 100./3
    dz_b = 2./100
    a0 = 100.
    z_a = Lz/2
    rhoprime13 = dgamma*z_a + a0*dz_b + dgamma/2*dz_b

    scalePert = 0
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

#Set time independent objects:
dir_sigma = './Results/' + RunName + '/NaturalBasis/'
dir_ivec = './Results/' + RunName + '/NaturalBasis/' 
#dir_sigma = './Results/' + RunName + '/NaturalBasis' + '_dt3/'
#dir_ivec = './Results/' + RunName + '/NaturalBasis' + '_dt3/'

State = np.zeros((Nx,Nz,Nt,nvars))

ivec_1 = np.zeros((Nx,Nz,nvars))
if nvars == 3: ivec0 = np.zeros((Nx,Nz,nvars))
ivec1 = np.zeros((Nx,Nz,nvars))

if BasisCheck2 == 1:
    #Define transforms to get back to State:
    fnm_kmag = dir_ivec + 'kmag_arr.txt'
    t1 = np.loadtxt(fnm_kmag)
    if nvars == 3: t2 = sqrt(-g*ct/bt)
    t3 = sqrt(g*cs/bs)

if sigma3D == 1:
    sigma_1_3D = np.zeros((Nx,Nz,Nt))*1j
    if nvars == 3: sigma0_3D = np.zeros((Nx,Nz,Nt))*1j
    sigma1_3D = np.zeros((Nx,Nz,Nt))*1j


#Set general plotting parameters assuming A4 page size:
A4Width = 8.27
MarginWidth = 1
width = A4Width-2*MarginWidth
height = 4
#For scaling the A4 plot dimensions:
ScaleFactor1 = 1
ScaleFactor2 = 1
width = width*ScaleFactor1
height = height*ScaleFactor2


def spectral_analysis(data,dt2,Welch=True):

    #Get dimensions of data (local variables):
    Nt = data.shape[0]
    if len(data.shape) > 1:
        Nx = data.shape[1]
        Nz = data.shape[2]
    else:
        Nx = 1
        Nz = 1

    signal_f = 1./dt2                           #Sampling frequency (needs to be units 1/s for Welch method)
    BigMode = np.zeros((Nx,Nz))
    f0 = 0.1                                    #Lowest frequency when finding dominant mode.

    if Welch == 0:
        #repeat non-periodic finite signal Np times:
        Np = 1
        signal_L = Nt*Np
        spectralCoef = np.zeros((int(signal_L/2.)+1,Nx,Nz))
        freqvec = np.arange(signal_L/2.+1)*1./signal_L          #assumes dt=1
                                                                #Note array length is Nt/2 and so max freq. will be Nyquist freq.
        freqvec         = freqvec*signal_f                      #uses actual signal frequency (Dedalus timestep)
        #freqvec        = freqvec*2*np.pi                       #converts to angular frequency (rad/s) - but less intuitive

    if Welch == 1:
        nwindows = 7
        if nwindows != 1: nperseg = int(2*Nt/(nwindows-1))
        else: nperseg = Nt
        print("(Nt, nperseg, nwindows): ", Nt,nperseg,nwindows)

        spectralCoef = np.zeros((int(nperseg/2.)+1,Nx,Nz))

    #Perform spectral analysis on time series at each grid point (possibly only one point):
    for jj in range(0,Nz):
        for ii in range(0,Nx):

            if len(data.shape) > 1: ts = data[:,ii,jj]
            else: ts = data

            if Welch == 0:
                #Try reflecting the signal:
                #ts_reflect = np.flip(ts,axis=0)
                #ts2 = np.concatenate((ts,ts_reflect))

                #Try applying a window function (a Tukey window here):  
                window = tukey(signal_L, alpha=0.05)
                ts2 = ts*window
                #plt.plot(ts2)
                #plt.show()

                ts_hat = np.fft.fft(ts2)
                psd = (1./(signal_f*signal_L))*abs(ts_hat)**2
                psd[1:int(signal_L/2.)+1] = 2*psd[1:int(signal_L/2.)+1]
                spectralCoef[:,ii,jj] = psd[0:int(signal_L/2.)+1]

                #Remove slow flows:
                fIdxVec = np.where(freqvec <= f0)
                fIdx = max(fIdxVec[0])
                #Find dominant frequencies:
                psd2 = psd[fIdx:int(signal_L/2.)+1]
                idx = np.where( psd2 == max(psd2) )
                BigMode[ii,jj] = freqvec[idx[0] + fIdx]

                nperseg = 0

            if Welch == 1:
                freqvec, psd = welch( ts,
                                      fs=signal_f,              # sampling rate
                                      window='hanning',         # apply a Hanning window before taking the DFT
                                      nperseg=nperseg,          # compute periodograms of nperseg-long segments of ts
                                      detrend=False)            # detrend ts by subtracting the mean

                spectralCoef[:,ii,jj] = psd

                #Remove slow flows when searching for dominant modes:
                fIdxVec = np.where(freqvec <= f0)
                fIdx = max(fIdxVec[0])
                ts_hat2 = psd[fIdx:int(nperseg/2.)+1]
                #Find dominant frequencies:
                idx = np.where( ts_hat2 == max(ts_hat2) )
                BigMode[ii,jj] = freqvec[idx[0] + fIdx]

    return spectralCoef, freqvec, nperseg;

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


if PlotCDFsVsN == 1:
    Nvec = np.array([1.,1.5,2,2.5,2.75,3,3.25,3.5,3.75,4,4.5,5])

    #for CapOmega limit 0
    #k=0 case:
    #N_IGW_k0_0 = np.array([2160.0,2180.0,2964.0,148.0,112.0,84.0,64.0,112.0,44.0,40.0,36.0,32.0])*2
    #N_MF_k0_0 = np.array([24.0,4.0,0.0,0.0,0.0,0.0,0.0,8.0,0.0,0.0,0.0,4.0])*2
    N_IGW_k0_0 = np.repeat(0,len(Nvec))
    N_MF_k0_0 = np.repeat(0,len(Nvec))
    #k not 0 case:
    N_IGW_knot0_0 = np.array([144.0,136.0,332.0,152.0,104.0,88.0,56.0,192.0,32.0,24.0,8.0,8.0])
    N_MF_knot0_0 = np.array([400.0,232.0,420.0,280.0,240.0,280.0,216.0,688.0,232.0,208.0,208.0,216.0])
    #N_IGW_knot0_0 = np.array([144.0,136.0,332.0,...,8.0])
    #N_MF_knot0_0 = np.array([400.0,232.0,420.0,...,216.0])

    #for CapOmega limit 0.001
    #k=0 case:
    #N_IGW_k0_001 = np.array([8763.2,6173.2,5663.3,148.0,112.0,84.0,64.0,112.0,44.0,40.0,36.0,32.0])*2
    #N_MF_k0_001 = np.array([3387.1,1669.0,966.3,607.5,498.0,404.4,325.2,267.1,220.9,194.0,0,4.0])*2
    N_IGW_k0_001 = np.array([6603.2,3993.2,2699.3,0,0,0,0,0,0,0,0,0])*2
    N_MF_k0_001 = np.array([3363.1,1665.0,966.3,607.5,498.0,404.4,325.2,259.1,220.9,194.0,0,0])*2
    #k not 0 case:
    N_IGW_knot0_001 = np.array([634.3,659.6,1578.6,807.1,597.0,594.5,416.9,1344.2,340.0,270.3,158.6,118.4])
    N_MF_knot0_001 = np.array([1807.6,591.9,755.8,402.4,315.5,396.4,289.0,976.9,313.6,269.9,261.4,265.9])
    #N_IGW_knot0_001 = np.array([634.3,...,118.4])
    #N_MF_knot0_001 = np.array([1807.6,...,265.9])

    #for CapOmega limit 0.01
    #k=0 case:
    #N_IGW_k0_01 = np.array([19288.2,13549.8,11415.9,1307.2,928.2,608.2,465.2,423.9,298.0,252.2,37.3,32.0])*2
    #N_MF_k0_01 = np.array([9340.2,5056.9,3183.5,2223.4,2001.2,1800.8,1641.5,1464.7,1301.4,1172.9,389.9,350.3])*2
    #N_IGW_k0_01 = np.array([17128.2,11369.8,8451.9,1159.2,816.2,524.2,401.2,311.9,254.0,212.2,1.3,0])*2
    #N_MF_k0_01 = np.array([9316.2,5052.9,3183.5,2223.4,2001.2,1800.8,1641.5,1456.7,1301.4,1172.9,389.9,346.3])*2
    N_IGW_k0_01 = np.array([17128.2,11369.8,8451.9,1159.2,816.2,524.2,401.2,311.9,254.0,212.2,1.3,0])*2
    N_MF_k0_01 = np.array([9316.2,5052.9,3183.5,2223.4,2001.2,1800.8,1641.5,1456.7,1301.4,1172.9,389.9,346.3])*2
    #k not 0 case:
    #N_IGW_knot0_01 = np.array([1049.0,1111.0,2584.4,1368.4,990.0,995.0,701.9,2263.3,579.9,459.5,281.5,193.4])
    #N_MF_knot0_01 = np.array([2933.0,864.6,1026.4,484.5,367.6,460.0,319.3,1196.4,352.3,300.5,288.0,284.9])
    #N_IGW_knot0_01 = np.array([1049.0,...,193.4])
    #N_MF_knot0_01 = np.array([2933.0,...,284.9])
    N_IGW_knot0_01 = np.array([0,0,0,0,0,1228.1,902.6,4453.8,922.5,835.1,748.8,705.8])
    N_MF_knot0_01 = np.array([0,0,0,0,0,459.8,319.0,1204.0,353.9,300.7,288.7,278.8])

    #find total counts for each N:
    #for CapOmega limit 0
    N_tot_k0_0 = np.add(N_IGW_k0_0,N_MF_k0_0)
    N_tot_knot0_0 = np.add(N_IGW_knot0_0,N_MF_knot0_0)
    N_tot_0 = np.add(N_tot_knot0_0,N_tot_k0_0)
    #for CapOmega limit 0.001
    N_tot_k0_001 = np.add(N_IGW_k0_001,N_MF_k0_001)
    N_tot_knot0_001 = np.add(N_IGW_knot0_001,N_MF_knot0_001)
    N_tot_001 = np.add(N_tot_knot0_001,N_tot_k0_001)
    #for CapOmega limit 0.01
    N_tot_k0_01 = np.add(N_IGW_k0_01,N_MF_k0_01)
    N_tot_knot0_01 = np.add(N_IGW_knot0_01,N_MF_knot0_01)
    N_tot_01 = np.add(N_tot_knot0_01,N_tot_k0_01)

    #find contributions from k=0 and k not 0 sets for mean flow and IGW bandwidths:
    #for CapOmega limit 0
    F_IGW_k0_0 = np.divide(N_IGW_k0_0,N_tot_0)
    F_MF_k0_0 = np.divide(N_MF_k0_0,N_tot_0)
    F_IGW_knot0_0 = np.divide(N_IGW_knot0_0,N_tot_0)
    F_MF_knot0_0 = np.divide(N_MF_knot0_0,N_tot_0)
    #for CapOmega limit 0.001
    F_IGW_k0_001 = np.divide(N_IGW_k0_001,N_tot_001)
    F_MF_k0_001 = np.divide(N_MF_k0_001,N_tot_001)
    F_IGW_knot0_001 = np.divide(N_IGW_knot0_001,N_tot_001)
    F_MF_knot0_001 = np.divide(N_MF_knot0_001,N_tot_001)
    #for CapOmega limit 0.01
    F_IGW_k0_01 = np.divide(N_IGW_k0_01,N_tot_01)
    F_MF_k0_01 = np.divide(N_MF_k0_01,N_tot_01)
    F_IGW_knot0_01 = np.divide(N_IGW_knot0_01,N_tot_01)
    F_MF_knot0_01 = np.divide(N_MF_knot0_01,N_tot_01)

    symsize_vec_k0 = (np.array([8,8,8,8,8,8,8,8,8]))**2
    symsize_vec_knot0 = (np.array([8,8,8,8,8,8,8,8,8]))**2

    #totals contributions from mean flow and IGW bandwidth:
    #for CapOmega limit 0
    F_IGW_0 = np.divide(np.add(N_IGW_k0_0,N_IGW_knot0_0),N_tot_0)
    F_MF_0 = 1 - F_IGW_0
    #for CapOmega limit 0.001
    F_IGW_001 = np.divide(np.add(N_IGW_k0_001,N_IGW_knot0_001),N_tot_001)
    F_MF_001 = 1 - F_IGW_001
    #for CapOmega limit 0.01
    F_IGW_01 = np.divide(np.add(N_IGW_k0_01,N_IGW_knot0_01),N_tot_01)
    F_MF_01 = 1 - F_IGW_01

    print(Nvec)
    print(F_IGW_01)
    print(F_MF_01)
    print( np.add(N_IGW_k0_01,N_MF_k0_01) )
    print( np.add(N_IGW_knot0_01,N_MF_knot0_01) )

    #check effect of varying near-resonance limit for N2=12.25 rad/s case:
    #limit_vec = [0.0001,0.001,0.01]
    #Nvec_limits = [3.5,3.5,3.5]
    #N_IGW_k0_N2_12_25 = np.array([112.0,112.0,423.9])*2
    #N_MF_k0_N2_12_25 = np.array([8.0,267.1,1464.7])*2
    #N_IGW_knot0_N2_12_25 = np.array([345.6,1344.2,2263.3])
    #N_MF_knot0_N2_12_25 = np.array([741.9,976.9,1196.4])
    #N_tot_k0_N2_12_25 = np.add(N_IGW_k0_N2_12_25, N_MF_k0_N2_12_25)
    #N_tot_knot0_N2_12_25 = np.add(N_IGW_knot0_N2_12_25, N_MF_knot0_N2_12_25)
    #N_tot_N2_12_25 = np.add(N_tot_knot0_N2_12_25,N_tot_k0_N2_12_25)
    #F_IGW_vec_N2_12_25 = np.divide(np.add(N_IGW_k0_N2_12_25,N_IGW_knot0_N2_12_25),N_tot_N2_12_25)
    #F_MF_vec_N2_12_25 = 1 - F_IGW_vec_N2_12_25
    #print(F_IGW_vec_N2_12_25)
    #print(F_MF_vec_N2_12_25)
    #-Yes the results were not the same so we then re-did all the calculations for different N at a different
    #resonant limit - thus removing the need for this section.

    #plot results:
    fig = plt.figure(1, figsize=(width,height))
    fig.set_tight_layout(True)
    grid = plt.GridSpec(1, 2, wspace=0.3, hspace=0.)
    ax1 = fig.add_subplot(grid[0,0])
    
    #plot IGW results:
    #for CapOmega limit 0
    ax1.plot(Nvec,F_IGW_0, linestyle='-', marker='.', markersize=14, color='k', linewidth=0.5, label=r'$|\Omega|=0$')
    #for CapOmega limit 0.001
    ax1.plot(Nvec,F_IGW_001, linestyle='-', marker='.', markersize=14, color='k', linewidth=2, label=r'$|\Omega|\leq 10^{-3}$')
    #for CapOmega limit 0.01
    ax1.plot(Nvec,F_IGW_01, linestyle='-', marker='.', markersize=14, color='k', linewidth=4, label=r'$|\Omega|\leq 10^{-2}$')
    #k=0 case:
    ax1.plot(Nvec, F_IGW_k0_01, linestyle='--', marker='.', markersize=7, color='silver', linewidth=2)
    #k not 0 case:
    ax1.plot(Nvec, F_IGW_knot0_01, linestyle=':', marker='.', markersize=7, color='silver', linewidth=2)

    ax1.plot([0,6],[0.5,0.5], linestyle='--',color='silver', linewidth=1)
    ax1.set_title(r'$F_{IGW}$')
    ax1.set_xlabel(r'$N$ (rad/s)')
    ax1.set_ylabel(r'Contribution to solution (%)')
    ax1.set_xlim(0,6)
    ax1.set_ylim(0,1)
    ax1.legend(frameon=False)

    #plot mean flow results:
    ax2 = fig.add_subplot(grid[0,1])
    #for CapOmega limit 0
    ax2.plot(Nvec,F_MF_0, linestyle='-', marker='.', markersize=14, color='k', linewidth=0.5)
    #for CapOmega limit 0.001
    ax2.plot(Nvec,F_MF_001, linestyle='-', marker='.', markersize=14, color='k', linewidth=2)
    #for CapOmega limit 0.01
    ax2.plot(Nvec,F_MF_01, linestyle='-', marker='.', markersize=14, color='k', linewidth=4)
    #k=0 case:
    ax2.plot(Nvec, F_MF_k0_01, linestyle='--', marker='.', markersize=7, color='silver', linewidth=2, label=r'$k=0$')
    #k not 0 case:
    ax2.plot(Nvec, F_MF_knot0_01, linestyle=':', marker='.', markersize=7, color='silver', linewidth=2,  label=r'$k\ne0$')

    ax2.plot([0,6],[0.5,0.5], linestyle='--',color='silver', linewidth=1)
    ax2.set_title(r'$F_{LFMF}$')
    ax2.set_xlabel(r'$N$ (rad/s)')
    ax2.set_ylabel(r'Contribution to solution (%)')
    ax2.set_xlim(0,6)
    ax2.set_ylim(0,1)
    ax2.legend(frameon=False, handlelength=3)

    OverplotPSDresults = 0
    if OverplotPSDresults == 1:
        #overplot PSD results
        meanflowarr = np.loadtxt('/home/ubuntu/BoussinesqLab/dedalus/meanflowarr.txt')
        psdIGWarr = np.loadtxt('/home/ubuntu/BoussinesqLab/dedalus/psdIGWarr.txt')

        F_IGW = psdIGWarr[:,3]/meanflowarr[:,0]
        F_IGW_max = np.max(F_IGW)
        F_IGW = F_IGW/F_IGW_max
        ax1.plot(N_vec,F_IGW, 'ok', fillstyle='none')

    plt.show()
    pdb.set_trace()



#Reconstruct State using the Natural Basis:
for tt in range(0,Nt):
            
    print(tt)
    tIdx = tt+t0

    fnm_sigma_1 = dir_sigma + 'sigma_1_' + str(tIdx) + '.txt'
    fnm_sigma1 = dir_sigma + 'sigma1_' + str(tIdx) + '.txt'
    sigma_1 = np.loadtxt(fnm_sigma_1).view(complex)
    sigma1 = np.loadtxt(fnm_sigma1).view(complex)
    if nvars == 3:
        fnm_sigma0 = dir_sigma + 'sigma0_' + str(tIdx) + '.txt'
        sigma0 = np.loadtxt(fnm_sigma0).view(complex)

    if sigma3D == 1:
        sigma_1_3D[:,:,tt] = sigma_1
        if nvars == 3: sigma0_3D[:,:,tt] = sigma0
        sigma1_3D[:,:,tt] = sigma1

    if tt == 0:
        for ll in range(0,nvars):
            #for nn in range(0,2):
            #    fnm_ivec_1 = dir_ivec + 'ivec_1_' + str(ll+1) + str(nn+1) + '.txt'
            #    fnm_ivec0 = dir_ivec + 'ivec0_' + str(ll+1) + str(nn+1) + '.txt'
            #    fnm_ivec1 = dir_ivec + 'ivec1_' + str(ll+1) + str(nn+1) + '.txt'
                #pdb.set_trace()
                #ivec_1[:,:,2*ll+nn] = np.loadtxt(fnm_ivec_1)
                #ivec0[:,:,2*ll+nn] = np.loadtxt(fnm_ivec0) 
                #ivec1[:,:,2*ll+nn] = np.loadtxt(fnm_ivec1)
            fnm_ivec_1 = dir_ivec + 'ivec_1_' + str(ll+1) + '.txt'
            fnm_ivec1 = dir_ivec + 'ivec1_' + str(ll+1) + '.txt'   
            ivec_1[:,:,ll] = np.loadtxt(fnm_ivec_1)
            ivec1[:,:,ll] = np.loadtxt(fnm_ivec1)
            if nvars == 3:
                fnm_ivec0 = dir_ivec + 'ivec0_' + str(ll+1) + '.txt'
                ivec0[:,:,ll] = np.loadtxt(fnm_ivec0) 

        PlotEigenvecs = 0
        if PlotEigenvecs == 1:
            plt.figure(figsize=(width,height))
     
            clevels = np.arange(21)*0.1-1
            print(clevels)
 
            plt.subplot(2, 2, 1)
            plt.imshow(ivec_1[:,:,0], cmap='RdBu')
            plt.yticks((0,40,80))
            plt.xticks((0,90,180))
            plt.colorbar()

            plt.subplot(2, 2, 2)
            plt.imshow(ivec1[:,:,0], cmap='RdBu')
            plt.yticks((0,40,80))
            plt.xticks((0,90,180))
            plt.colorbar()

            plt.subplot(2, 2, 3)
            plt.imshow(ivec_1[:,:,1], cmap='RdBu')
            plt.yticks((0,40,80))
            plt.xticks((0,90,180))
            plt.colorbar()

            plt.subplot(2, 2, 4)
            plt.imshow(ivec1[:,:,1], cmap='RdBu')
            plt.yticks((0,40,80))
            plt.xticks((0,90,180))
            plt.colorbar()

            plt.show()
            pdb.set_trace()

        if BasisCheck2 == 1:
            for jj in range(1,Nz):             
                for ii in range(0,Nx):
                   
                    if nvars == 3: 
                        ivec_1[ii,jj,0] = ivec_1[ii,jj,0]/t1[ii,jj]
                        ivec0[ii,jj,0] = ivec0[ii,jj,0]/t1[ii,jj]
                        ivec1[ii,jj,0] = ivec1[ii,jj,0]/t1[ii,jj]

                        ivec_1[ii,jj,1] =  ivec_1[ii,jj,1]/t2
                        ivec0[ii,jj,1] =  ivec0[ii,jj,1]/t2
                        ivec1[ii,jj,1] =  ivec1[ii,jj,1]/t2

                        ivec_1[ii,jj,2] =  ivec_1[ii,jj,2]/t3
                        ivec0[ii,jj,2] =  ivec0[ii,jj,2]/t3
                        ivec1[ii,jj,2] =  ivec1[ii,jj,2]/t3

                    if nvars == 2:
                        ivec_1[ii,jj,0] = ivec_1[ii,jj,0]/t1[ii,jj]
                        ivec1[ii,jj,0] = ivec1[ii,jj,0]/t1[ii,jj]

                        ivec_1[ii,jj,1] =  ivec_1[ii,jj,1]/t3
                        ivec1[ii,jj,1] =  ivec1[ii,jj,1]/t3

        #ivec_1 = ivec_1.view(complex)
        #ivec0 = ivec0.view(complex)
        #ivec1 = ivec1.view(complex)
        #pdb.set_trace()
    
    for vv in range(0,nvars):
        
        if nvars == 3:    
            fnc = np.multiply(sigma_1,ivec_1[:,:,vv]) + np.multiply(sigma0,ivec0[:,:,vv]) +\
		  np.multiply(sigma1,ivec1[:,:,vv])
        if nvars == 2:
            fnc = np.multiply(sigma_1,ivec_1[:,:,vv]) + np.multiply(sigma1,ivec1[:,:,vv])
      
        #Use Dedalus transforms and environment:
        tmp = domain.new_field()
        tmp.meta['z']['parity'] = -1
        tmp['c'] = fnc[0:int(Nx/2.),:]
        State[:,:,tt,vv] = tmp['g']


if BasisCheck1==1:
    #Choose data for comparisons:
    #Streamfunction:
    #data = State[:,:,:,0]
    #Salinity data:
    data = State[:,:,:,1]

    #The below section of code is for comparing the new salinity field
    #(after making the linear operator Skew Hermitian) computed by:
    #1) simply multiplying the original salinity field by a constant,
    #2) using the Natural basis.
    #Therefore this section checks the validity of the Natural basis code.
    #I use a timeseries at the central domain point for this comparison.
    #This script does not read in the original Dedalus output so I first 
    #created a text file of 1) above which I now read into this script:
    fnm = './StateS_2_N2_02_25_sp.txt'
    StateS_2_sp = np.loadtxt(fnm)
    #Find and print max difference:
    diff = StateS_2_sp.flatten() - data[int(Nx/2.),int(Nz/2.),:].flatten()
    print(np.max(abs(diff)))

    #For plotting fields and differences for the comparisons:
    if MakePlot == 1:

        if PlotXZ == 1 or PlotTZ == 1:
            PlotTitle = ''
            FigNmBase = 'psi'
            clevels = 50
            cmap = 'PRGn'
            tIdx = 10
            plt.contourf(data[:,:,tIdx].transpose(),clevels,cmap=cmap)
            plt.colorbar()

        if PlotT == 1:
            plt.plot(t,StateS_2_sp, 'ok')
            plt.plot(t,data[int(Nx/2.),int(Nz/2.),:].flatten(), '+r', markersize=15)
            plt.show()


#Analyse the Natural Basis of State:
if FindMainBasisParts == 1:

    if PlotMainBasisParts==1:
        fig = plt.figure(1, figsize=(width,height))
        grid = plt.GridSpec(1, 1, wspace=0., hspace=0.)
        ax1 = fig.add_subplot(grid[0,0])
        #ax1.set_ylim(-10,10)
        #ax1.set_ylim(10**(-10),100)
        ax1.set_ylabel('')
        ax1.set_xlabel(r'$t$ (s)')
        ax1.set_yscale('log') 

        #linewidthvec = [3,3,3,3,3,3,3,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
        #linecolorvec = ['k','b','c','g','y','m','r','k','b','c','g','y','m','r'] 
        #linecolorvec = ['black','silver','blue','cyan','lightblue','green','orange','gold','red','m'] 
        linecolorvec = ['black','silver','m','red','blue','cyan','lightblue','green','orange','gold']

    #Initialise objects required to find main basis parts by using 
    #a convergence method. The method compares power spectrums of State 
    #and a truncated natural basis for vertical structure of State.
    if ConvergePowerLimit == 1:
        if DedalusTransforms == 0:
            Welch = 0
            signal_f = 1./dz                                            	#Sampling frequency (needs to be units 1/s for Welch method)

            if Welch == 0:
                SpectralCoef0    	= np.zeros((int(Nz/2.)+1,Nt))
                SpectralCoef1    	= np.zeros((int(Nz/2.)+1,Nt))
                freqvec         	= np.arange(int(Nz/2.)+1)*(1./Nz)       #assumes dz=1
                         	                                           	#Note array length is Nz/2 and so max freq. will be Nyquist freq.
                freqvec         	= freqvec*signal_f                      #uses actual signal frequency (Aegir vertical resolution)
                #freqvec        	= freqvec*2*np.pi                       #converts to angular frequency (rad/m) - but less intuitive 

            if Welch == 1:
                nwindows = 7
                nperseg = int(2*Nt/(nwindows-1))
                print(Nt,nperseg,nwindows)
                SpectralCoef0 	= np.zeros((int(nperseg/2.)+1,Nt))
                SpectralCoef1 	= np.zeros((int(nperseg/2.)+1,Nt))
   
        if DedalusTransforms == 1:
            SpectralCoef0	= np.zeros((Nz,Nt))
            SpectralCoef1       = np.zeros((Nz,Nt))
 
        #PowerLimit  	= 0.1
        PowerLimit  	= 0.01
        ErrorLimit  	= 15
        epsilon     	= 100
        ts 		= 0

    if ConvergePowerLimit == 0: 
        PowerLimit 	= 0.01
        ErrorLimit 	= 15
        epsilon 	= 100
        ts 		= 0
    
    #Iterate until error (denoted epsilon) is smaller than chosen limit (a % error):
    count0 = 0
    while epsilon > ErrorLimit:

        #Dynamic lists to store main basis parts during search for given error limit: 
        ks1 = []
        ns1 = []
        ks2 = []
        ns2 = []
        ks3 = []
        ns3 = []

        #re-set counters at start of each iteration:
        count1 = 0
        count2 = 0
        count3 = 0

        #Find main parts of Natural basis given some PowerLimit:
        for jj in range(0,Nz):
            for ii in range(0,int(Nx/2)):

                lambdaX = ii
                lambdaZ = jj

                f1 = abs(sigma_1_3D[ii,jj,:])
                if nvars == 3: f2 = abs(sigma0_3D[ii,jj,:])
                f3 = abs(sigma1_3D[ii,jj,:])

                if sum(f1) > PowerLimit:
                    if PlotMainBasisParts==1 and ConvergePowerLimit==0 and PlotFastMode1==1:
                        label1 = 'sigma_1' + ', ' + '(' + str(lambdaX) + ', ' + str(lambdaZ) + ')'
                        #color1 = linecolorvec[count1]
                        data1 = np.abs(sigma_1_3D[ii,jj,:])
                        #data1 = np.real(sigma_1_3D[ii,jj,:])
                        ax1.plot(t, data1, linewidth=0.1+0.1*count1, c='k', linestyle='-', label=label1)

                    count1 = count1 + 1

                    ks1.append(lambdaX)
                    ns1.append(lambdaZ)

                if (nvars==3) and (sum(f2)>PowerLimit):
                    if PlotMainBasisParts==1 and ConvergePowerLimit==0:
                        label2 = 'sigma0' + ', ' + '(' + str(lambdaX) + ', ' + str(lambdaZ) + ')'
                        #color2 = linecolorvec[count2]
                        data2 = np.abs(sigma0_3D[ii,jj,:])
                        #data2 = np.real(sigma0_3D[ii,jj,:])
                        ax1.plot(t, data2, linewidth=2, linestyle='--', label=label2)

                    count2 = count2 + 1

                    ks2.append(lambdaX)
                    ns2.append(lambdaZ)

                FastMode2=1
                if (FastMode2==1) and (sum(f3)>PowerLimit):
                    #We need to consider these parts when converging to a PowerLimit, however, 
                    #there is no need to plot these sigma1 parts as they are just the negative of the sigma_1 ones:
                    if PlotMainBasisParts==1 and ConvergePowerLimit==0 and PlotFastMode2==1:
                        label3 = 'sigma1' + ', ' + '(' + str(lambdaX) + ', ' + str(lambdaZ) + ')'
                        #color3 = linecolorvec[count3]
                        data3 = np.abs(sigma1_3D[ii,jj,:])
                        #data3 = np.real(sigma1_3D[ii,jj,:])
                        ax1.plot(t, data3, linewidth=2, linestyle=':', label=label3)

                    count3 = count3 + 1

                    ks3.append(lambdaX)
                    ns3.append(lambdaZ)

        print('count1: ', count1)
        print('count2: ', count2)
        print('count3: ', count3)

        #Estimate State using a reduced Natural basis:
        Estimate 	= np.zeros((Nx,Nz,Nt,nvars))
        epsilon_vec 	= np.zeros(Nt)
        MainFreqIdx0 	= np.zeros(Nt, dtype=np.int8)
        MainFreqIdx1 	= np.zeros(Nt, dtype=np.int8)

        RemoveKeyModes = 0
        if RemoveKeyModes == 1:
            ks1 = np.array(ks1)
            ks3 = np.array(ks3)
            ns1 = np.array(ns1)
            ns3 = np.array(ns3)
            #idxs = np.where(ks1 != 0)
            idxs = np.where(ks1 == 0)
            idxs = np.array(idxs).flatten()
            ks1 = ks1[idxs]
            ns1 = ns1[idxs]
            #idxs = np.where(ks3 != 0)
            idxs = np.where(ks3 == 0)
            idxs = np.array(idxs).flatten()
            ks3 = ks3[idxs]
            ns3 = ns3[idxs]

        sigma_1_main		= np.zeros((Nx,Nz,Nt))*1j
        sigma0_main		= np.zeros((Nx,Nz,Nt))*1j
        sigma1_main		= np.zeros((Nx,Nz,Nt))*1j
        sigma_1_main[ks1,ns1,:]		= sigma_1_3D[ks1,ns1,:]
        if nvars == 3: 
            sigma0_main[ks2,ns2,:]	= sigma0_3D[ks2,ns2,:]
        sigma1_main[ks3,ns3,:]		= sigma1_3D[ks3,ns3,:]

        for tt in range(0, Nt):
            for vv in range(0,nvars):
                if nvars == 3:
                    fnc = np.multiply(sigma_1_main[:,:,tt],ivec_1[:,:,vv]) + \
                          np.multiply(sigma0_main[:,:,tt],ivec0[:,:,vv]) + \
                          np.multiply(sigma1_main[:,:,tt],ivec1[:,:,vv])
                if nvars == 2:
                    fnc = np.multiply(sigma_1_main[:,:,tt],ivec_1[:,:,vv]) + np.multiply(sigma1_main[:,:,tt],ivec1[:,:,vv])
 
                tmp = domain.new_field()
                tmp.meta['z']['parity'] = -1
                tmp['c'] = fnc[0:int(Nx/2.),:]
                Estimate[:,:,tt,vv] = tmp['g']

            if ConvergePowerLimit == 1: 

                if nvars == 3: varIdx = 2
                if nvars == 2: varIdx = 1

                if DedalusTransforms == 0:
                    f0 = np.mean(State[:,:,tt,varIdx], axis=0)
                    f1 = np.mean(Estimate[:,:,tt,varIdx], axis=0)
                    #f0 = State[int(Nx/2.),:,tt,varIdx]
                    #f1 = Estimate[int(Nx/2.),:,tt,varIdx]

                    if Welch == 0:
                        #This assumes signal is periodic:
                        f0_hat = np.fft.fft(f0)
                        f1_hat = np.fft.fft(f1)
                        psd0 = abs(f0_hat)**2
                        psd1 = abs(f1_hat)**2
                        psd0[1:int(Nz/2.)+1] = 2*psd0[1:int(Nz/.2)+1]
                        psd1[1:int(Nz/2.)+1] = 2*psd1[1:int(Nz/.2)+1]
                        SpectralCoef0[:,tt] = psd0[0:int(Nz/2.)+1]
                        SpectralCoef1[:,tt] = psd1[0:int(Nz/2.)+1]

                    if Welch == 1:
                        freqvec, psd0 = welch( f0,
                                   fs=signal_f,              # sampling rate
                                   window='hanning',         # apply a Hanning window before taking the DFT
                                   nperseg=nperseg,          # compute periodograms of nperseg-long segments of ts
                                   detrend=False)            # do not detrend ts by subtracting the mean
                        freqvec, psd1 = welch( f1,
                                   fs=signal_f,              # sampling rate
                                   window='hanning',         # apply a Hanning window before taking the DFT
                                   nperseg=nperseg,          # compute periodograms of nperseg-long segments of ts
                                   detrend=False)            # do not detrend ts by subtracting the mean

                        SpectralCoef0[:,tt] = psd0
                        SpectralCoef1[:,tt] = psd1
          
                if DedalusTransforms == 1:

                    #Compute Fourier-SinCos basis coefficients:    
                    tmp_state = domain.new_field()
                    tmp_estimate = domain.new_field()
                    tmp_state.meta['z']['parity'] = -1
                    tmp_estimate.meta['z']['parity'] = -1

                    tmp_state['g'] = State[:,:,tt,varIdx]
                    tmp_estimate['g'] = Estimate[:,:,tt,varIdx]
 
                    f0_hat = np.mean(tmp_state['c'], axis=0)
                    f1_hat = np.mean(tmp_estimate['c'], axis=0)

                    SpectralCoef0[:,tt] = abs(f0_hat)**2
                    SpectralCoef1[:,tt] = abs(f1_hat)**2
 
                MainFreqIdx0[tt] = int(np.where( SpectralCoef0[:,tt] == np.max(SpectralCoef0[:,tt]) )[0])
                MainFreqIdx1[tt] = int(np.where( SpectralCoef1[:,tt] == np.max(SpectralCoef1[:,tt]) )[0])

                print('%.3f' % MainFreqIdx0[tt], '%.3f' % MainFreqIdx1[tt])
                if MainFreqIdx0[tt] != MainFreqIdx1[tt]: print('Main frequencies are not the same!')

                epsilon_vec[tt] = (SpectralCoef0[MainFreqIdx0[tt],tt] - \
       				   SpectralCoef1[MainFreqIdx0[tt],tt])/SpectralCoef0[MainFreqIdx0[tt],tt]*100

        if ConvergePowerLimit == 1:        
            epsilon = np.max(abs(epsilon_vec))
            print('% error: ', epsilon)
            print('PowerLimit: ', PowerLimit)
            print(' ')
            if (epsilon > ErrorLimit): PowerLimit = PowerLimit - 0.05*PowerLimit

        if ConvergePowerLimit == 0: epsilon = ErrorLimit
        count0 += 1


    if PlotMainBasisParts==1:    
        plt.xlabel(r'$t$ (s)')
        #plt.ylim(10**(-4),10**(2))
        plt.legend()
        plt.show()

    if PlotState_MainBasis == 1:

        if nvars == 2: vIdx = 1
        if nvars == 3: vIdx = 2

        #Set arrays for contouring:
        if PlotTZ==1:
            data1 = State[int(Nx/2.),:,:,vIdx]
            data2 = Estimate[int(Nx/2.),:,:,vIdx]
            xgrid = t2d_z
            ygrid = z2d_t
            xlabel = r'$t$ (s)'
            ylabel = r'$z$ (m)'
            #xlim = (np.min(t),np.max(t))
            xlim = (0,60)
            ylim = (0,Lz)
        if PlotXZ==1:
            tIdx = int(Nt/2.)
            data1 = State[:,:,tIdx,vIdx].transpose()
            data2 = Estimate[:,:,tIdx,vIdx].transpose()
            xgrid = x2d
            ygrid = z2d
            xlabel = r'$x$ (m)'
            ylabel = r'$z$ (m)'
            xlim = (0,Lx)
            ylim = (0,Lz)

        fig1 = plt.figure(figsize=(width,height))
        grid1 = plt.GridSpec(1, 1, wspace=0.0, hspace=0.0)
        ax1 = fig1.add_subplot(grid1[0,0])
        if Modulated == 0: ax1.set_title(r'$S$ (g/kg)' )
        if Modulated == 1: ax1.set_title(r'$\zeta$ (g/kg)' )
        ax1.set_xlim(xlim)
        ax1.set_xlabel(xlabel)
        ax1.set_ylim(ylim)
        ax1.set_ylabel(ylabel)

        nlevs = 41
        SMin = -.02
        SMax = .02
        #SMin = np.min(State[int(Nx/2.),:,:,vIdx])
        #SMax = np.max(State[int(Nx/2.),:,:,vIdx])
        dS = (SMax-SMin)/(nlevs-1)
        clevels = np.arange(nlevs)*dS + SMin
        #cmap = 'PRGn'
        cmap = 'bwr'

        i1 = ax1.contourf(xgrid,ygrid,data1,clevels,cmap=cmap,extend='both')
        fig1.colorbar(i1)
        plt.show()

        fig2 = plt.figure(figsize=(width,height))
        grid2 = plt.GridSpec(1, 1, wspace=0.0, hspace=0.0)
        ax2 = fig2.add_subplot(grid2[0,0])
        if Modulated == 0: ax2.set_title(r'Estimate of $S$ (g/kg)')
        if Modulated == 1: ax2.set_title(r'Estimate of $\zeta$ (g/kg)')
                         #'{:2d}'.format(count1) + ', ' + '{:2d}'.format(count3) +\
                         #', ' + '{:5.2f}'.format(PowerLimit) + ', ' + '{:5.2f}'.format(epsilon) + '%' )
        ax2.set_xlim(xlim)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.set_ylim(ylim)
        i2 = ax2.contourf(xgrid,ygrid,data2,clevels,cmap=cmap,extend='both')
        fig2.colorbar(i2)

        #Annotate plot with key modes:
        #modes have been sorted into k=0 and k not 0 cases and then by vertical wavenumber.
        idxs0 = np.where(np.array(ks3) == 0)[0].flatten()
        idxsNot0 = np.nonzero(np.array(ks3))[0].flatten()

        tmp = np.array(ns3).flatten()
        idxs0sorted = np.argsort(tmp[idxs0])
        idxsNot0sorted = np.argsort(tmp[idxsNot0])
        tmp0sorted = np.sort(tmp[idxs0])
        tmpNot0sorted = np.sort(tmp[idxsNot0])
        ns3_arr = np.concatenate((tmp0sorted,tmpNot0sorted))
        
        tmp0 = np.array(ks3)[idxs0]
        tmpNot0 = np.array(ks3)[idxsNot0]
        tmp0sorted = tmp0[idxs0sorted]
        tmpNot0sorted = tmpNot0[idxsNot0sorted]
        ks3_arr = np.concatenate((tmp0sorted,tmpNot0sorted))

        tmp1 = list(zip(ks1, ns1))
        labels1=[]
        labels1.append(r'$\alpha=-1$')
        for i in tmp1: labels1.append(str(i))
        tmp2 = list(zip(ks3_arr, ns3_arr))
        labels2=[]
        labels2.append(r'$\alpha=+1$')
        for i in tmp2: labels2.append(str(i))
 
        dz_txt = 0.02
        for i in range(0,len(labels1)):
            if i == 0: 
                ha='center'
                dx_txt = 0
            else: 
                ha='left'
                dx_txt = 3.
            if i <= 20:
                x = 0.4*max(xlim)-dx_txt
                y = 0.425-dz_txt*i
            elif i <= 40:
                x = 0.4*max(xlim) + 4
                y = 0.425-dz_txt*(i-20)
            else:
                x = 0.4*max(xlim) + 11
                y = 0.425-dz_txt*(i-40)
            ax2.annotate(labels1[i], xy=(x,y), xytext=(x,y), ha=ha, size=8) 
        for i in range(0,len(labels2)): 
            if i == 0: 
                ha='center'
                dx_txt = 0
            else: 
                ha='left'
                dx_txt = 3.
            if i <= 20:
                x = 0.7*max(xlim)-dx_txt
                y = 0.425-dz_txt*i
            elif i <= 40:
                x = 0.7*max(xlim) + 4
                y = 0.425-dz_txt*(i-20)
            else:
                x = 0.7*max(xlim) + 11
                y = 0.425-dz_txt*(i-40)
            ax2.annotate(labels2[i], xy=(x,y), xytext=(x,y), ha=ha, size=8) 

        plt.show() 
        #pdb.set_trace()
 
    if PlotSpectrum == 1:
        tIdx = int(Nt/2.)
        plt.semilogy(kk_cosine, SpectralCoef0[:,tIdx], 'k-')
        plt.semilogy(kk_cosine, SpectralCoef1[:,tIdx], 'b-')
        plt.xlabel(r'$n$ (rad/m)')
        plt.ylabel(r'$|\langle{S}\rangle|$')
        plt.show()
    
    if PlotSigmaSums == 1:
        fig=plt.figure(figsize=(width,height))
        grid = plt.GridSpec(1, 1, wspace=0, hspace=0.0)
        ax1 = fig.add_subplot(grid[0,0])
        Nmodes = len(ks1)+len(ks3)
        sumvec = np.zeros((Nmodes))
        for i in range(0,Nmodes):
            if i < len(ks1): sumvec[i] = np.sum(abs(sigma_1_3D[ks1[i],ns1[i],:]))
            else: sumvec[i] = np.sum(abs(sigma1_3D[ks3[i-len(ks1)],ns3[i-len(ks1)],:]))
        sumvec = sumvec / np.max(sumvec) * 10
        sumvec = sumvec**2
        sumvec = np.pi/4*sumvec
        ax1.scatter( kk[ks1]/(2*np.pi), kk_cosine[ns1]/np.pi, s=sumvec[0:len(ks1)], c='k', marker='o')
        ax1.scatter( kk[ks3]/(2*np.pi), kk_cosine[ns3]/np.pi, s=sumvec[len(ks1):], c='grey', marker='o')
        #ax1.set_xlim(-5,5)
        #ax1.set_ylim(0,20)
        ax1.set_xlabel(r'$k$')
        ax1.set_ylabel(r'$n$')
        ax1.grid(True)
        plt.show()
        

if AnalyseMainBasisParts == 1:

    Nk_1 = len(ks1)
    Nk1 = len(ks3)

    if SpectralAnalysis == 1:
        Welch = 0
        signal_f = 1./dt	#Sampling frequency (needs to be units 1/s for Welch method)
 
        if Welch == 0:
            SpectralCoef0       = np.zeros((int(Nt/2.)+1,Nk_1))
            SpectralCoef1       = np.zeros((int(Nt/2.)+1,Nk1))
            freqvec             = np.arange(int(Nt/2.)+1)*(1./Nt)       #assumes dt=1
                                                                        #Note array length is Nt/2 and so max freq. will be Nyquist freq.
            freqvec             = freqvec*signal_f                      #uses actual signal frequency (Aegir timestep resolution)
            #freqvec            = freqvec*2*np.pi                       #converts to angular frequency (rad/s) - but less intuitive 

        if Welch == 1:
            dnmntr              = 2.
            nperseg             = int(Nt/dnmntr)
            SpectralCoef0       = np.zeros((int(nperseg/2.)+1,Nk_1))
            SpectralCoef1       = np.zeros((int(nperseg/2.)+1,Nk1))

    for i in range(0,Nk_1):

        if SpectralAnalysis == 1:
            f0 = sigma_1_3D[ks1[i],ns1[i],:]

            if Welch == 0:
                #This assumes signal is periodic:
                f0_hat = np.fft.fft(f0)
                psd0 = abs(f0_hat)**2
                psd0[1:int(Nt/2.)+1] = 2*psd0[1:int(Nt/2.)+1]
                SpectralCoef0[:,i] = psd0[0:int(Nt/2.)+1]

            if Welch == 1:
                freqvec, psd0 = welch( f0,
                                       fs=signal_f,              # sampling rate
                                       window='hanning',         # apply a Hanning window before taking the DFT
                                       nperseg=nperseg,          # compute periodograms of nperseg-long segments of ts
                                       detrend=False)            # do not detrend ts by subtracting the mean
                SpectralCoef0[:,i] = psd0

        if DispersionRelation == 1:
            kmag = sqrt(kk[ks1[i]]**2 + kk_cosine[ns1[i]]**2)
            omega_k = abs(kk[ks1[i]])/kmag*sqrt(-g*(ct*bt-cs*bs)) 
            lambda_z = 1/(kk_cosine[ns1[i]]/np.pi)
            print(lambda_z, omega_k/(2*np.pi), sqrt(N2))

    for i in range(0,Nk1):

        if SpectralAnalysis == 1:
            f1 = sigma1_3D[ks3[i],ns3[i],:]

            if Welch == 0:
                #This assumes signal is periodic:
                f1_hat = np.fft.fft(f1)
                psd1 = abs(f1_hat)**2
                psd1[1:int(Nt/2.)+1] = 2*psd1[1:int(Nt/2.)+1]
                SpectralCoef1[:,i] = psd1[0:int(Nt/2.)+1]

            if Welch == 1:
                freqvec, psd1 = welch( f1,
                                       fs=signal_f,              # sampling rate
                                       window='hanning',         # apply a Hanning window before taking the DFT
                                       nperseg=nperseg,          # compute periodograms of nperseg-long segments of ts
                                       detrend=False)            # do not detrend ts by subtracting the mean
                SpectralCoef1[:,i] = psd1

        if DispersionRelation == 1:
            kmag = sqrt(kk[ks3[i]]**2 + kk_cosine[ns3[i]]**2)
            omega_k = abs(kk[ks3[i]])/kmag*sqrt(-g*(ct*bt-cs*bs))     
            lambda_z = 1/(kk_cosine[ns3[i]]/np.pi)
            print(lambda_z, omega_k/(2*np.pi), sqrt(N2))

    if SpectralAnalysis == 1:
        plt.semilogy(freqvec,SpectralCoef0[:,0])
        plt.semilogy(freqvec,SpectralCoef0[:,1])
        plt.semilogy(freqvec,SpectralCoef0[:,2])
        plt.semilogy(freqvec,SpectralCoef0[:,3])
        plt.semilogy(freqvec,SpectralCoef0[:,4])
        plt.semilogy(freqvec,SpectralCoef0[:,5])
        plt.semilogy(freqvec,SpectralCoef0[:,6])
        plt.semilogy(freqvec,SpectralCoef0[:,7])
        #plt.semilogy(freqvec,SpectralCoef0[:,8])
        #plt.semilogy(freqvec,SpectralCoef0[:,9])
        plt.show()




if Resonances == 1:
    
    #From equation (...) in CodeEquations.pdf.
    print(" ")
    print("Find wavenumber space for nonlinear term")

    #Note that it was necessary to round numbers before applying
    #relational operators to avoid large rounding errors.
    triad_x_mask 	= np.zeros((Nx,Nx,Nx), dtype=bool)
    Ntriads_x		= np.zeros((Nx))
    for i in range(0,Nx):
        for i1 in range(0,Nx):
            for i2 in range(0,Nx):
                k1 = kk[i1]
                k2 = kk[i2]
                ksum = k1 + k2
                logical = round(ksum,5) == round(kk[i],5)
                triad_x_mask[i,i1,i2] = logical
        Ntriads_x[i] = sum(triad_x_mask[i,:,:])
    Ntriads_x = Ntriads_x.astype(int)

    #Plot wavenumber space for x-direction:
    #plt.plot(np.sort(kk),Ntriads_x)
    #print(kk)
    #plt.contourf(sum(triad_x_mask,0), 1, colors=['white','black'])
    #plt.contourf(triad_x_mask[1,:,:], 1, colors=['white','black'])
    #plt.colorbar()
    #plt.grid(True)
    #plt.show()
    #pdb.set_trace()

    #This is how to get all points at once (but doesn't give you 
    #points for each k):
    #k1,k2 = np.meshgrid(kk,kk)
    #k1plusk2 = k1 + k2
    #mask_x = np.isin(k1plusk2, kk)     
    #idxs = np.where(mask_x==True)
    #np.array(idxs).shape 

    triad_z_mask        = np.zeros((Nz,Nz,Nz), dtype=bool)
    Ntriads_z           = np.zeros((Nz))
    for j in range(1,Nz):
        for j1 in range(1,Nz):
            for j2 in range(1,Nz):
                n1 = kk_cosine[j1]
                n2 = kk_cosine[j2]
                nsum1 = n1 + n2
                nsum2 = n1 - n2
                nsum3 = n2 - n1
                logical = round(nsum1,5) == round(kk_cosine[j],5) or\
                          round(nsum2,5) == round(kk_cosine[j],5) or\
                          round(nsum3,5) == round(kk_cosine[j],5)
                triad_z_mask[j,j1,j2] = logical
        Ntriads_z[j] = sum(triad_z_mask[j,:,:])
    Ntriads_z = Ntriads_z.astype(int)

    #Plot wavenumber space for z-direction:
    #print(kk_cosine)
    #plt.plot(np.sort(kk_cosine),Ntriads_z)
    #plt.imshow(sum(triad_z_mask,0), origin='lower', cmap='binary')
    #plt.contourf(kk_cosine,kk_cosine,triad_z_mask[0,:,:], 1, colors=['white','black'])
    #plt.contourf(triad_z_mask[1,:,:], 1, colors=['white','black'])
    #plt.colorbar()
    #plt.grid(True)
    #plt.show()
    #pdb.set_trace()

    Npnts = np.zeros((Nx,Nz))
    for j in range(0,Nz):
        for i in range(0,Nx):
            idxs_triads_x = np.array(np.where(triad_x_mask[i,:,:] == True))
            idxs_triads_z = np.array(np.where(triad_z_mask[j,:,:] == True))
            count_x = len(idxs_triads_x[0,:])
            count_z = len(idxs_triads_z[0,:]) 
            Npnts[i,j] = count_x*count_z
    print("total number of non-zero points: ", int(np.sum(Npnts)))

    #OmegaRange = np.max(OmegaArr)
    #OmegaRange = Nx*Nz*2
    #nlevs = 100
    #levels = np.arange(nlevs)*(OmegaRange/(nlevs-1))
    #cmap = plt.get_cmap('Greys')
    #norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    #plt.pcolormesh(Npnts, cmap=cmap, norm=norm)
    #plt.colorbar()
    #plt.show()
    #pdb.set_trace()


    if LinearNonLinear == 1:

        print("Check form of non-linear term")
        #Choose time points:
        nt = Nt
        #nt = 30

        #Fast wave averaging parameters:
        OmegaLimit = 1.
        tau = 1.
        nt2 = (tau-0)/dt
        tvec = np.arange(nt2)*dt/OmegaLimit

        #Initialise array to store result:
        RHS = np.zeros((Nx,Nz,nt))*1j

        si = ks1[0]
        ei = ks1[0]+1
        sj = ns1[0]
        ej = ns1[0]+1
        #print(ks1[0],ns1[0])
        #si = 0
        #ei = 1
        #sj = 1
        #ej = 2

        Integrate = 0
        ref = 0

        for tt in range(1,nt):
 
            print(" ")
            print(tt)

            #subset of k,n points:        
            for j in range(sj,ej):
                for i in range(si,ei):

                    k = kk[i]
                    n = kk_cosine[j]
                    kmag = sqrt(k**2 + n**2)
                    omega_k = abs(k)/kmag*sqrt(-g*(ct*bt-cs*bs))

                    C_linear = -sqrt(g*cs*bs)*1j*k/kmag
                    #C_linear = -bs*1j*k
                    if Integrate == 0:
                        linear 	= C_linear*(\
                            	  sigma_1_3D[i,j,tt]*ivec_1[i,j,0] +\
			   	  sigma1_3D[i,j,tt]*ivec1[i,j,0] )
                        diffusion  = -kappas*kmag**2*(\
                            	     sigma_1_3D[i,j,tt]*ivec_1[i,j,1] +\
                                     sigma1_3D[i,j,tt]*ivec1[i,j,1] )
                    else:
                        linear = C_linear*(\
                                 ivec_1[i,j,0]*np.trapz(sigma_1_3D[i,j,ref:tt],dx=dt) +\
                                 ivec1[i,j,0]*np.trapz(sigma1_3D[i,j,ref:tt],dx=dt) )
                                 #ivec_1[i,j,0]*simps(sigma_1_3D[i,j,ref:tt],dx=dt,even='avg') +\
                                 #ivec1[i,j,0]*simps(sigma1_3D[i,j,ref:tt],dx=dt,even='avg') )
                        diffusion = -kappas*kmag**2*(\
                                    ivec_1[i,j,1]*np.trapz(sigma_1_3D[i,j,ref:tt],dx=dt) +\
                                    ivec1[i,j,1]*np.trapz(sigma1_3D[i,j,ref:tt],dx=dt) )
                                    #ivec_1[i,j,1]*simps(sigma_1_3D[i,j,ref:tt],dx=dt,even='avg') +\
                                    #ivec1[i,j,1]*simps(sigma1_3D[i,j,ref:tt],dx=dt,even='avg') )
                    #Initialise non-linear term:
                    nonLinear = 0
                    tmp = 0

                    #Only loop over required points for some k,n:
                    idxs_triads_x = np.array(np.where(triad_x_mask[i,:,:] == True))
                    idxs_triads_z = np.array(np.where(triad_z_mask[j,:,:] == True))

                    count0 = 0
                    for count_i in range(0,len(idxs_triads_x[0,:])):
                        for count_j in range(0,len(idxs_triads_z[0,:])):

                            #print(count0)

                            i1 = idxs_triads_x[0,count_i]
                            i2 = idxs_triads_x[1,count_i]
                            j1 = idxs_triads_z[0,count_j]
                            j2 = idxs_triads_z[1,count_j]

                            k1 = kk[i1]
                            k2 = kk[i2]
                            n1 = kk_cosine[j1]
                            n2 = kk_cosine[j2]
                        
                            kmag1 = sqrt(k1**2 + n1**2)
                            kmag2 = sqrt(k2**2 + n2**2)
                            omega_k1 = abs(k1)/kmag1*sqrt(-g*(ct*bt-cs*bs))
                            omega_k2 = abs(k2)/kmag2*sqrt(-g*(ct*bt-cs*bs))
                            C_nonlinear = 1./kmag1
                            #C_nonlinear = 1

                            if Integrate == 0: 
                                tmp +=\
			            (sigma_1_3D[i1,j1,tt]*ivec_1[i1,j1,0] + sigma1_3D[i1,j1,tt]*ivec1[i1,j1,0])*\
                                    (sigma_1_3D[i2,j2,tt]*ivec_1[i2,j2,1] + sigma1_3D[i2,j2,tt]*ivec1[i2,j2,1])*\
			            (1j*k1*n2-1j*k2*n1)*C_nonlinear
                            else:
                                interact1 = ivec_1[i1,j1,0]*ivec_1[i2,j2,1]*sigma_1_3D[i1,j1,ref:tt]*sigma_1_3D[i2,j2,ref:tt]
                                interact2 = ivec_1[i1,j1,0]*ivec1[i2,j2,1]*sigma_1_3D[i1,j1,ref:tt]*sigma1_3D[i2,j2,ref:tt]
                                interact3 = ivec1[i1,j1,0]*ivec_1[i2,j2,1]*sigma1_3D[i1,j1,ref:tt]*sigma_1_3D[i2,j2,ref:tt]
                                interact4 = ivec1[i1,j1,0]*ivec1[i2,j2,1]*sigma1_3D[i1,j1,ref:tt]*sigma1_3D[i2,j2,ref:tt]
                                tmp += (\
                                    np.trapz(interact1,dx=dt) +\
                                    np.trapz(interact2,dx=dt) +\
                                    np.trapz(interact3,dx=dt) +\
                                    np.trapz(interact4,dx=dt) )*\
                                    (1j*k1*n2-1j*k2*n1)*C_nonlinear
				    #simps(interact1,dx=dt,even='avg') +\
				    #simps(interact2,dx=dt,even='avg') +\
				    #simps(interact3,dx=dt,even='avg') +\
			   	    #simps(interact4,dx=dt,even='avg')*\
                                    #(1j*k1*n2-1j*k2*n1)/kmag1

                            #(ivec_1[i2,j2,0]*ivec_1[i1,j1,1]/kmag2/sqrt(g*cs*bs) - ivec_1[i1,j1,0]*ivec_1[i2,j2,1]/kmag1/sqrt(g*cs*bs)) +\
                            #1./tau*sum(exp(-1j*(-omega_k1-omega_k2+omega_k)*tvec)) +\
                            #exp(-1j*(-omega_k1-omega_k2+omega_k)*t[tt]) +\
			    #alpha1 = -1, alpha2 = +1:
			    #sigma_1_3D[i1,j1,tt]*sigma1_3D[i2,j2,tt]*\
                            #(ivec1[i2,j2,0]*ivec_1[i1,j1,1]/kmag2/sqrt(g*cs*bs) - ivec_1[i1,j1,0]*ivec1[i2,j2,1]/kmag1/sqrt(g*cs*bs)) +\
                            #1./tau*sum(exp(-1j*(-omega_k1+omega_k2+omega_k)*tvec)) +\
                            #exp(-1j*(-omega_k1+omega_k2+omega_k)*t[tt]) +\
			    #alpha1 = +1, alpha2 = -1:
                            #sigma1_3D[i1,j1,tt]*sigma_1_3D[i2,j2,tt]*\
                            #(ivec_1[i2,j2,0]*ivec1[i1,j1,1]/kmag2/sqrt(g*cs*bs) - ivec1[i1,j1,0]*ivec_1[i2,j2,1]/kmag1/sqrt(g*cs*bs)) +\
                            #1./tau*sum(exp(-1j*(omega_k1-omega_k2+omega_k)*tvec)) +\
                            #exp(-1j*(omega_k1-omega_k2+omega_k)*t[tt]) +\
 			    #alpha1 = +1, alpha2 = +1:
			    #sigma1_3D[i1,j1,tt]*sigma1_3D[i2,j2,tt]*\
                            #(ivec1[i2,j2,0]*ivec1[i1,j1,1]/kmag2/sqrt(g*cs*bs) - ivec1[i1,j1,0]*ivec1[i2,j2,1]/kmag1/sqrt(g*cs*bs)) \
                            #1./tau*sum(exp(-1j*(omega_k1+omega_k2+omega_k)*tvec)) +\
                            #exp(-1j*(omega_k1+omega_k2+omega_k)*t[tt]) +\

                            #alpha = +1:
                            #alpha1 = -1, alpha2 = -1:
                            #sigma_1_3D[i1,j1,tt]*sigma_1_3D[i2,j2,tt]*\
                            #(ivec_1[i2,j2,0]*ivec_1[i1,j1,1]-ivec_1[i1,j1,0]*ivec_1[i2,j2,1]) +\
                            #1./tau*sum(exp(-1j*(-omega_k1-omega_k2-omega_k)*tvec)) +\
                            #exp(-1j*(-omega_k1-omega_k2-omega_k)*t[tt]) +\
                            #alpha1 = -1, alpha2 = +1:
                            #sigma_1_3D[i1,j1,tt]*sigma1_3D[i2,j2,tt]*\
                            #(ivec1[i2,j2,0]*ivec_1[i1,j1,1]-ivec_1[i1,j1,0]*ivec1[i2,j2,1]) +\
                            #exp(-1j*(-omega_k1+omega_k2-omega_k)*t[tt]) +\
                            #alpha1 = +1, alpha2 = -1:
                            #sigma1_3D[i1,j1,tt]*sigma_1_3D[i2,j2,tt]*\
                            #(ivec_1[i2,j2,0]*ivec1[i1,j1,1]-ivec1[i1,j1,0]*ivec_1[i2,j2,1]) +\
                            #1./tau*sum(exp(-1j*(omega_k1-omega_k2-omega_k)*tvec)) +\
                            #exp(-1j*(omega_k1-omega_k2-omega_k)*t[tt]) +\
                            #alpha1 = +1, alpha2 = +1:
                            #sigma1_3D[i1,j1,tt]*sigma1_3D[i2,j2,tt]*\
                            #(ivec1[i2,j2,0]*ivec1[i1,j1,1]-ivec1[i1,j1,0]*ivec1[i2,j2,1])\
                            #1./tau*sum(exp(-1j*(omega_k1+omega_k2-omega_k)*tvec))
                            #exp(-1j*(omega_k1+omega_k2-omega_k)*t[tt])
			    #) 

                                #count0 += 1       
                            count0 += 1       
                    
                    nonLinear = tmp 
                    RHS[i,j,tt] = linear + nonLinear + diffusion


        #check values:
        LHS = np.zeros((Nx,Nz,nt))*1j

        if Integrate == 0:
            scheme = 'centred'
            #scheme = 'forward'
            if scheme == 'centred':
                t_start = 1
                end_offset = 1
            if scheme == 'forward':
                t_start = 0
                end_offset = 1
            sigma3D_arr = np.zeros((Nx,Nz,Nt))*1j
        else:
            t_start = 1
            end_offset = 0

        for j in range(sj,ej):
            for i in range(si,ei):

                if Integrate == 0:
                    #sum over alpha before computing derivative:
                    sigma3D_arr[i,j,:] = sigma_1_3D[i,j,:]*ivec_1[i,j,1] + sigma1_3D[i,j,:]*ivec1[i,j,1]

                for tt in range(t_start,nt-end_offset):

                    if Integrate == 0:
                        if scheme == 'centred':
                            LHS[i,j,tt] = (sigma3D_arr[i,j,tt+1]-sigma3D_arr[i,j,tt-1])/(2*dt)
                        if scheme == 'forward':
                            LHS[i,j,tt] = (sigma3D_arr[i,j,tt+1]-sigma3D_arr[i,j,tt])/dt
                    else:
                        LHS[i,j,tt] = ivec_1[i,j,1]*(sigma_1_3D[i,j,tt] - sigma_1_3D[i,j,ref]) +\
                                      ivec1[i,j,1]*(sigma1_3D[i,j,tt] - sigma1_3D[i,j,ref])

        #print(sigma_1_3D[0,1,:])
        #print(sigma1_3D[0,1,:])
        #print(ivec_1[0,1,1])
        #print(ivec1[0,1,1])
 
        #plt.figure()
        plt.plot(t,RHS[si,sj,:], '-+b', label='RHS')
        #plt.figure()
        plt.plot(t,LHS[si,sj,:], '-ok', label='LHS')
        #plt.ylim((-1E-8,1E-8))
        plt.xlabel(r'$t$ (s)')
        plt.legend()
        plt.grid(True)
        plt.show()


    if NearResonanceSearch == 1:
        OmegaLimit = 0.
        #OmegaLimit = 0.0001
        #OmegaLimit = 0.001
        OmegaLimit = 0.01
        #OmegaLimit = 0.1
        #OmegaLimit = 1
        #OmegaLimit = 999
        Omega = []
        alphavec = [-1,1]

        #remove k=0 or k!=0 modes:
        ks = np.asarray(ks1+ks3)	#convert list to array
        if keyWaveModes == 1 and keyNonWaveModes == 0:
            idxs = np.where(ks != 0) 	#search returns a tuple!
        if keyNonWaveModes == 1 and keyWaveModes == 0:
            idxs = np.where(ks == 0) 	#search returns a tuple!
        if keyWaveModes == 1 and keyNonWaveModes == 1: idxs = np.where(ks != 999999)
        idxs = np.array(idxs).flatten()	#convert to array and flatten

        Nk = len(idxs)
        print("total # of key (wave or/and non-wave) modes: ", Nk)

        ks1 = np.array(ks1)
        ks3 = np.array(ks3)
        ns1 = np.array(ns1)
        ns3 = np.array(ns3)
        if keyWaveModes == 1 and keyNonWaveModes == 0:
            idxs = np.where(ks1 != 0)
        if keyNonWaveModes == 1 and keyWaveModes == 0:
            idxs = np.where(ks1 == 0)
        if keyWaveModes == 1 and keyNonWaveModes == 1: idxs = np.where(ks1 != 999999)
        idxs = np.array(idxs).flatten()
        ks1 = ks1[idxs]
        ns1 = ns1[idxs]

        if keyWaveModes == 1 and keyNonWaveModes == 0:
            idxs = np.where(ks3 != 0)
        if keyNonWaveModes == 1 and keyWaveModes == 0:
            idxs = np.where(ks3 == 0)
        if keyWaveModes == 1 and keyNonWaveModes == 1: idxs = np.where(ks3 != 999999)
        idxs = np.array(idxs).flatten()
        ks3 = ks3[idxs]
        ns3 = ns3[idxs]

        #reverse vectors to reveal symmetry from alphas when looking at key wave modes:
        if keyWaveModes == 1 and keyNonWaveModes == 0:
            ks3 = np.flip(ks3, axis=0)
            ns3 = np.flip(ns3, axis=0)

        #Initialise arrays for resonance results:
        OmegaArr = np.zeros((Nk,Nx,Nz*2,len(alphavec),len(alphavec)))
        OmegaBool = np.zeros((Nk,Nx,Nz*2,len(alphavec),len(alphavec)),dtype=bool)

        #Define, k, n and alpha based on main basis elements found above:
        for v in range(0,Nk):
           
            count0 = 0

            if v < len(ks1):
                i = ks1[v]
                j = ns1[v]
                alpha = 0
            else:
                i = ks3[v-len(ks1)]
                j = ns3[v-len(ks1)]
                alpha = 1

            k = kk[i]
            n = kk_cosine[j] 
            kmag = sqrt(k**2 + n**2)
            omega_k = alphavec[alpha]*abs(k)/kmag*sqrt(-g*(ct*bt-cs*bs))

            #Only loop over required points for some k,n:
            idxs_triads_x = np.array(np.where(triad_x_mask[i,:,:] == True))
            idxs_triads_z = np.array(np.where(triad_z_mask[j,:,:] == True))
     
            #if Nk != 0:
            #    print(idxs_triads_x)
            #    print(idxs_triads_z)
            #    print(kk)
            #    pdb.set_trace() 
            #plt.figure() 
            #plt.scatter( kk[idxs_triads_x[0,:]], kk[idxs_triads_x[1,:]] )
            #plt.scatter( kk_cosine[idxs_triads_z[0,:]], kk_cosine[idxs_triads_z[1,:]] )
            #plt.show()
            #pdb.set_trace()

            for count_i in range(0,len(idxs_triads_x[0,:])):
                for count_j in range(0,len(idxs_triads_z[0,:])):

                    i1 = idxs_triads_x[0,count_i]
                    i2 = idxs_triads_x[1,count_i]
                    j1 = idxs_triads_z[0,count_j]
                    j2 = idxs_triads_z[1,count_j]

                    k1 = kk[i1]
                    k2 = kk[i2]
                    n1 = kk_cosine[j1]
                    n2 = kk_cosine[j2]

                    #for i1 in range(0,Nx):
                    #    k1 = kk[i1]
                    #    k2 = k - k1
		    #
                    #    for j1 in range(0,Nz):
                    #        n1 = kk_cosine[j1]
                    #        for j2 in range(0,Nz):
                    #            n2 = kk_cosine[j2]
     		    #
                    #            ksum = k1 + k2
                    #            logical_x = round(ksum,5) == round(k,5)		
 		    #		nsum1 = n1 + n2
                    #	 	nsum2 = n1 - n2
                    #		nsum3 = n2 - n1
                    #		logical_z = round(nsum1,5) == round(kk_cosine[j],5) or\
                    #      	            round(nsum2,5) == round(kk_cosine[j],5) or\
                    #      		    round(nsum3,5) == round(kk_cosine[j],5)
                    #
                    #            if logical_x == True and logical_z == True:                                   

                    kmag1 = sqrt(k1**2 + n1**2)
                    kmag2 = sqrt(k2**2 + n2**2)

                    for alpha1 in range(0,len(alphavec)):
                        for alpha2 in range(0,len(alphavec)):
                                        
                            omega_k1 = alphavec[alpha1]*abs(k1)/kmag1*sqrt(-g*(ct*bt-cs*bs))
                            omega_k2 = alphavec[alpha2]*abs(k2)/kmag2*sqrt(-g*(ct*bt-cs*bs))

                            tmp = omega_k1 + omega_k2 - omega_k
 
                            #Store Omega values for contour plotting: 
                            OmegaArr[v,count_i,count_j,alpha1,alpha2] = tmp
                            OmegaBool[v,count_i,count_j,alpha1,alpha2] = True

                            #Find list of near-resonance Omega values:
                            if abs(tmp) <= OmegaLimit:
                                Omega.append(tmp)

                            #Find near-resonant sets:
                            #if tmp == 0 and (k1 != 0 or k2 != 0):
                                #print('Exact Resonance')
                            if abs(tmp) <= OmegaLimit:
                                #print(omega_k,omega_k1,omega_k2)

                                try: myfile
                                except NameError: 
                                    fnm = 'resonance_sets_' + str(N2) + '.txt'
                                    myfile = open(fnm, 'w')
                                res_set = [alpha,alpha1,alpha2,i,i1,i2,j,j1,j2,tmp]
                                #res_set = [alphavec[alpha], alphavec[alpha1], alphavec[alpha2],\
                                #      kk[i]/(2*pi), kk[i1]/(2*pi), kk[i2]/(2*pi),\
                                #      kk_cosine[j]/pi, kk_cosine[j1]/pi, kk_cosine[j2]/pi]
                                str1 = str(res_set).strip('[]')
                                myfile.write("%s\n" % str1)

                            #print(count0)
                            count0 += 1


        #check max Omega:
        #print(max(Omega))
        #pdb.set_trace()

        try: myfile
        except NameError: var_exists = False
        else: var_exists = True
        if var_exists == True: myfile.close()

        #check code is working as expected:
        minOmega2 = np.min(np.abs(OmegaArr[OmegaBool]))
        if var_exists == True:
            minOmega1 = np.min(np.abs(np.array(Omega)))
            print('min omega: ', minOmega1, minOmega2)
        else:
            print('min omega: ', minOmega2)

        #min_idxs1 = np.where(OmegaArr==minOmega2)
        #print(min_idxs1)

        #Analyse resonant sets in details:
        #Read in all resonant sets:
        res_sets = np.loadtxt(fnm, delimiter=',')

        if AnalyseCapOmega == 1 and var_exists == True:
            tmp = np.sort(np.array(Omega))
            Idxs0orabove = np.where(tmp >= 0)
            ydata = tmp[Idxs0orabove]
            idxs0 = np.where(ydata==0)
            max0 = np.max(idxs0[0])
            if OmegaLimit > 0 and (max0+1) != len(ydata):
                print('first non-zero capOmega: ',ydata[max0+1])
            #ydata = np.sort(np.abs(np.array(Omega)))
            
            SortIdxs = np.argsort(res_sets[:,9])
            tmp = res_sets[SortIdxs,:]
            res_sets_sorted = np.squeeze(tmp[Idxs0orabove,:])

            if listOmegaGrad == 1:
                ydataGrad = np.zeros((len(ydata)))
                for i in range(0,len(ydata)):
                    if i == 0:                  	ydataGrad[i] = ydata[i+1]-ydata[i]
                    if i == len(ydata)-1:               ydataGrad[i] = ydata[i]-ydata[i-1]
                    if (i>0) and (i<len(ydata)-1):      ydataGrad[i] = (ydata[i+1]-ydata[i-1])/2.

                ydataGradSmooth = smooth(ydataGrad,window_len=1001,window='flat')

                fig = plt.figure(1, figsize=(width,height))
                fig.set_tight_layout(True)
                grid = plt.GridSpec(1, 1, wspace=0., hspace=0.)
                ax1 = fig.add_subplot(grid[0,0])
                ax1.plot(ydataGrad, 'k', linewidth=1)
                ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                ax1.grid(True, which='major')
                #ax1.set_yscale("log")
                ax1.set_ylabel(r'$\Omega^{\prime}$ (rad/s)')
                ax1.set_xlabel(r'idx')

                #Find minimum point associated with change in capOmega structure:
                idxS = 0
                idxE = 100000
                MaxIdxs = np.where(ydataGradSmooth[idxS:idxE] == np.max(ydataGradSmooth[idxS:idxE]))[0]
                idxMax = int(np.max(MaxIdxs)) + idxS
                tmp = np.where(ydataGradSmooth[idxMax:idxE] == np.min(ydataGradSmooth[idxMax:idxE]))[0] + idxMax
                minOmegaGradIdx = int(np.max(tmp))

                ax2 = ax1.twinx() 
                ax2.plot(ydataGradSmooth, color='gray', linewidth=2)
                ax2.set_ylim(0,np.max(ydataGradSmooth))
                ax2.plot([minOmegaGradIdx,minOmegaGradIdx],[0,np.max(ydataGrad)], color='gray', linewidth=2)

                plt.show()

                print(res_sets_sorted[minOmegaGradIdx,:])

            if listOmega == 1:
                plt.rcParams.update({'font.size': 18})
                fig = plt.figure(1, figsize=(width,height))
                grid = plt.GridSpec(1, 1, wspace=0., hspace=0.)
                ax2 = fig.add_subplot(grid[0,0])
                fig.set_tight_layout(True)

                #ax2.plot(ydata, 'k', linewidth=2)
                ax2.plot(np.sort(Omega), 'k', linewidth=2)
                ax2.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
                #plt.plot([0,len(ydata)],[ydata[max0],ydata[max0]], 'r')
                #plt.plot([0,len(ydata)],[ydata[minOmegaGradIdx],ydata[minOmegaGradIdx]], 'gray')
                #plt.plot([0,len(ydata)],[.001,.001], 'k')
                #plt.plot([0,len(ydata)],[.01,.01], 'k')
                #plt.plot([0,len(ydata)],[.1,.1], 'k')
                ax2.grid(True, which='major')
                ax2.set_ylabel(r'$\Omega$ (rad/s)')
                ax2.set_xlabel(r'idx')
                plt.show()

                fig = plt.figure(1, figsize=(width,height))
                grid = plt.GridSpec(1, 1, wspace=0., hspace=0.)
                ax3 = fig.add_subplot(grid[0,0])
                fig.set_tight_layout(True)               
 
                #use a log scale:
                ax3.plot(ydata, 'k', linewidth=2)
                ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                #plt.plot([0,len(ydata)],[ydata[max0],ydata[max0]], 'r')
                #ax3.plot([0,len(ydata)],[ydata[minOmegaGradIdx],ydata[minOmegaGradIdx]], 'gray')
                #plt.plot([0,len(ydata)],[.001,.001], 'k')
                #plt.plot([0,len(ydata)],[.01,.01], 'k')
                #plt.plot([0,len(ydata)],[.1,.1], 'k')
                ax3.grid(True, which='major')
                ax3.set_ylabel(r'$\Omega$ (rad/s)')
                ax3.set_xlabel(r'idx')
                ax3.set_yscale("log")
                plt.show()

            if listOmegaPhaseSpace == 1:
                Welch = 1
                spectralCoef, freqvec, nperseg = spectral_analysis(ydata,1,Welch=Welch)
                npersegStr = str(nperseg)
                
                plt.figure()
                plt.plot(freqvec.flatten(), spectralCoef.flatten(), 'k', linewidth=2) 
                plt.ylabel(r'PSD ($[\Omega]^2$)')
                plt.xlabel(r'1/idx')
                plt.tight_layout()
                plt.show()

                #use a log scale:
                plt.figure()
                plt.plot(freqvec.flatten(), spectralCoef.flatten(), 'k', linewidth=2) 
                plt.ylabel(r'PSD ($[\Omega]^2$)')
                plt.xlabel(r'1/idx')
                plt.yscale("log")
                plt.tight_layout()
                plt.show()

            if IntegrateCapOmega == 1:
                
                if PlotWeightFnc == 1:
                    fig0, axs0 = plt.subplots(1,1, figsize=(width,height*2/3.))
                    fig0.subplots_adjust(wspace=0, hspace=0)
                    fig0.set_tight_layout(True)
                    axs0.set_yscale('log')

                NT = 4
                tfact = 6000
                T = (np.arange(NT)+1)*(tfact*dt)
                linewidthvec = np.arange(NT)*0.5+0.5
                capOmegaIntegral = np.zeros((len(ydata),NT))
                capOmegaIntWeights = np.zeros((len(np.array(Omega)),NT)) #for weighting PDFs later.
                for i in range(0,NT):
                    capOmegaIntegral[:,i] = (1./T[i])*(1./(1j*ydata))*(1 - np.exp(-1j*ydata*T[i]))
                    idxsNaN = np.isnan(capOmegaIntegral[:,i])
                    capOmegaIntegral[idxsNaN,i] = 1
                    capOmegaIntegral[:,i] = np.absolute(capOmegaIntegral[:,i])

                    capOmegaIntWeights[:,i] = (1./T[i])*(1./(1j*np.array(Omega)))*(1 - np.exp(-1j*np.array(Omega)*T[i]))
                    idxsNaN = np.isnan(capOmegaIntWeights[:,i])
                    capOmegaIntWeights[idxsNaN,i] = 1
                    capOmegaIntWeights[:,i] = np.absolute(capOmegaIntWeights[:,i])
               
                    if PlotWeightFnc == 1:
                        axs0.plot(capOmegaIntegral[:,i], 'k' ,linewidth=linewidthvec[i], label=str(int(T[i])) )
 
                if PlotWeightFnc == 1:
                    axs0.set_ylabel(r'|$(1/T)\int_0^T\,e^{-i\,\Omega\,s}$ ds|')
                    axs0.set_xlabel(r'idx')
                    axs0.legend(title=r'$T$ (s)')
                    plt.show()

                    plt.figure()
                    plt.plot(capOmegaIntWeights[:,NT-1], 'k', label=str(int(T[i])) )
                    plt.ylabel(r'|$(1/T)\int_0^T\,e^{-i\,\Omega\,s}$ ds|')
                    plt.xlabel(r'idx')
                    plt.legend(title=r'$T$ (s)')
                    plt.show()
               

        #Find unique set of triads ignoring occurences due to different alpha combinations:
        if FindUniqueSet == 1:
            idxs = np.unique(res_sets[:,3:9], axis=0, return_index=True)
            res_sets2 = res_sets[np.sort(idxs[1]),:]
 
        #Only look at particular (alpha,alpha1,alpha2) combination and take advantage of
        #symmetry in the alpha's:
        if FindOneAlphaSet == 1:
            if keyWaveModes == 1 and keyNonWaveModes == 0: 
                tmp1 = (res_sets[:,0:3] == (0,0,0)).all(axis=1).nonzero()[0]
                tmp2 = (res_sets[:,0:3] == (0,0,1)).all(axis=1).nonzero()[0]
                tmp3 = (res_sets[:,0:3] == (0,1,0)).all(axis=1).nonzero()[0]
                tmp4 = (res_sets[:,0:3] == (0,1,1)).all(axis=1).nonzero()[0]
                tmp5 = (res_sets[:,0:3] == (1,0,0)).all(axis=1).nonzero()[0]
                tmp6 = (res_sets[:,0:3] == (1,0,1)).all(axis=1).nonzero()[0]
                tmp7 = (res_sets[:,0:3] == (1,1,0)).all(axis=1).nonzero()[0]
                tmp8 = (res_sets[:,0:3] == (1,1,1)).all(axis=1).nonzero()[0]
                #idxs = np.concatenate((tmp4,tmp5))
                idxs = np.concatenate((tmp1,tmp2,tmp3,tmp6,tmp7,tmp8))
                #idxs = np.concatenate((tmp1,tmp2,tmp3))
                #idxs = np.concatenate((tmp6,tmp7,tmp8))
            if keyNonWaveModes == 1 and keyWaveModes == 0: 
                tmp1 = (res_sets[:,0:3] == (1,0,0)).all(axis=1).nonzero()[0]
                tmp2 = (res_sets[:,0:3] == (1,0,1)).all(axis=1).nonzero()[0]
                tmp3 = (res_sets[:,0:3] == (1,1,0)).all(axis=1).nonzero()[0]
                tmp4 = (res_sets[:,0:3] == (1,1,1)).all(axis=1).nonzero()[0]
                idxs = tmp2
                #idxs = tmp3
            #Check ordering:
            #print(idxs)
            #plt.plot(idxs)
            #plt.show()
            res_sets2 = res_sets[idxs,:]
            #np.savetxt('./res_sets2.txt',res_sets2)
            capOmegaWeights = capOmegaIntWeights[idxs,NT-1].flatten()

        if (keyWaveModes == 1 and keyNonWaveModes == 0) or (keyNonWaveModes == 1 and keyWaveModes == 0):
            Nk2 = res_sets2.shape[0]
            print("# of modes below some Omega limit: ", Nk2)

            #Initialise arrarys for historgram plots:
            if ComputePDF == 1 or InteractionCoef == 1 or SigmaSigma == 1:
                omega12_vec = np.zeros((Nk2*2))
                n12_vec = np.zeros((Nk2*2))          
                omega_vec = np.zeros((Nk2))
                n_vec = np.zeros((Nk2))
                color_vec = np.zeros((Nk2))
               
            if FindMaxModes == 1:
                i_vec = np.zeros((Nk2))
                j_vec = np.zeros((Nk2))
                kmag12ij_vec = np.zeros((Nk2*2)) 
 
            if InteractionCoef == 1:
                interactCoef = np.zeros((Nk2))
            if SigmaSigma == 1:
                sigmaSigma = np.zeros((Nk2,Nt))*1j
            if WaveSpeeds == 1:
                cgx_vec = np.zeros((Nk))
                cgx_n_vec = np.zeros((Nk))
                count1 = 0

            if Plot3D == 1:
                fig = plt.figure()
                ax = fig.gca(projection='3d')

            #Scale symbol size by magnitude of resonance (i.e. the phase):
            OmegaMax = np.max(np.abs(res_sets2[:,9]))
            if OmegaLimit > 0 and OmegaMax != 0: 
                AbsOmegaMin1 = np.abs(res_sets2[:,9])/OmegaMax + 1
                #1 added to Omega values here so that direct resonance points scale.
                OmegaScaled = 1./AbsOmegaMin1
                symsize = OmegaScaled*5
            else: symsize = np.ones(len(res_sets2[:,9]))*5

            #If only a few main modes then set colours manually below, otherwise
            #automatically find colors from a continuous colorscale:
            if ManualColors == 0: 
                #cmap = cm.get_cmap('Blues')
                cmap = cm.get_cmap('Greys')
                count0 = 0
                #avoid ends of colormap (i.e. very white and black shades):
                cmap0 = 0.2
                if keyWaveModes == 1: dnmntr = Nk/2.
                else: dnmntr = Nk
                cmap_step = (1-cmap0-cmap0/2.)/dnmntr

            #Make dynamic history of plotted sets to enable plot symbols and colours to reveal
            #resonant structure and symmetry:
            past_key_modes = []
            #past_modes = []

            #Plot symbol transparency parameter:
            alphaC=0.5


            count00 = 0
            count01 = 0

            for ii in range(0,Nk2):

                alpha = int(res_sets2[ii,0])
                alpha1 = int(res_sets2[ii,1])
                alpha2 = int(res_sets2[ii,2])

                i = int(res_sets2[ii,3])
                j = int(res_sets2[ii,6])
                i1 = int(res_sets2[ii,4])
                j1 = int(res_sets2[ii,7])
                i2 = int(res_sets2[ii,5])
                j2 = int(res_sets2[ii,8])

                k = kk[i]
                n = kk_cosine[j]
                kvec = np.array([k,n])
                kmag = np.linalg.norm(kvec)
                alt = k/kmag
                #alt = np.arccos(cos_theta)*180/np.pi
                omega_k = alphavec[alpha]*abs(k)/kmag*sqrt(-g*(ct*bt-cs*bs))
                #omega_k = omega_k/(2*np.pi)
                if omega_k != 0:
                    c = omega_k/kmag
                    cg_x = alphavec[alpha]*sqrt(-g*(ct*bt-cs*bs))*(1/kmag - np.abs(k)*k/kmag**3)
                    cg_z = alphavec[alpha]*sqrt(-g*(ct*bt-cs*bs))*(-np.abs(k)*n/kmag**3)
                else:
                    c = 0
                    cg_x = 0
                    cg_z = 0

                k1 = kk[i1]
                n1 = kk_cosine[j1]
                kvec1 = np.array([k1,n1])
                kmag1 = np.linalg.norm(kvec1)
                alt1 = k1/kmag1
                #alt1 = np.arccos(cos_theta1)*180/np.pi
                omega_k1 = alphavec[alpha1]*abs(k1)/kmag1*sqrt(-g*(ct*bt-cs*bs))
                #omega_k1 = omega_k1/(2*np.pi)
                if omega_k1 != 0:
                    c1 = omega_k1/kmag1
                    cg_x1 = alphavec[alpha1]*sqrt(-g*(ct*bt-cs*bs))*(1/kmag1 - np.abs(k1)*k1/kmag1**3)
                    cg_z1 = alphavec[alpha1]*sqrt(-g*(ct*bt-cs*bs))*(-np.abs(k1)*n1/kmag1**3)
                else:
                    c1 = 0
                    cg_x1 = 0
                    cg_z1 = 0       

                k2 = kk[i2]
                n2 = kk_cosine[j2]
                kvec2 = np.array([k2,n2])
                kmag2 = np.linalg.norm(kvec2)
                alt2 = k2/kmag2
                #alt2 = np.arccos(cos_theta2)*180/np.pi
                omega_k2 = alphavec[alpha2]*abs(k2)/kmag2*sqrt(-g*(ct*bt-cs*bs))
                #omega_k2 = omega_k2/(2*np.pi)
                if omega_k2 != 0:
                    c2 = omega_k2/kmag2
                    cg_x2 = alphavec[alpha2]*sqrt(-g*(ct*bt-cs*bs))*(1/kmag2 - np.abs(k2)*k2/kmag2**3)
                    cg_z2 = alphavec[alpha2]*sqrt(-g*(ct*bt-cs*bs))*(-np.abs(k2)*n2/kmag2**3)
                else:
                    c2 = 0
                    cg_x2 = 0
                    cg_z2 = 0

                if xAxis_angle == 1: angle = np.dot(kvec1,kvec2)/(kmag1*kmag2)

                #Save data for histograms:
                if ComputePDF == 1 or InteractionCoef == 1:
                    omega12_vec[2*ii] = omega_k1
                    omega12_vec[2*ii+1] = omega_k2
                    n12_vec[2*ii] = n1
                    n12_vec[2*ii+1] = n2 
                    omega_vec[ii] = omega_k
                    n_vec[ii] = n
                    
                if FindMaxModes == 1:
                    i_vec[ii] = i
                    j_vec[ii] = j
                    vec1 = np.array([i1,j1])
                    vec2 = np.array([i2,j2])
                    kmag1ij = np.linalg.norm(vec1)
                    kmag2ij = np.linalg.norm(vec2)
                    kmag12ij_vec[2*ii] = kmag1ij
                    kmag12ij_vec[2*ii+1] = kmag2ij

                if WaveSpeeds == 1:
                    if ([i,j] not in past_key_modes):
                        Idx = count1
                        cgx_vec[Idx] = cg_x
                        cgx_n_vec[Idx] = n
                        count1 += 1

                if InteractionCoef == 1:
                    if alpha == 0: eigenvec = ivec_1[i,j,:]
                    if alpha == 1: eigenvec = ivec1[i,j,:]
                    if alpha1 == 0: eigenvec1 = ivec_1[i1,j1,:] 
                    if alpha1 == 1: eigenvec1 = ivec1[i1,j1,:] 
                    if alpha2 == 0: eigenvec2 = ivec_1[i2,j2,:] 
                    if alpha2 == 1: eigenvec2 = ivec1[i2,j2,:]
                    matrix1 = np.zeros((2,2))
                    matrix2 = np.zeros((2,2))
                    matrix1[0,0] = kmag2/kmag
                    matrix1[1,1] = 1
                    matrix2[0,0] = kmag1/kmag
                    matrix2[1,1] = 1
                    interactCoef[ii] = -(k1*n2-k2*n1)*(\
                    1./kmag1*eigenvec1[0]*np.dot( (np.mat(matrix1)*np.mat(eigenvec2).reshape((2,1))).reshape((2,)), eigenvec)-\
                    1./kmag2*eigenvec2[0]*np.dot( (np.mat(matrix2)*np.mat(eigenvec1).reshape((2,1))).reshape((2,)), eigenvec))
                    #n.b. np.dot requires vectors to have shape (2,) rather than (2,1), but also we need to give vector 
                    #the shape (2,1) so that inner dimensions agree during the matrix multiplication.

                    #check computation:
                    #if k1!=0 and k2!=0 and n1!=0 and n2!=0: 
                    #if round(interactCoef[ii],4)==0: count00 += 1
                    #if round((k1*n2-k2*n1),4)==0: count01 += 1
                    #if interactCoef[ii]==0: count00 += 1
                    #if (k1*n2-k2*n1)==0: count01 += 1

                if SigmaSigma == 1:
                    if alpha1 == 0: sig1 = sigma_1_3D[i1,j1,:]
                    if alpha1 == 1: sig1 = sigma1_3D[i1,j1,:]
                    if alpha2 == 0: sig2 = sigma_1_3D[i2,j2,:]
                    if alpha2 == 1: sig2 = sigma1_3D[i2,j2,:]
                    sigmaSigma[ii,:] = sig1*sig2 


                #Symbol/line colour selections:
                if keyWaveModes == 1 and ManualColors == 1:
                    if Modulated == 1:
                        if N2 == 2.25:
                            if i==1 and j==1: color='k'
                            if i==1 and j==3: color='grey'
                            if i==1 and j==5: color='silver'
                        if N2 == 9:
                            if i==1 and j==3: color='k'
                            if i==1 and j==4: color='grey'
                            if i==1 and j==5: color='silver'
                    if Modulated == 0:
                        if N2 == 2.25:
                            if i==1 and j==1: color='k'
                            if i==1 and j==2: color='dimgrey'
                            if i==1 and j==3: color='grey'
                            if i==1 and j==4: color='darkgrey'
                            if i==1 and j==5: color='silver'
                            if i==2 and j==5: color='silver'
                            if i==1 and j==6: color='lightgrey'
                            if i==1 and j==7: color='gainsboro'
                            if i==1 and j==8: color='whitesmoke'

                if ManualColors == 0:
                    #Group triads by key mode using colouring:
                    #n.b. need to order resonant sets by key mode to keep plot clear.
                    if ([i,j] not in past_key_modes):
                        cIdx = cmap_step*count0 + cmap0
                        color = cmap(cIdx) 
                        count0 += 1
                    if InteractionCoef == 1 or SigmaSigma == 1: 
                        color_vec[ii] = cIdx

                #Change line style to reveal alpha symmetries: 
                if (alpha1 == 0) and (alpha2 == 0): linestyle = "-"
                if (alpha1 == 1) and (alpha2 == 1): linestyle = "-"
                if (alpha1 == 1) and (alpha2 == 0): linestyle = "--"
                if (alpha1 == 0) and (alpha2 == 1): linestyle = "--"

                #Change symbol colors to reveal repeat occurences:
                #if ([i1,i2,j1,j2] not in past_modes): fill = 'top'
                #else: fill = 'bottom'
                #Too difficult to visualise this so better to discuss in text.

                fill = 'full'

                #choose linewidth
                lw = 0.5

                if Plot3D == 0:
                    if xAxis_k == 1:
                        plt.plot([k1,k2],[omega_k1,omega_k2],\
                        color=color, marker='o', markersize=symsize[ii], alpha=alphaC, linestyle=linestyle, linewidth=lw, fillstyle=fill)
                        plt.plot(k,omega_k, color=color, marker='o', markersize=15, fillstyle='none')
                    if xAxis_n == 1:
                        plt.plot([n1,n2],[omega_k1,omega_k2],\
                        color=color, marker='o', markersize=symsize[ii], alpha=alphaC, linestyle=linestyle, linewidth=lw, fillstyle=fill)
                        plt.plot(n,omega_k, color=color, marker='o', markersize=15, fillstyle='none')
                    if xAxis_r == 1:
                        plt.plot([kmag1,kmag2],[omega_k1,omega_k2],\
                        color=color, marker='o', markersize=symsize[ii], alpha=alphaC, linestyle=linestyle, linewidth=lw, fillstyle=fill)
                        plt.plot(kmag,omega_k, color=color, marker='o', markersize=15, fillstyle='none')
                    if xAxis_alt == 1:
                        plt.plot([alt1,alt2],[omega_k1,omega_k2],\
                        color=color, marker='o', markersize=symsize[ii], alpha=alphaC, linestyle=linestyle, linewidth=lw, fillstyle=fill)
                        plt.plot(alt,omega_k, color=color, marker='o', markersize=15, fillstyle='none')
                    if xAxis_angle == 1:
                        plt.plot([angle1,angle2],[omega_k1,omega_k2],\
                        color=color, marker='o', markersize=symsize[ii], alpha=alphaC, linestyle=linestyle, linewidth=lw, fillstyle=fill)
                        plt.plot(angle,omega_k, color=color, marker='o', markersize=15, fillstyle='none')
                    if xAxis_c == 1:
                        plt.plot([c1,c2],[omega_k1,omega_k2],\
                        color=color, marker='o', markersize=symsize[ii], alpha=alphaC, linestyle=linestyle, linewidth=lw, fillstyle=fill)
                        plt.plot(c,omega_k, color=color, marker='o', markersize=15, fillstyle='none')
                    if xAxis_cg_x == 1:
                        plt.plot([cg_x1,cg_x2],[omega_k1,omega_k2],\
                        color=color, marker='o', markersize=symsize[ii], alpha=alphaC, linestyle=linestyle, linewidth=lw, fillstyle=fill)
                        plt.plot(cg_x,omega_k, color=color, marker='o', markersize=15, fillstyle='none')
                    if xAxis_cg_z == 1:
                        plt.plot([cg_z1,cg_z2],[omega_k1,omega_k2],\
                        color=color, marker='o', markersize=symsize[ii], alpha=alphaC, linestyle=linestyle, linewidth=lw, fillstyle=fill)
                        plt.plot(cg_z,omega_k, color=color, marker='o', markersize=15, fillstyle='none')

                if Plot3D == 1:
                    #ax.plot([k,k1,k2,k], [n,n1,n2,n], [omega_k,omega_k1,omega_k2,omega_k], color=color, marker='o', markersize=symsize)
                    ax.plot([k1,k2], [n1,n2], [omega_k1,omega_k2],\
                    color=color, marker='o', markersize=symsize[ii], alpha=alphaC, linestyle=linestyle, linewidth=lw, fillstyle=fill)
                    #if ([i,j] not in past_key_modes):
                    ax.plot([k,k], [n,n], [omega_k,omega_k], color=color, marker='o', markersize=15, fillstyle='none')

                past_key_modes.append([i,j])
                #past_modes.append([i1,i2,j1,j2])

            #print(count00,count01)

            if FindMaxModes == 1:
                #plt.figure()
                #plt.plot(k_vec)
                #plt.show()

                #plt.figure()
                #plt.plot(n_vec)
                #plt.show()

                maxModes = np.zeros((Nk))
                xvec = np.zeros((Nk))
                count0 = 0
                past_key_modes = []
                for ii in range(0,Nk2):
                    #i = i_vec[ii]
                    #j = j_vec[ii]
                    omega = omega_vec[ii]
                    #if ([i,j] not in past_key_modes):
                    if (omega not in past_key_modes):
                        #bool_i = i_vec == i
                        #bool_j = j_vec == j             
                        #tmp = np.multiply(bool_i,bool_j)
                        #bool_ij = np.repeat(tmp,2)
                        #maxModes[count0] = np.max(kmag12ij_vec[bool_ij])

                        tmp = omega_vec==omega
                        bool_omega = np.repeat(tmp,2)
                        maxModes[count0] = np.max(np.abs(omega12_vec[bool_omega]))

                        #vec = np.array([i,j])
                        #kmag_ij = np.linalg.norm(vec)            
                        #xvec[count0] = kmag_ij
                        xvec[count0] = np.abs(omega)
                        

                        #past_key_modes.append([i,j])
                        past_key_modes.append(omega)
                        count0 += 1


            #print(omega_vec)
            if Plot3D == 0:
                #xvec = np.arange(Nk*2)                      
                #plt.scatter(kmag_vec,omega_vec, c='k')

                if xAxis_r == 1: plt.xlabel(r'$|\mathbf{k}|,|\mathbf{k}_1|,|\mathbf{k}_2|$ (rad/m)')
                if xAxis_k == 1: plt.xlabel(r'$k,k_1,k_2$ (rad/m)')
                if xAxis_n == 1: plt.xlabel(r'$n,n_1,n_2$ (rad/m)')
                if xAxis_alt == 1: plt.xlabel(r'$\cos(\phi),\cos(\phi_1),\cos(\phi_2)$')
                if xAxis_c == 1: plt.xlabel(r'$c=\omega/|\mathbf{k}|$ (m/s)')
                if xAxis_cg_x == 1: plt.xlabel(r'$c_{g_{x}}$ (m/s)')
                if xAxis_cg_z == 1: plt.xlabel(r'$c_{g_{z}}$ (m/s)')
                #plt.ylabel(r'$\omega/(2\pi)$')
                plt.ylabel(r'$\omega_{\bf k}^{\alpha},\omega_{{\bf k}_1}^{\alpha_1},\omega_{{\bf k}_2}^{\alpha_2}$ (rad/s)')
            if Plot3D == 1:
                ax.view_init(elev=130, azim=30)
                ax.set_xlabel(r'$k,k_1,k_2$ (rad/m)')
                ax.set_ylabel(r'$n,n_1,n_2$ (rad/m)')
                ax.set_zlabel(r'$\omega_{\bf k}^{\alpha},\omega_{{\bf k}_1}^{\alpha_1},\omega_{{\bf k}_2}^{\alpha_2}$ (rad/s)')
            plt.show()


            if FindMaxModes == 1:
                #plot results:
                fig_0 = plt.figure(1, figsize=(width,height))
                fig_0.set_tight_layout(True)
                grid_0 = plt.GridSpec(1, 1, wspace=0., hspace=0.)
                axs_0 = fig_0.add_subplot(grid_0[0,0])
                axs_0.plot(xvec,maxModes, '^', c='k')
                #axs_0.set_xlabel('Wavenumber (non-dimensional) of mode of modulated system')
                #axs_0.set_ylabel('Max wavenumber (non-dimensional) of linear waves\n associated with some mode of solution')
                axs_0.set_xlabel('Mode (frequency) of modulated system (rad/s)')
                axs_0.set_ylabel('Max frequency of linear waves\n associated with some mode of solution (rad/s)')
                plt.show()

                pdb.set_trace()


            #Make histograms of triads:
            if ComputePDF == 1:

                #compute PDF sums of IGW and meanflow modes defined using well mode:
                tmp1 = np.loadtxt('/home/ubuntu/BoussinesqLab/dedalus/meanflowarr.txt')
                tmp2 = np.loadtxt('/home/ubuntu/BoussinesqLab/dedalus/psdIGWarr.txt')
                N_vec_tmp = np.array((0.5, 1, 1.5, 2, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.5, 5))
                Nidx = np.where(N_vec_tmp == np.sqrt(N2))
                wellmode = tmp1[int(Nidx[0]),3]
                maxIGWmode = tmp2[int(Nidx[0]),2] + wellmode
                print('well mode: ',wellmode)

                if keyNonWaveModes == 1:
                    #For this k=0 case if either k1 or k2 are zero then they both must be zero (i.e. no contribution):
                    idxsNon0 = np.where(omega12_vec != 0)
                    logVec1 = omega12_vec != 0
                if keyWaveModes == 1:
                    #For this case of k not 0, k1 and k2 are not both necessarily zero together so we need a
                    #more careful search. Fine all cases when k1 and k2 are zero together:
                    logVec1 = np.ones((2*Nk2))
                    for ww in range(0,Nk2):
                        logical = (omega12_vec[2*ww] == 0) and (omega12_vec[2*ww+1] == 0)
                        if logical: logVec1[2*ww:2*ww+1] = 0
                    idxsNon0 = np.where(logVec1 != 0)               

                if InteractionCoef == 1:
                    #Account for zeros from interaction coefficient:
                    interactCoef12 = np.repeat(interactCoef,2)
                    logVec2 = interactCoef12 != 0  
                    #tmp = np.round(interactCoef12, decimals=5)
                    #logVec2 = tmp != 0

                    logVec3 = np.multiply(logVec1,logVec2)
                    idxsNon0 = np.where(logVec3 != 0)
                    #check operations:
                    #print(logVec1[0:50])
                    #print(logVec2[0:50])
                    #print(logVec3[0:50])

                if FrequencyAve == 1 and OmegaLimit != 0: 
                    #weights = np.repeat(OmegaScaled,2)
                    #weights = capOmegaWeights*np.abs(interactCoef)
                    weights = capOmegaWeights
                    weights = np.repeat(weights,2)
                else: weights = np.ones((len(n12_vec)))

                #color by key (wave or non-wave) mode:
                colors = np.repeat(color_vec,2)

                density = False

                hist, bin_edges = np.histogram(n12_vec[idxsNon0], bins=kk_cosine, weights=weights[idxsNon0], density=density)
                MaxCountIdx1 = np.where(hist==np.max(hist))
                print('most common bin: ', bin_edges[MaxCountIdx1])
                plt.hist(n12_vec[idxsNon0], bins=kk_cosine, weights=weights[idxsNon0], density=density, color='k' )
                plt.xlabel(r'$n_1,\,n_2$ (rad/m)')
                if density: plt.ylabel(r'PDF')
                else: plt.ylabel(r'weighted counts') 
                plt.show()


                sortedOmega12 = np.unique(np.sort(omega12_vec[idxsNon0]))
                omega12diff = sortedOmega12[1] -  sortedOmega12[0]
                print('smallest non-zero angular frequency difference: ', omega12diff)
                
                #print( np.min(kk[kk>0]) )
                domega = np.min(kk[kk>0])/np.sqrt(np.max(kk)**2+np.max(kk_cosine)**2)*np.sqrt(N2)              
                #print(domega)
 
                omega0 = -np.sqrt(N2)
                omegaMax = np.sqrt(N2)
                Nomega = 400 + 1
                domega = (omegaMax-omega0)/(Nomega-1)
                omega_bins = np.arange(Nomega)*domega + omega0

                hist, bin_edges = np.histogram(omega12_vec[idxsNon0], bins=omega_bins, weights=weights[idxsNon0], density=density)
                MaxCountIdx2 = np.where(hist==np.max(hist))
                print('most common bin: ', bin_edges[MaxCountIdx2])

                fig = plt.figure(1, figsize=(width,height))
                fig.set_tight_layout(True)
                grid = plt.GridSpec(1, 1, wspace=0., hspace=0.)
                ax1 = fig.add_subplot(grid[0,0])
                #logy = True
                logy = False
                ax1.hist(omega12_vec[idxsNon0], bins=omega_bins, weights=weights[idxsNon0], density=density, color='k', log=logy)
                if logy: ax1.set_ylim(1E-1,1E4)
                ax1.set_xlabel(r'$\omega^{\alpha_1}_{{\bf k}_1},\,\omega^{\alpha_2}_{{\bf k}_2}$ (rad/s)')
                if density: ax1.set_ylabel(r'PDF')
                else: ax1.set_ylabel(r'weighted counts')

                ymin,ymax = plt.ylim()
                xpos = (maxIGWmode-wellmode)*(1./3) + wellmode
                if logy==0: ypos = ymax/2
                else: ypos = 1E3
                #set hatch density:
                c_h = 3
                #if keyWaveModes == 1:
                #    #colour and label IGW region:
                #    ax1.fill_between([wellmode,maxIGWmode],0,ymax, facecolor='whitesmoke')
                #    ax1.text(xpos, ymax/2, 'IGW', ha='center', va='center', fontsize=12)
                #    #colour and label MF region:
                #    ax1.fill_between([0,wellmode],0,ymax, facecolor='silver')                
                #    ax1.text(wellmode/2, ymax/2, 'MF', ha='center', va='center', fontsize=12)
                if keyNonWaveModes == 1 or keyWaveModes == 1: 
                    #colour and label IGW region:
                    ax1.fill_between([-maxIGWmode,wellmode],0,ymax, facecolor='whitesmoke')
                    ax1.fill_between([wellmode,maxIGWmode],0,ymax, facecolor='whitesmoke')
                    ax1.text(-xpos, ypos, 'IGW', ha='center', va='center', fontsize=12)
                    ax1.text(xpos, ypos, 'IGW', ha='center', va='center', fontsize=12)
                    #colour and label MF region:
                    ax1.fill_between([-wellmode,0],0,ymax, facecolor='silver')
                    ax1.fill_between([0,wellmode],0,ymax, facecolor='silver')                
                    ax1.text(0, ypos, 'LFMF', ha='center', va='center', fontsize=12)
              
                plt.show()

                if ComputeCumulativeDistFnc == 1:
                    tmp1 = np.where(np.abs(bin_edges) > wellmode)
                    tmp2 = np.where(np.abs(bin_edges) <= wellmode)
                    idxsIGW = tmp1[0][0:len(tmp1[0])-1]
                    idxsMF = tmp2[0]
                    sumIGW = np.sum(hist[idxsIGW])
                    sumMF = np.sum(hist[idxsMF])
                    print('total counts: ', np.sum(hist))
                    print('IGW/MF: ', sumIGW,sumMF)
                    IGWFraction = sumIGW/np.sum(hist)
                    MeanflowFraction = sumMF/np.sum(hist)
                    print(IGWFraction,MeanflowFraction)


            if WaveSpeeds == 1:
                maxIdx = np.where(cgx_vec==np.max(cgx_vec))
                plt.plot(cgx_n_vec,cgx_vec)
                plt.show()
                print(cgx_n_vec)
                print(cgx_vec[maxIdx])
                print(cgx_n_vec[maxIdx])
                print(1./(cgx_n_vec[maxIdx]/np.pi))
                print(0.45/cgx_vec[maxIdx])

            if PlotInteractCoef == 1:

                fig00, ax00 = plt.subplots(1,1, figsize=(width,height))
                fig00.subplots_adjust(wspace=0.4, hspace=0.)
                fig00.set_tight_layout(True)

                xdata = omega12_vec[0::2]
                ydata = interactCoef
                colors = color_vec
                
                #xdata = omega12_vec
                #ydata = np.repeat(interactCoef,2)
                #colors = np.repeat(color_vec,2)

                ax00.scatter(xdata,ydata, color=cmap(colors), marker='.', s=symsize*10, alpha=0.5)
                ax00.scatter(omega_vec,interactCoef, edgecolors=cmap(color_vec), marker='o', s=30, facecolors='none', alpha=0.5)
                ax00.set_xlabel(r'$\omega^{\alpha}_{{\bf k}}, \omega^{\alpha_1}_{{\bf k}_1}, \omega^{\alpha_2}_{{\bf k}_2}$ (rad/s)')
                ax00.set_ylabel(r'$C^{\alpha, \alpha_1, \alpha_2}_{{\bf k}, {\bf k}_1, {\bf k}_2}$')
                ax00.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                plt.show()

            if SigmaSigma == 1:

                fig01, ax01 = plt.subplots(1,1, figsize=(width,height))
                fig01.subplots_adjust(wspace=0.4, hspace=0.)
                fig01.set_tight_layout(True)
                #ax01.set_yscale('log')
                ax01.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

                ax01.set_xlabel(r'$t$ (s)')
                if CombineSigmaSigmaWithC == 0:
                    xdata = t
                    ydata = sigmaSigma
                    weight = 1
                    ax01.set_ylabel(r'$\sigma^{\alpha_1}_{{\bf k}_1}\,\sigma^{\alpha_2}_{{\bf k}_2}$ ([$\zeta$])')
                    idxE = Nk2
                else:
                    
                    xdata = omega12_vec[0::2]
                    ydata = sigmaSigma
                    weight = interactCoef
                    ax01.set_ylabel(r'$\sigma^{\alpha_1}_{{\bf k}_1}\,\sigma^{\alpha_2}_{{\bf k}_2}\,C^{\alpha, \alpha_1, \alpha_2}_{{\bf k}, {\bf k}_1, {\bf k}_2}$ ([$\zeta$])')
                    #idxE = Nt
                    idxE = Nk2
             
                for i in range(0,idxE):
                    if CombineSigmaSigmaWithC == 0: 
                        ax01.plot(xdata,ydata[i,:]*weight, '-k', linewidth=3, color=cmap(color_vec[i]) )
                    if CombineSigmaSigmaWithC == 1: 
                        #ax01.scatter(xdata,ydata[:,i]*weight, marker='o', color=cmap(color_vec) )
                        ax01.scatter(t,ydata[i,:]*weight[i], marker='o', color=cmap(color_vec[i]) )
                plt.show() 

                ComplexPlane = 0
                if ComplexPlane == 1:
                    fig02 = plt.figure()
                    ax02 = fig02.gca(projection='3d')
                    for c in range(0,Nk2):
                        ax02.plot(sigmaSigma[c,:].real,sigmaSigma[c,:].imag,t,  '-k', linewidth=3, color=cmap(color_vec[c]) )

                    plt.show()

        #Find unique set of resonance triads and construct salinity profile
        #by linear superposition of waves:
        #res_sets_uniq = np.unique(res_sets, axis=0)
        #S_res_1d = 0
        #for i in range(0,len(res_sets_uniq[:,0])):
        #    for j in range(0,3):
        #        k = kk[int(res_sets_uniq[i,j])]
        #        n = kk_cosine[int(res_sets_uniq[i,j+3])]
        #        S_res_1d += np.sin(n*z)

        #What happens when you add up all possible n modes:
        S_res_1d = 0
        for i in range(0,len(kk_cosine)):
            n = kk_cosine[i]
            S_res_1d += np.sin(n*z)

        #plt.plot(S_res_1d,z)
        #plt.show()
        #pdb.set_trace()
 
        #Use Dedalus transforms to find spectrum:
        #S_res_2d_d = domain.new_field()
        #S_res_2d_d.meta['z']['parity'] = -1
        #S_res_2d = np.tile(S_res_1d,(Nx,1))
        #S_res_2d_d['g'] = S_res_2d
        #psd = abs(S_res_2d_d['c'][0,:])**2 # there is no x-variation so choose the zero mode.
        #
        #Find number of steps from vertical gradient of salinity.
        #First add background salinity field to perturbations:
        #S_res_2d += Sbase
        #
        #Compute vertical salinity gradient:
        #def d_dz(f,Nx,Nz,z):
        #    fz = np.zeros((Nx,Nz))
        #    for jj in range(0,Nz):
        #        for ii in range(0,Nx): 
        #            #Use centered scheme except next to boundaries:
        #            if (jj != 0) and (jj != Nz-1): df = f[ii,jj+1] - f[ii,jj-1]
        #            if jj == 0: df = f[ii,jj+1] - f[ii,jj]
        #            if jj == Nz-1: df = f[ii,jj] - f[ii,jj-1]
        #
        #            if (jj != 0) and (jj != Nz-1): fz[ii,jj] = df/(2*dz)
        #            else: fz[ii,jj] = df/dz
        #    return fz
        #
        #Sz_res = d_dz(S_res_2d,Nx,Nz,z) 
        #
        #Plot the vertical salinity gradient:
        #cmap = 'PRGn'
        #xgrid = x2d
        #ygrid = z2d
        #plt.contourf(xgrid,ygrid,Sz_res.transpose(),50,cmap=cmap)
        #plt.colorbar()
        #plt.show()
        #
        #Count the number of steps using simplified step-tracking software:
        #zIdx_offset = 0
        #Sz_res_min = np.min(abs(Sz_res))
        #GammaLimit = bs*0.01
        #tmp1 = np.zeros((Nx,Nz),dtype=bool)
        #xIdx = int(Nx/2.)
        #
        #for jj in range(0+zIdx_offset,Nz-zIdx_offset):
        #    #if (Sz_res[xIdx,jj] <= 0):
        #    if abs(Sz_res[xIdx,jj]) <= GammaLimit:
        #        tmp1[:,jj] = True
        #    else: tmp1[:,jj] = False
        #    #else: tmp1[:,jj] = False
        #
        #    n_steps = 0
        #    flag = 0
        #    for jj in range(0+zIdx_offset,Nz-zIdx_offset):
        #        if (tmp1[xIdx,jj] == True) and (flag == 0):
        #            n_steps += 1
        #            flag = 1
        #            j0 = jj
        #        if (tmp1[xIdx,jj] == False):
        #            flag = 0
        #
        #logical_not0 = S_res_2d > (0+np.max(S_res_1d)*0.1)
        #logic1 = np.logical_and( logical_not0, tmp1)
        #plt.contourf(tmp1.transpose(), 1, colors=['white','black'])
        #plt.show()
        #print(n_steps)
        #
        #estimate number of steps: 
        #idx_max = np.where(psd == np.max(psd))
        #wavelength = (1./(kk_cosine[idx_max]/pi))
        #n_steps = Lz/wavelength
        

        
        #Plot results:
        if MakePlot == 1:
            if PlotMainTriads == 1:

                fig0, axs0 = plt.subplots(1,2, figsize=(width,height)) 
                fig0.subplots_adjust(wspace=0.4, hspace=0.2)

                axs0[0].plot(S_res_1d,z)
                axs0[0].plot([0,0],[0,Lz],'grey')
                axs0[0].set_ylim(0,Lz)
                axs0[0].set_xlabel(r'$S$')
                axs0[0].set_ylabel(r'$z$ (m)')

                axs0[1].plot(kk_cosine/pi,psd, '.-k')
                axs0[1].set_xlabel(r'$f$ (1/m)')
                axs0[1].set_ylabel('PSD')
                axs0[1].set_xlim(0,np.max(kk_cosine)/pi/2)

                label = 'max(PSD)=' + str(psd[idx_max]).strip("[]") + '\n'\
                        + 'wavelength(max(PSD))=' + str(wavelength).strip("[]") + ' (m)' + '\n'\
                        + '# steps=' + str(n_steps).strip("[]")
                axs0[1].text(0.5, 0.8, label, dict(size=8),\
                horizontalalignment='center', verticalalignment='center', transform=axs0[1].transAxes)

                plt.show()

            if contourOmega == 1:

                if keyWaveModes == 1: 
                    rows = int(Nk/2.)
                    row0 = 0
                    row0 = int(Nk/2.)
                if keyNonWaveModes == 1: 
                    rows = Nk
                    row0 = 0
  
                fig0, axs0 = plt.subplots(rows,4, figsize=(width,height), sharex='all', sharey='all')
                fig0.subplots_adjust(wspace=0.2, hspace=0.2)

                #OmegaRange = np.max(OmegaArr)
                OmegaRange = 4.
                nlevs = 41
                levels = np.arange(nlevs)*(OmegaRange/(nlevs-1))-OmegaRange/2.
                #print(levels)
                #cmap = plt.get_cmap('PiYG')
                #cmap = plt.get_cmap('RdPu')
                #cmap = plt.get_cmap('RdGy')
                #cmap = plt.get_cmap('PRGn')
                cmap = plt.get_cmap('bwr')
                norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

                idxs_min_bool = OmegaArr==0
                logic = np.logical_and(idxs_min_bool,OmegaBool)
                marker='.'
                s=5
                marker_col = 'k'
                mfc='k'

                #cmap='bwr' 
                #extend="both"
                PlotRaster = 1

                for v in range(row0,rows+row0):
           
                    if v < len(ks1):
                        i = ks1[v]
                        j = ns1[v]
                        alpha = 0
                    else:
                        i = ks3[v-len(ks1)]
                        j = ns3[v-len(ks1)]
                        alpha = 1
 
                    if PlotRaster == 0:
                        idxs = np.where( OmegaArr[v,:,:,0,0] <= OmegaLimit )
                        if len(idxs)>0:
                            counts_i = idxs_triads_x[0,idxs[0][:]]
                            counts_j = idxs_triads_z[0,idxs[1][:]]
                            axs0[v,0].scatter(counts_i,counts_j)

                    if PlotRaster == 1:

                        im = axs0[v-row0,0].pcolormesh(OmegaArr[v,:,:,0,0], cmap=cmap, norm=norm)
                        axs0[v-row0,0].set_xlim(-18,2*Nz)
                        axs0[v-row0,0].set_ylim(-4,Nx)
                        axs0[v-row0,0].set_xticks([0,180])
                        axs0[v-row0,0].set_yticks([0,40])
                        axs0[v-row0,0].tick_params(axis='y', which='major', labelsize=8)
                        ylabel = '(' + str(alphavec[alpha]) + ',' + str(i) + ',' + str(j) + ')'
                        axs0[v-row0,0].set_ylabel(ylabel, fontsize=6, rotation='horizontal', labelpad=20)
                   
                        idxs_min = np.where(logic[v,:,:,0,0]==True)
                        idxs_min_k = idxs_min[0]
                        idxs_min_n = idxs_min[1]
                        axs0[v-row0,0].plot(idxs_min_n,idxs_min_k, linestyle='None',\
                        marker=marker, color=marker_col, mfc=mfc, markersize=s)

                        im = axs0[v-row0,1].pcolormesh(OmegaArr[v,:,:,0,1], cmap=cmap, norm=norm)
                        axs0[v-row0,1].set_xlim(-18,2*Nz)
                        axs0[v-row0,1].set_ylim(-4,Nx)
                        axs0[v-row0,1].set_xticks([0,180])
                        axs0[v-row0,1].set_yticks([0,40])

                        idxs_min = np.where(logic[v,:,:,0,1]==True)
                        idxs_min_k = idxs_min[0]
                        idxs_min_n = idxs_min[1]
                        axs0[v-row0,1].plot(idxs_min_n,idxs_min_k, linestyle='None',\
                        marker=marker, color=marker_col, mfc=mfc, markersize=s)

                        im = axs0[v-row0,2].pcolormesh(OmegaArr[v,:,:,1,0], cmap=cmap, norm=norm)
                        axs0[v-row0,2].set_xlim(-18,2*Nz)
                        axs0[v-row0,2].set_ylim(-4,Nx)
                        axs0[v-row0,2].set_xticks([0,180])
                        axs0[v-row0,2].set_yticks([0,40])

                        idxs_min = np.where(logic[v,:,:,1,0]==True)
                        idxs_min_k = idxs_min[0]
                        idxs_min_n = idxs_min[1]
                        axs0[v-row0,2].plot(idxs_min_n,idxs_min_k, linestyle='None',\
                        marker=marker, color=marker_col, mfc=mfc, markersize=s)

                        im = axs0[v-row0,3].pcolormesh(OmegaArr[v,:,:,1,1], cmap=cmap, norm=norm)
                        axs0[v-row0,3].set_xlim(-18,2*Nz)
                        axs0[v-row0,3].set_ylim(-4,Nx)
                        axs0[v-row0,3].set_xticks([0,180])
                        axs0[v-row0,3].set_yticks([0,40])
                        #plt.colorbar(im1, ax=axs0[v-rows0,3], format='%.3f')                

                        idxs_min = np.where(logic[v,:,:,1,1]==True)
                        idxs_min_k = idxs_min[0]
                        idxs_min_n = idxs_min[1]
                        axs0[v-row0,3].plot(idxs_min_n,idxs_min_k, linestyle='None',\
                        marker=marker, color=marker_col, mfc=mfc, markersize=s)


                cbar = fig0.colorbar(im, ax=axs0.ravel().tolist(), shrink=0.95, label=r'$\Omega$ (rad/s)')
                axs0[0,0].set_title('(-1, -1)')
                axs0[0,1].set_title('(-1, +1)')
                axs0[0,2].set_title('(+1, -1)')
                axs0[0,3].set_title('(+1, +1)')

                plt.show()



pdb.set_trace()
