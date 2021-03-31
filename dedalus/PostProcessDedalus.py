### Code to post process Dedalus results for EPSRC project ###
#Authors: Paul Burns, Beth Wingate
#Date: 2019
#Email: p.burns2@exeter.ac.uk, b.wingate@exeter.ac.uk


#Load in required libraries:
import h5py
import numpy as np
from numpy import *
from scipy import *
from numpy import fft
from scipy import fftpack
from scipy.signal import welch
from scipy.signal.windows import tukey
import pdb #to pause execution use pdb.set_trace()
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dedalus import public as de
import sys
from netCDF4 import Dataset
import netCDF4 as nc4


#Passed variables:
#(First argument is plot time, second is RunName. 
#If you need to pass the RunName and don't need the plot time, you still must 
#pass the plot time in as a dummy variable even though you won't use it).
if len(sys.argv) > 1:
    tIdx = int(sys.argv[1])
if len(sys.argv) > 2:
    RunName = str(sys.argv[2])
    #Extract N2 value of file from name:
    N2_array_MJ = [0.09, 0.25, 1, 2.25, 4, 6.25, 7.5625, 9, 10.5625, 12.25, 14.0625, 16, 20.25, 25]
    tmp = RunName.split('_')
    N2 = N2_array_MJ[int(tmp[1])]



#Program control:
Gusto		= 0
Modulated       = 0
Linear 		= 0
Inviscid	= 0
FullDomain      = 1
SinglePoint	= 0
MultiPoint	= 0
ProblemType 	= 'Layers'
#ProblemType 	= 'KelvinHelmholtz'
VaryN           = 1
#ParkRun 	= 14
#ParkRun 	= 18
ParkRun 	= -1
scalePert	= 0
forced          = 0
if VaryN == 1:
    #N2		= 0.09
    N2		= 0.25
    #N2 	= 1
    N2		= 2.25		
    #N2 	= 4
    #N2		= 6.25
    #N2		= 7.5625
    #N2 	= 9
    #N2		= 10.5625
    #N2 	= 12.25
    #N2		= 14.0625
    #N2		= 16
    #N2		= 20.25
    #N2		= 25


#User must make sure correct data is read in for some analysis:
#var_nms = ['psi']
var_nms = ['S']
#var_nms = ['psi','S']
#var_nms = ['psi','S','psi_r','S_r']
#var_nms = ['psi_r','S_r']
#var_nms = ['S','S_r']
#var_nms = ['PE_tot','PE_L','KE_tot']
#var_nms = ['PE_L','PE_adv','PE_N','PE_diff','KE_b','KE_p','KE_adv','KE_diff','KE_x','KE_z','psi','S']
#var_nms = ['PE_L','PE_N','PE_diff','KE_b','KE_p','KE_diff','KE_x','KE_z','S','psi']
#var_nms = ['PE_L','PE_N','KE_b','KE_p','KE_x','KE_z','S','psi']
#var_nms = ['p_check']
Nvars = len(var_nms)

#Choose which diagnostics to compute:
#n.b. code has been written to make each switch/variable 
#largely independent of the others. This makes it easier for the
#user and helped to make the code more object orientated/modular to 
#minimise repetition.
FullFields              = 1
StatePsi                = 0
StateS                  = 0
Density			= 1
StateS_2                = 0
PlotStairStartEnd	= 0
Flow                    = 0
dSdz                    = 0
drhodz			= 1
TrackSteps              = 0
TrackInterfaces         = 0
Fluxes			= 0
UseProxySz 		= 0
dUdz                    = 0
Richardson              = 0
Vorticity               = 0
KineticE                = 0
PotentialE              = 0
Energy             	= 0
read_energy		= 0
check_p             	= 0
ForwardTransform     	= 0
CoefficientSpace	= 0

SpectralAnalysis        = 0
AnalyseS                = 1
MeanFlowAnalysis	= 0
PlotBigMode		= 0
CheckPSD		= 0
CheckPSD2		= 0
PSD_vs_N_plot		= 0
PSD_mod_unmod_plot	= 0

TimescaleSeparation	= 0
OverlayModulated	= 1
IGWmethod 		= 0
step_prediction		= 0

NaturalBasis            = 0
nvars	            	= 2
BasisCheck1             = 0
BasisCheck2             = 0
MakeCoordRotation       = 0
TestModulation		= 0

#General statistical processing:
xMean = 0
tMean = 0

tMean_slide = 0
#Choose sliding-window length for averaging:
#N.B. some data will be lost from start and end of time period.
#N.B. Choose an odd length to make window symmetric around some t point:
#Nt_mean = 1801
Nt_mean = 21
wing = Nt_mean//2

#Choose type of plot:
MakePlot 	= 1
PlotXZ 		= 0
PlotTZ 		= 1
PlotT 		= 0
PlotZ 		= 0
MakeMovie 	= 0
filledContour 	= 1
NoPlotLabels    = 0

#Write analysis to file
w2f_analysis = 0


#Setup parameters for reading Dedalus data into this program:
if VaryN == 0:
    #Options when reading data:  
    #dir_state = '/gpfs/ts0/projects/Research_Project-183035/ForcedResults/' + 'State' + RunName + '/'
    dir_state = '/gpfs/ts0/projects/Research_Project-183035/tmp/' + 'State' + RunName + '/'
    #dir_state = '/gpfs/ts0/home/pb412/dedalus/Results/' + 'State' + RunName + '/'

    #dir_state = '/home/ubuntu/dedalus/Results/State/'
    #dir_state = '/gpfs/ts0/projects/Research_Project-183035/Results/StateN2_03_83_forced01/'
    #dir_state = '/gpfs/ts0/projects/Research_Project-183035/Results/StateN2_03_83_forced02/'
    #dir_state = '/gpfs/ts0/projects/Research_Project-183035/Results/StateN2_03_83_forced03/'
    #dir_state = '/gpfs/ts0/projects/Research_Project-183035/Results/StateN2_03_83_forced04/'
    #dir_state = '/gpfs/ts0/projects/Research_Project-183035/Results/StateN2_03_83_forced05/'
    #dir_state = '/gpfs/ts0/projects/Research_Project-183035/Results/StateN2_03_83_forced06/'
    #dir_state = '/gpfs/ts0/projects/Research_Project-183035/Results/StateN2_03_83_forced08/'
    #dir_state = '/gpfs/ts0/projects/Research_Project-183035/Results/StateN2_03_83_forced09/'
    #dir_state = '/gpfs/ts0/projects/Research_Project-183035/Results/StateN2_03_83_forced10/'
    #dir_state = '/gpfs/ts0/projects/Research_Project-183035/Results/StateN2_03_83_forced11/'
    #dir_state = './Results/State/'
    #dir_state = './Results/StateN2_02_25_lnr/'
    #dir_state = './Results/StateN2_02_25_lnr_invis/'
    #dir_state = './Results/State_mesh0/'
    #dir_state = './Results/State_mesh1/'
    #dir_state = './Results/State_mesh2/'
    #dir_state = './Results/State_mesh3/'
    #dir_state = './Results/State_mesh4/'

if VaryN == 1:
    if N2 == 0.09: 	RunName = 'StateN2_00_09'
    if N2 == 0.25: 	RunName = 'StateN2_00_25'
    if N2 == 1: 	RunName = 'StateN2_01'
    if N2 == 2.25: 	RunName = 'StateN2_02_25'
    if N2 == 4:		RunName = 'StateN2_04'
    if N2 == 6.25:	RunName = 'StateN2_06_25'
    if N2 == 7.5625:	RunName = 'StateN2_07_5625'
    if N2 == 9:		RunName = 'StateN2_09'
    if N2 == 10.5625:	RunName = 'StateN2_10_5625'
    if N2 == 12.25:	RunName = 'StateN2_12_25'
    if N2 == 14.0625:	RunName = 'StateN2_14_0625'
    if N2 == 16:	RunName = 'StateN2_16'
    if N2 == 20.25:	RunName = 'StateN2_20_25'
    if N2 == 25:	RunName = 'StateN2_25'
    if forced==1 or SinglePoint==1 or Modulated==1:
        if forced == 1:
            RunName = RunName + '_k05n028x10'
            #RunName = RunName + '_k05n028x2'
            #RunName = RunName + '_k05n014x5'
            #RunName = RunName + '_k05n014'
        if SinglePoint == 1:
            RunName = RunName + '_dt0.01_sp'
            #RunName = RunName + '_dt0.005_sp'
        if Modulated == 1 and forced == 0 and SinglePoint == 0:
            RunName = RunName + '_R'
    #dir_state = './Results/' + RunName + '/'
    dir_state = '/home/ubuntu/dedalus/Results/' + RunName + '/'


if Gusto == 0:
    #Each Dedalus output file contains 1 min of data - this is assumed constant:
    secPerFile = 60.

    if SpectralAnalysis==1 and MeanFlowAnalysis==0:
        StartMin = 1
        nfiles = 10
        nfiles = 29
    elif (SpectralAnalysis==1 and MeanFlowAnalysis==1) or (SpectralAnalysis==1 and CheckPSD2==1):
        StartMin = 1
        nfiles = 30
    else:
        StartMin = 1
        nfiles = 2

    #Model output/write timestep:
    if FullDomain == 1: dt = 1e-1
    if SinglePoint==1 or MultiPoint==1: 
        #dt = 1e-2
        dt = 0.008
        #dt = 5e-3
        #dt = 1e-3

if Gusto == 1: dt = 0.004



#Analysis timestep:
#(This is important for computations involving numerous large arrays - 
#it avoids exceeding system memory limits).
#Effectively we use a subset of the model output data for the analysis:
if SpectralAnalysis==1 and MeanFlowAnalysis==0 and CheckPSD2==0: 
    if forced == 0: dt2=0.2
    if forced == 1: 
        dt2=0.2
        dt2=dt
elif SpectralAnalysis==1 and MeanFlowAnalysis==1 and CheckPSD2==0: dt2=1.
elif SpectralAnalysis==1 and CheckPSD2==1: dt2=dt
else:
    dt2 = dt
    dt2 = 0.1
    #dt2 = 0.2
    #dt2 = 0.4
    #dt2 = 0.02
    #dt2 = 0.04
    #dt2 = 0.08
    #dt2 = 0.16
    #dt2 = 1.
    #dt2 = 2.
    #dt2 = 5

#Set up grid and related objects using Dedalus virtual environment:
#System's physical dimensions:
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

#grid axes and lengths:
if Gusto == 0:
    x_basis = de.Fourier('x', Nx, interval=(0, Lx), dealias=3/2)
    z_basis = de.SinCos('z', Nz, interval=(0, Lz), dealias=3/2)
    domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
    x = domain.grid(0)[:,0]
    z = domain.grid(1)[0,:]
    dx = x[1]-x[0]
    dz = z[1]-z[0]
if Gusto == 1:
    points = np.loadtxt('/home/ubuntu/firedrake/src/gusto/points.txt')
    x = points[:,0]
    z = points[:,1]
    dz = points[2,1]-points[1,1]
    dx = dz

#Create time axis for plotting:
if Gusto == 0:
    ntPerFile = secPerFile/dt
    tq = dt2/dt
    Nt = ntPerFile*nfiles/tq
    Nt = int(Nt)
    t = np.arange(Nt)*dt2 + (StartMin-1)*secPerFile
if Gusto == 1:
    te = 2*60.
    tq = dt2/dt
    Nt = te/dt/tq 
    Nt = int(Nt)
    t = np.arange(Nt)*dt2
    

#Construct some general arrays for contour plots:
if FullDomain == 1:
    x2d = np.tile(x,(Nz,1))
    z2d = np.tile(z,(Nx,1)).transpose()
    t2d_z = np.tile(t,(Nz,1))
    z2d_t = np.tile(z,(Nt,1)).transpose()

#Provide program with info. about chosen point(s):
if SinglePoint == 1:
    Nx2 = 1
    Nz2 = 1
    xIdx = int(Nx/2.)
    zIdx = int(Nz/2.)

if MultiPoint == 1:
    Nx2 = 3
    Nz2 = 3
    xIdx = (np.array([1./4,1./2,3./4])*Nx).astype('int')
    zIdx = (np.array([1./4,1./2,3./4])*Nz).astype('int')
    #print(xIdx,zIdx)

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

if FullFields == 1 and Gusto == 0:
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


#Set general plotting parameters assuming A4 page size:
A4Width = 8.27
MarginWidth = 1
height = 5
width = A4Width-2*MarginWidth
#width = height*1.2
#For scaling the A4 plot dimensions:
ScaleFactor = 1
#ScaleFactor = 0.7
width = width*ScaleFactor
height = height*ScaleFactor


if Gusto == 0:
    #Read in Dedalus results:
    fileIdx = np.arange(nfiles) 

    if FullDomain==1:
        if 'psi' in var_nms: Psi = np.zeros((Nt,Nx,Nz))
        if 'T' in var_nms: T = np.zeros((Nt,Nx,Nz))
        if 'S' in var_nms: S = np.zeros((Nt,Nx,Nz))
        if 'S_r' in var_nms and CoefficientSpace == 0: S_r = np.zeros((Nt,Nx,Nz))
        if 'S_r' in var_nms and CoefficientSpace == 1: S_r = np.zeros((Nt,int(Nx/2.),Nz))
        if 'psi_r' in var_nms and CoefficientSpace == 0: Psi_r = np.zeros((Nt,Nx,Nz))
        if 'psi_r' in var_nms and CoefficientSpace == 1: Psi_r = np.zeros((Nt,int(Nx/2.),Nz))
        #Energy variables:
        if 'PE_tot' in var_nms: PE_tot = np.zeros((Nt,1,1))
        if 'PE_L' in var_nms: PE_L = np.zeros((Nt,1,1))
        if 'KE_tot' in var_nms: KE_tot = np.zeros((Nt,1,1))
    if SinglePoint==1 or MultiPoint==1:
        if 'psi' in var_nms: Psi = np.zeros((Nt,Nx2,Nz2))
        if 'T' in var_nms: T = np.zeros((Nt,Nx2,Nz2))
        if 'S' in var_nms: S = np.zeros((Nt,Nx2,Nz2))
        if 'S_r' in var_nms: S_r = np.zeros((Nt,Nx2,Nz2))
        if 'psi_r' in var_nms: Psi_r = np.zeros((Nt,Nx2,Nz2))

    for jj in range(0,Nvars):
        for ii in fileIdx:
            if len(sys.argv) > 2: 
                fnm = dir_state + 'State' + RunName + '_s' + str(ii+StartMin) + '.h5'
            else:
                fnm = dir_state + 'State_s' + str(ii+StartMin) + '.h5'
            hdf5obj = h5py.File(fnm,'r')

            #with h5py.File(fnm,'r') as hdf5obj:
            #    base_items = list(hdf5obj.items())
            #    print(base_items)
            #    g1 = hdf5obj.get('tasks')
            #    g1_items = list(g1.items())
            #    print(g1_items)
            #pdb.set_trace()

            tmp_ = hdf5obj.get('tasks/'+var_nms[jj])
            idxS = ii*ntPerFile/tq
            idxE = (ii+1)*ntPerFile/tq
            idxS = int(idxS)
            idxE = int(idxE)
            print(idxS,idxE)

            #Prognostic variables:
            if var_nms[jj] == 'psi':
                if FullDomain==1 or MultiPoint==1: Psi[idxS:idxE,:,:] = np.array(tmp_)[::int(tq),:,:]
                if SinglePoint==1: Psi[idxS:idxE] = np.array(tmp_)[::int(tq)]
            if var_nms[jj] == 'S':
                if FullDomain==1 or MultiPoint==1: S[idxS:idxE,:,:] = np.array(tmp_)[::int(tq),:,:]
                if SinglePoint==1: S[idxS:idxE] = np.array(tmp_)[::int(tq)]
            #Modulated fields:
            if var_nms[jj] == 'S_r':
                if FullDomain==1 or MultiPoint==1: S_r[idxS:idxE,:,:] = np.array(tmp_)[::int(tq),:,:]
                if SinglePoint: S_r[idxS:idxE] = np.array(tmp_)[::int(tq)]
            if var_nms[jj] == 'psi_r':
                if FullDomain==1 or MultiPoint==1: Psi_r[idxS:idxE,:,:] = np.array(tmp_)[::int(tq),:,:]
                if SinglePoint==1: Psi_r[idxS:idxE] = np.array(tmp_)[::int(tq)]

            #Energy variables:
            if var_nms[jj] == 'PE_tot':
                PE_tot[idxS:idxE] = np.array(tmp_)[::int(tq)]
            if var_nms[jj] == 'PE_L':
                PE_L[idxS:idxE] = np.array(tmp_)[::int(tq)]
            if var_nms[jj] == 'KE_tot':
                KE_tot[idxS:idxE] = np.array(tmp_)[::int(tq)]
            if var_nms[jj] == 'KE_x':
                KE_x[idxS:idxE] = np.array(tmp_)[::int(tq)]
            if var_nms[jj] == 'KE_z':
                KE_z[idxS:idxE] = np.array(tmp_)[::int(tq)]

            #Check initial pressure:
            if var_nms[jj] == 'p_check':
                p_check[idxS:idxE,:,:] = np.array(tmp_)[::int(tq),:,:]

            hdf5obj.close()

if Gusto == 1:
    #Read in Gusto output netCDF data:
    dataPath = "/home/ubuntu/firedrake/src/gusto/results/" + RunName + "_gusto" + "/point_data.nc"
    f = nc4.Dataset(dataPath, 'r')
    bgrp = f.groups['b_gradient']
    b_gradients = bgrp.variables['b_gradient'][0:int(te/dt):int(tq),:,:]
    #print(ts_masked.shape)
    dbdz = b_gradients[:,:,1]

    #Transform buoyancy into salinity:
    data_zt = -1./(g*cs)*dbdz
    #Code assumes a full 2D grid of spatial points, so simply add the vertical profile 
    #to a 3D space-time numpy array:
    data = np.zeros((Nt,Nx,Nz))
    xIdx = int(Nx/2.)
    data[:,xIdx,:] = data_zt


#Define some useful functions:
def x_mean(f):
    data = np.mean(f,1)
    return data

def t_mean(f):
    #Average across full time period:
    data = np.mean(f,0)
    return data

def t_mean_slide(f,Nt,Nx,Nz,wing):
    data = np.zeros((Nt,Nx,Nz))
    for tt in range(wing,Nt-wing):
        data[tt,:,:] = np.mean(f[tt-wing:tt+wing,:,:],0)
    return data

def d_dz(f,Nt,Nx,Nz,z):
    fz = np.zeros((Nt,Nx,Nz))
    for jj in range(0,Nz):
        for ii in range(0,Nx):
            #Use centered scheme except next to boundaries:
            if (jj != 0) and (jj != Nz-1): df = f[:,ii,jj+1] - f[:,ii,jj-1]
            if jj == 0: df = f[:,ii,jj+1] - f[:,ii,jj]
            if jj == Nz-1: df = f[:,ii,jj] - f[:,ii,jj-1]

            if (jj != 0) and (jj != Nz-1): fz[:,ii,jj] = df/(2*dz)
            else: fz[:,ii,jj] = df/dz
    return fz

def d_dx(f,Nt,Nx,Nz,x):
    fx = np.zeros((Nt,Nx,Nz))
    for jj in range(0,Nz):
        for ii in range(0,Nx):
            #Use centered scheme except next to boundaries:
            if (ii != 0) and (ii != Nx-1): df = f[:,ii+1,jj] - f[:,ii-1,jj]
            if ii == 0: df = f[:,ii+1,jj] - f[:,ii,jj]
            if ii == Nx-1: df = f[:,ii,jj] - f[:,ii-1,jj]

            if (ii != 0) and (ii != Nx-1): fx[:,ii,jj] = df/(2*dx)
            else: fx[:,ii,jj] = df/dx
    return fx

def d_dt(f,Nt,t):
    if len(f.shape) > 1: ft = np.zeros((Nt,Nx,Nz))
    else: ft = np.zeros((Nt))
    for tt in range(0,Nt):
        #Use centered scheme except next to boundaries:
        if len(f.shape) > 1:
            if (tt != 0) and (tt != Nt-1): df = f[tt+1,:,:] - f[tt-1,:,:]
            if tt == 0: df = f[tt+1,:,:] - f[tt,:,:]
            if tt == Nt-1: df = f[tt,:,:] - f[tt-1,:,:]

            if (tt != 0) and (tt != Nt-1): ft[tt,:,:] = df/(2*dt2)
            else: ft[tt,:,:] = df/dt2
        else:
            if (tt != 0) and (tt != Nt-1): df = f[tt+1] - f[tt-1]
            if tt == 0: df = f[tt+1] - f[tt]
            if tt == Nt-1: df = f[tt] - f[tt-1]

            if (tt != 0) and (tt != Nt-1): ft[tt] = df/(2*dt2)
            else: ft[tt] = df/dt2
    return ft


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
    f0 = 0.1					#Lowest frequency when finding dominant mode.

    if Welch == 0:
        #repeat non-periodic finite signal Np times:
        Np = 1
        signal_L = Nt*Np
        spectralCoef = np.zeros((int(signal_L/2.)+1,Nx,Nz))
        freqvec = np.arange(signal_L/2.+1)*1./signal_L      	#assumes dt=1
                                                		#Note array length is Nt/2 and so max freq. will be Nyquist freq.
        freqvec         = freqvec*signal_f      		#uses actual signal frequency (Dedalus timestep)
        #freqvec        = freqvec*2*np.pi       		#converts to angular frequency (rad/s) - but less intuitive

    if Welch == 1:
        #Description of method and it's application:
        #Essentially this method divides signal into a number of segments that are nperseg points long. 
        #A form of spectral analysis (periodogram) is performed on each segment after a Hann window is applied.
        #Then an average periodogram is computed from the set of periodograms.
        #To reduce variance (noise) the segments overlap and the default overlap is nperseg/2.
        #So if nperseg is longer then we have higher frequency resolution but possibly more variance/noise.

        #Here we vary the number of points in the periodograms with the timestep. This avoids a set
        #number of points being distributed over different bandwidths, which results in 
        #relatively low frequency resolution at low frequencies for small dt, making the results less comparable.
        #Of course we need to always apply our method to the physical problem - we need to compute 
        #the spectrum for bandwiths that include the fast waves of our system, whilst using a practical 
        #method with the data we have (i.e. the number of points is finite, affecting window length choice).         
        #Additionally we should consider that the variance reduction will be greater when using more
        #windows, that is, for smaller dt. 

        #Try keeping number of points in PSD constant and show frequency resolution problems:
        #nperseg = 2000

        #Vary number of points with timestep to make results more comparable (see above):
        #nperseg = (nfiles/3.*60)/dt2	#(tested for 15 min of data - possibly adjust otherwise).
        #nwindows = Nt/(nperseg*0.5)+1
        if MeanFlowAnalysis == 1: nwindows = 1
        else: nwindows = 7
        if nwindows != 1: nperseg = int(2*Nt/(nwindows-1))
        else: nperseg = Nt
        print(Nt,nperseg,nwindows)

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
                #BigMode[ii,jj] = freqvec[idx[0] + fIdx]

    return spectralCoef, freqvec, nperseg;


if FullFields == 1 and Gusto == 0:
    #add base state:
    if Modulated == 0: S += Sbase
    else: S_r += Sbase

    #print(np.min(Sbase), np.max(Sbase))
    #pdb.set_trace()


if Density == 1:
    #calculate density using a linear equation of state where rho=rho(S):
    if Modulated == 0:
        S0 = np.mean(S)
        rho = rho0*(1 + cs*(S-S0))
        #print(np.min(rho),np.max(rho))
    if Modulated == 1:
        S0 = np.mean(S_r)
        rho_r = rho0*(1 + cs*(S_r-S0))
        #print(np.min(rho_r),np.max(rho_r))

    #pdb.set_trace()


if Flow == 1:

    if Modulated == 0:
        data = d_dz(Psi,Nt,Nx,Nz,z)
        data2 = -d_dx(Psi,Nt,Nx,Nz,x)
    else:
        data = d_dz(Psi_r,Nt,Nx,Nz,z)
        data2 = -d_dx(Psi_r,Nt,Nx,Nz,x)

    if MakePlot == 1:
        PlotTitle = r'$u$'
        PlotTitle2 = r'$w$'
        FigNmBase = 'Flow'
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
                
        #xlim=(10,30)


if Vorticity == 1:
    Psi_x = d_dx(Psi,Nt,Nx,Nz,x)
    Psi_z = d_dz(Psi,Nt,Nx,Nz,z)
    Psi_xx = d_dx(Psi_x,Nt,Nx,Nz,x)
    Psi_zz = d_dz(Psi_z,Nt,Nx,Nz,z)
    data = -(Psi_xx + Psi_zz)

    if MakePlot == 1:
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


if KineticE == 1:
    KEplot = 1

    if read_energy == 1:

        if Linear == 0 and Inviscid == 0: data = KE_p + KE_b + KE_adv + KE_diff
        if Linear == 1 and Inviscid == 0: data = KE_p + KE_b + KE_diff
        if Linear == 0 and Inviscid == 1: data = KE_p + KE_b + KE_adv
        if Linear == 1 and Inviscid == 1: data = KE_p + KE_b
        #data2 = KE_x + KE_z

        compute_lhs = 1
        if compute_lhs == 1:
            u = d_dz(Psi,Nt,Nx,Nz,z)
            w = -d_dx(Psi,Nt,Nx,Nz,x)
            KE_ = 1./2*( u**2 + w**2 )

            #Sum over domain to find conserved quantity:
            KE_tot = np.zeros((Nt))
            for tt in range(0,Nt):
                KE_tot[tt] = np.trapz( np.trapz( KE_[tt,:,:], axis=0, dx=dx), axis=0, dx=dz)
            dKE_dt = d_dt(KE_tot,Nt,t)

        if KEplot == 1:
            fig=plt.figure(figsize=(width,height))
            grid = plt.GridSpec(1, 1, wspace=0.0, hspace=0.0)
            ax1 = fig.add_subplot(grid[0,0])

            ax1.set_xlim(0,10)
            ax1.set_xlabel(r'$t$ (s)')
            ax1.set_ylabel(r'$E$ (m$^2$/s$^3$)')

            ax1.plot(t,KE_p.flatten(), 'grey', linestyle='-', linewidth=1, label=r'$KE_p$')
            ax1.plot(t,KE_b.flatten(), 'grey', linestyle='-', linewidth=2, label=r'$KE_b$')
            if Linear == 0:
                ax1.plot(t,KE_adv.flatten(), 'c', linestyle='-', label=r'$KE_{adv}$')
            if Inviscid == 0:
                ax1.plot(t,KE_diff.flatten(), 'grey', linestyle=':', linewidth=2, label=r'$KE_d$')
            ax1.plot(t,data.flatten(), 'k', linestyle='-', linewidth=3, label=r'$KE$')
            #ax1.plot(t,data2.flatten(),'.k')
            ax1.plot(t,KE_x.flatten(), 'k', linestyle='-', linewidth=2, label=r'$KE_x$')
            ax1.plot(t,KE_z.flatten(), 'k', linestyle='--', linewidth=2, label=r'$KE_z$')
            #ax1.plot(t,KE_x.flatten()/KE_z.flatten(), 'k', linestyle='-.', linewidth=2, label=r'$KE_x/KE_z$')
            ax1.plot(t,dKE_dt.flatten(), 'ok')

            ax1.legend(ncol=1, frameon=False)
            plt.show()   

    if MakePlot == 1:
        PlotTitle = ''
        #ylim = (-10,10)


if PotentialE == 1:
    PEplot = 1

    if read_energy == 1:

        #reverse signs: 
        #PE_L = -PE_L
        #PE_N = -PE_N
        #if Linear == 0:
        #    PE_adv = -PE_adv
        #    PE_N = -PE_N
        #if Inviscid == 0:
        #    PE_diff = -PE_diff

        if Linear == 0 and Inviscid == 0: data = PE_L + PE_N + PE_adv + PE_diff
        if Linear == 0 and Inviscid == 1: data = PE_L + PE_N + PE_adv 
        if Linear == 1 and Inviscid == 0: data = PE_L + PE_N + PE_diff 
        if Linear == 1 and Inviscid == 1: data = PE_L + PE_N 

        compute_lhs = 1
        if compute_lhs == 1:
            PE_ = np.zeros((Nt,Nx,Nz))
            for jj in range(0,Nz):
                PE_[:,:,jj] = g*cs*S[:,:,jj]*z[jj]

            #Sum over domain to find conserved quantity:
            PE_tot = np.zeros((Nt))
            for tt in range(0,Nt):
                PE_tot[tt] = np.trapz( np.trapz(PE_[tt,:,:], axis=0, dx=dx), axis=0, dx=dz)
            dPE_dt = d_dt(PE_tot,Nt,t)

        if PEplot == 1:
            fig=plt.figure(figsize=(width,height))
            grid = plt.GridSpec(1, 1, wspace=0.0, hspace=0.0)
            ax1 = fig.add_subplot(grid[0,0])

            ax1.set_xlim(0,10)
            ax1.set_xlabel(r'$t$ (s)')
            ax1.set_ylabel(r'$E$ (m$^2$/s$^3$)')

            ax1.plot(t,PE_L.flatten(), 'k', linestyle='-', linewidth=1, label=r'$PE_L$')
            ax1.plot(t,PE_N.flatten(), 'k', linestyle='--', linewidth=2, label=r'$PE_N$')
            if Linear == 0:
                ax1.plot(t,PE_adv.flatten(), 'k', linestyle='-', linewidth=2, label=r'$PE_{adv}$')
                ax1.plot(t,PE_N.flatten(), 'k', linestyle='--', linewidth=2, label=r'$PE_N$')
            if Inviscid == 0:
                ax1.plot(t,PE_diff.flatten(), 'k', linestyle=':', linewidth=2, label=r'$PE_d$')
            ax1.plot(t,data.flatten(), 'k', linestyle='-', linewidth=3, label=r'$PE$')
            ax1.plot(t,dPE_dt.flatten(), 'ok')

            ax1.legend(ncol=1, frameon=False)
            plt.show()

    if MakePlot == 1:
        PlotTitle = ''
        ylim = (-20,20)


if Energy == 1:
    Eplot = 1

    if read_energy == 1:

        PE_tot = PE_tot
        KE_tot = KE_tot
        PE_tot = PE_tot + 1./6*cs*bs*g*Lx*Lz**3
        data = KE_tot + PE_tot

        if Eplot == 1:
            fig=plt.figure(figsize=(width,height))
            grid = plt.GridSpec(1, 1, wspace=0.0, hspace=0.0)
            ax1 = fig.add_subplot(grid[0,0])

            ax1.set_xlim(0,np.max(t))
            ax1.set_xlabel(r'$t$ (s)')
            ax1.set_ylabel(r'$E$ (m$^2$/s$^3$)')

            ax1.plot(t,data.flatten(), 'k', linestyle='-', linewidth=3, label=r'$E$')
            ax1.plot(t,PE_tot.flatten(), 'grey', linestyle='-', linewidth=2, label=r'$PE$')
            ax1.plot(t,KE_tot.flatten(), 'k', linestyle='-', linewidth=2, label=r'$KE$')
            ax1.plot(t,PE_L.flatten(), 'grey', linestyle=':', linewidth=3, label=r'$PE_L$')

            ax1.legend(ncol=2, frameon=False)
            plt.show()


    if read_energy == 0:
        #n.b. this relies on KE and PE sections above in contrast to rest of code.
        data = KE_lhs_tot + PE_lhs_tot
        Eplot = 1
        if Eplot == 1:
            plt.plot(t,KE_rhs_tot,'.k-')
            plt.plot(t,PE_rhs_tot,'r')
            plt.plot(t,data,'g')
            plt.show()

    if MakePlot == 1:
        PlotTitle = ''
        #ylim = (-20,20)


if check_p == 1:
    data = p_check

    if MakePlot == 1:
        PlotTitle = ''
        clevels=50
        cmap='binary'
        #ylim = (-10,10)


if dSdz == 1:
    
    if Gusto == 0: 
        if Modulated == 0: data = d_dz(S,Nt,Nx,Nz,z)
        else: data = d_dz(S_r,Nt,Nx,Nz,z)

    if MakePlot == 1:
        if NoPlotLabels == 0: 
            PlotTitle = r'$\partial S/\partial z$ (g/kg/m)' + ', ' + RunName
        else: PlotTitle = ''
        FigNmBase = 'dSdz'

        if filledContour == 1: 
            #nlevs = 21
            nlevs = 41
            #nlevs = 81
        else: nlevs = 41

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
                #SzMin = -300
                #SzMax = 300
            if N2==3.83 or N2==4:
                if Modulated == 0:
                    SzMin = -1000
                    SzMax = 1000
                if Modulated == 1:
                    SzMin = np.min(data)
                    SzMax = np.max(data)
            if N2==6.25 or N2==7.5625:
                SzMin = -1500
                SzMax = 1500
            if N2==9 or N2==10.5625:
                if Modulated == 0:
                    SzMin = -2000
                    SzMax = 2000
                else:
                    SzMin = np.min(data)
                    SzMax = np.max(data)
            if N2==12.25 or N2==14.0625:
                SzMin = -2500
                SzMax = 2500
            if N2==16:
                SzMin = -3000
                SzMax = 3000
            if N2==20.25:
                SzMin = -4000
                SzMax = 4000
            if N2==25:
                #dS = (Sbase[1]-Spert0/2) - (Sbase[0]+Spert0/2)
                #dz = Lz/(Nz-1)
                #SzMin = round(dS/dz,-2)
                #SzMax = -SzMin
                SzMin = -5000
                SzMax = 5000
        dSz = (SzMax - SzMin)/(nlevs-1)
        clevels = np.arange(nlevs)*dSz + SzMin
        if filledContour == 1: 
            #cmap = 'PRGn'
            cmap = 'bwr'
            #cmap = 'tab20b'
        else:
            col1 = ['k']*(int(nlevs/2.-1))
            col2 = ['grey']*(int(nlevs/2.))
            colorvec = col1+col2
        #xlim = (0,np.max(t))
        xlim = (0,60)
        xlim = ((StartMin-1)*secPerFile,np.max(t))

if drhodz == 1:

    if Gusto == 0:
        if Modulated == 0: data = d_dz(rho,Nt,Nx,Nz,z)
        else: data = d_dz(rho_r,Nt,Nx,Nz,z)

    if MakePlot == 1:
        if NoPlotLabels == 0:
            PlotTitle = r'$\partial\rho/\partial z$ (kg m$^{-4}$)'
        else: PlotTitle = ''
        FigNmBase = 'drhodz'

        if filledContour == 1:
            #nlevs = 21
            nlevs = 41
            #nlevs = 81
        else: nlevs = 41

        if ParkRun < 0:
            if N2==0.1 or N2==0.25 or N2==0.09:
                rhozMin = -100
                rhozMax = 100
            if N2==1:
                rhozMin = -300
                rhozMax = 300
            if N2==2.25:
                rhozMin = -500
                rhozMax = 500
                #rhozMin = -300
                #rhozMax = 300
        drhoz = (rhozMax - rhozMin)/(nlevs-1)
        clevels = np.arange(nlevs)*drhoz + rhozMin
        if filledContour == 1:
            #cmap = 'PRGn'
            cmap = 'bwr'
            #cmap = 'tab20b'
        else:
            col1 = ['k']*(int(nlevs/2.-1))
            col2 = ['grey']*(int(nlevs/2.))
            colorvec = col1+col2
        #xlim = (0,np.max(t))
        #xlim = ((StartMin-1)*secPerFile,np.max(t))
        xlim = (0,60)


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


if TrackSteps == 1:

    UseShear = 0

    #Choose x point for analysis.
    #When averaging I simply store the average at this index to 
    #simplify the code:
    xIdx = int(Nx/2.)    
    #xIdx = int(Nx/4.)    

    if UseShear == 1:
        u = d_dz(Psi,Nt,Nx,Nz,z)
        uz = d_dz(u,Nt,Nx,Nz,z)
        if xMean == 1:
            tmp = x_mean(uz)
            data = np.zeros((Nt,Nx,Nz))
            data[:,xIdx,:] = tmp
        else:
            data = uz
    
    if UseShear == 0: 
        if Gusto == 0: Sz = d_dz(S,Nt,Nx,Nz,z)

        if xMean == 1:
            tmp = x_mean(Sz)
            data = np.zeros((Nt,Nx,Nz))
            data[:,xIdx,:] = tmp

        if xMean == 0: 
            if Gusto == 0:
                if FullFields == 1: data = Sz
                if FullFields == 0: data = S
            if Gusto == 1: Sz = data

    tmp1 = np.zeros((Nt,Nx,Nz),dtype=bool)
    tmp2 = np.zeros((Nt))
    tmp3 = np.zeros((Nt,50))
    tmp4 = np.zeros((Nt,50))
    tmp5 = np.zeros((Nt,50))
    if forced == 1:
        tmp6 = np.zeros((Nt))		#to compute d_dt(number of steps)
        tmp7 = np.zeros((Nt,50))	#to compute d_dt(step depth)
        tmp8 = np.zeros((Nt,50))	#to compute d_dt(mid step point)

    if forced == 0:
        #Exclude boundary layer effects:
        if (N2 == 0.09) or (N2 == 0.25): zIdx_offset = int(.1/dz)
        if (N2 != 0.09) and (N2 != 0.25): zIdx_offset = int(.05/dz)
        if UseShear == 1: zIdx_offset = 0
    if forced == 1: zIdx_offset = 0

    #Automatically exclude initial chaos (depends on N2):
    if FullFields == 1 and forced == 0:
        i = 0
        flag0=1
        while flag0 == 1:
            logical = Sz[i,:,0+zIdx_offset:Nz-zIdx_offset] > 0
            if np.max(logical)==0:
                tIdx_offset = i
                offset_t = t[i]
                flag0 = 0
            i += 1
        print("offset time: ", offset_t)

    if FullFields == 0 and forced == 0:
        if N2 == 0.25:		offset_t = 19.
        if N2 == 1:		offset_t = 8.8
        if N2 == 2.25:		offset_t = 6.7
        if N2 == 4:		offset_t = 4.1
        if N2 == 6.25:		offset_t = 2.9
        if N2 == 7.5625:	offset_t = 2.6
        if N2 == 9:		offset_t = 2.4
        if N2 == 10.5625:	offset_t = 2.2
        if N2 == 12.25:		offset_t = 1.9
        if N2 == 14.0625:	offset_t = 1.7
        if N2 == 16:		offset_t = 1.6
        if N2 == 20.25:		offset_t = 1.3
        if N2 == 25:		offset_t = 1.1
        tIdx_offset = int(offset_t/dt2)
        print("offset time: ", offset_t)

    #Deprecated code for manually choosing start time for search:
    #if N2 == 0.09: tIdx_offset = int(30./dt2)
    #if N2 == 0.25: tIdx_offset = int(20./dt2)
    #if N2 == 1: tIdx_offset = int(10./dt2)
    #if N2 == 2.25: tIdx_offset = int(7./dt2)
    #if N2 == 3.83 or N2 == 4: tIdx_offset = int(6./dt2)
    #if N2 == 6.25: tIdx_offset = int(6./dt2)
    #if N2 == 9: tIdx_offset = int(3./dt2)
    #if N2 == 12.25: tIdx_offset = int(3./dt2)
    #if N2 == 16: tIdx_offset = int(2./dt2)
    #if N2 == 20.25: tIdx_offset = int(1.5/dt2)
    #if N2 == 25: tIdx_offset = int(1./dt2)
    #tIdx_offset = 0

    if forced == 1: tIdx_offset = 0

    if UseShear == 0: 
        #epsilon = -np.min(data)
        if FullFields == 1: epsilon = bs*0.9
        if FullFields == 0: epsilon = 0.1
    if UseShear == 1:
        print(np.max(abs(data))) 
        epsilon = 0.00025
 
    #Search for steps by searching for relatively small 
    #vertical gradients of salinity field.  
    MaxSteps0 = 0
    MaxSteps1 = 1 
    count0 = 0    
    while MaxSteps1 >= MaxSteps0:

        print('count: ',count0)
        if count0 != 0: MaxSteps0 = MaxSteps1

        for tt in range(0,Nt):        
            for ii in range(0,Nx):
                for jj in range(0+zIdx_offset,Nz-zIdx_offset):
                    if forced == 0: logical0 = (data[tt,ii,jj] <= 0) or UseShear==1
                    if forced == 1: logical0 = True
                    if logical0:
                        if forced == 0: 
                            if UseShear == 0: logical1 = abs(data[tt,ii,jj]) <= epsilon
                            if UseShear == 1: logical1 = abs(data[tt,ii,jj]) >= epsilon
                        if forced == 1: logical1 = abs(data[tt,ii,jj]) <= epsilon or data[tt,ii,jj] > 0
                        
                        if logical1 == True: 
                            tmp1[tt,ii,jj] = True
                        else:
                            tmp1[tt,ii,jj] = False                 
                    else: tmp1[tt,ii,jj] = False
        
            count = 0
            flag = 0
            for jj in range(0+zIdx_offset,Nz-zIdx_offset):
                if (tmp1[tt,xIdx,jj] == True) and (flag == 0):
                    count += 1
                    flag = 1
                    j0 = jj
                if (tmp1[tt,xIdx,jj] == False) and (flag == 1):
                    flag = 0
                    if count != 0:
                        dz = z[jj]-z[j0]
                        z_mid = dz/2. + z[j0]
                        Fz_mean = np.mean(data[tt,xIdx,j0:jj])
                        tmp3[tt,count-1]=dz
                        tmp4[tt,count-1]=Fz_mean
                        tmp5[tt,count-1]=z_mid
                if (count==1) and (flag==1) and (jj==Nz-zIdx_offset-1):
                    dz = z[jj]-z[j0]
                    z_mid = dz/2. + z[j0]
                    Fz_mean = np.mean(data[tt,xIdx,j0:jj])
                    tmp3[tt,count-1]=dz
                    tmp4[tt,count-1]=Fz_mean
                    tmp5[tt,count-1]=z_mid

            tmp2[tt] = count

        if forced == 1:
            #We need to start a new time loop again here due to the stencil used 
            #for computing time gradients.
            for tt in range(0,Nt):
                if tt != 0 and tt != (Nt-1):
                    tmp6[tt] = (tmp2[tt+1]-tmp2[tt-1])/(2*dt2)
                    for nn in range(0,count):
                        tmp7[tt,nn] = (tmp3[tt+1,nn]-tmp3[tt-1,nn])/(2*dt2)
                        tmp8[tt,nn] = (tmp5[tt+1,nn]-tmp5[tt-1,nn])/(2*dt2)
                if tt == 0:
                    tmp6[tt] = (tmp2[tt+1]-tmp2[tt])/dt2
                    for nn in range(0,count):
                        tmp7[tt,nn] = (tmp3[tt+1,nn]-tmp3[tt,nn])/dt2
                        tmp8[tt,nn] = (tmp5[tt+1,nn]-tmp5[tt,nn])/dt2
                if tt == (Nt-1):
                    tmp6[tt] = (tmp2[tt]-tmp2[tt-1])/dt2
                    for nn in range(0,count):
                        tmp7[tt,nn] = (tmp3[tt,nn]-tmp3[tt-1,nn])/dt2
                        tmp8[tt,nn] = (tmp5[tt,nn]-tmp5[tt-1,nn])/dt2

        MaxSteps1 = np.max(tmp2[tIdx_offset:])
        print('max # steps: ', MaxSteps1, MaxSteps0, ' epsilon: ', epsilon)

        if (MaxSteps1 >= MaxSteps0):
            epsilon = epsilon - epsilon*0.1

        #if (MaxSteps1 > MaxSteps0):
        step_mask = tmp1
        step_count = tmp2
        step_dz = tmp3
        step_dS = tmp4
        if forced == 1:
           d_dt_step = tmp6
           d_dt_stepDz = tmp7
           d_dt_step_z = tmp8
        #print('array update: ', count0)

        if (count0 == 0) and (MaxSteps1==MaxSteps0): MaxSteps1 = 1

        #Stop 'while' loop:
        #Essentially this avoids using the while loop which was initially designed
        #to try searching for steps by refining epsilon until the max number of steps
        #was found. However, this approach was deprecated in favour of using a fixed epsilon for 
        #for each run set at 0.9*bs (i.e. 10% lower than background stratification for all runs).
        MaxSteps1 = MaxSteps0-1 
        count0 += 1

    if w2f_analysis == 1:
        if Gusto == 0:
            if FullFields == 1: dir_TrackSteps = './Results/' + RunName + '/TrackSteps/'
            if FullFields == 0: dir_TrackSteps = './Results/' + RunName + '/TrackSteps2/'
        if Gusto == 1:
            dir_TrackSteps =  './Results/' + RunName + '_gusto' + '/TrackSteps/'
        #Create directory if it doesn't exist:
        if not os.path.exists(dir_TrackSteps):
            os.makedirs(dir_TrackSteps)

        fnm1 = dir_TrackSteps + 'steps_t.txt'
        fnm2 = dir_TrackSteps + 'steps_dz.txt'
        fnm3 = dir_TrackSteps + 'steps_dS.txt'
        np.savetxt(fnm1,step_count)
        np.savetxt(fnm2,step_dz)
        np.savetxt(fnm3,step_dS)
        if forced == 1:
            fnm4 = dir_TrackSteps + 'd_dt_step.txt'
            fnm5 = dir_TrackSteps + 'd_dt_stepDz.txt'
            fnm6 = dir_TrackSteps + 'd_dt_step_z.txt'
            np.savetxt(fnm4,d_dt_step)
            np.savetxt(fnm5,d_dt_stepDz)
            np.savetxt(fnm6,d_dt_step_z)


    if MakePlot == 1:
        fig=plt.figure(figsize=(width,height))
        grid = plt.GridSpec(1, 1, wspace=0.4, hspace=0.4)
        ax1 = fig.add_subplot(grid[0,0])

        plot_data = step_mask[:,xIdx,:].transpose().astype(int)
        xgrid = t2d_z
        ygrid = z2d_t
        i1 = ax1.contourf(xgrid,ygrid,plot_data, 1, colors=['white','black'])
        if forced == 0: ax1.plot([t[tIdx_offset],t[tIdx_offset]],[0,Lz],'grey')
        ax1.set_xlabel(r'$t$ (s)')
        ax1.set_ylabel(r'$z$ (m)')
        ax1.set_ylim(0,Lz)
        ax1.set_xlim(0,np.max(t))
        #ax1.set_xlim(0,30)
        cb = plt.colorbar(i1, ticks=[0,1])
 
        #ax2 = fig.add_subplot(grid[0,1])
        #ax2.plot(t,step_count)
        #ax2.set_xlabel(r'$t$ (s)')
        #ax2.set_ylabel('Number of steps')

        #ax3 = fig.add_subplot(grid[1,1])
        #ax3.hist(step_count, bins=np.arange(20)+1)
        #ax3.set_xlabel('Number of steps')
        #ax3.set_ylabel('Count')

        #plt.show()
        FigNmBase = 'TrackSteps'
        plt.savefig(FigNmBase + RunName + '_tz_' + str(nfiles) + '.png') 


if TrackInterfaces == 1:
    Sz = d_dz(S,Nt,Nx,Nz,z)
    step_mask = np.zeros((Nt,Nx,Nz),dtype=bool)

    #Exclude boundary layer effects:
    if (N2 == 0.09) or (N2 == 0.25): zIdx_offset = int(.1/dz)
    if (N2 != 0.09) and (N2 != 0.25): zIdx_offset = int(.05/dz)
    zIdx_offset = 0

    tIdx_offset = 0

    #Choose x point for analysis:
    xIdx = int(Nx/2.)

    #epsilon = np.min(abs(data[:,xIdx,:])) 
    epsilon = bs*1.1

    for tt in range(0+tIdx_offset,Nt):
        Fz_min = np.min(data[tt+tIdx_offset,xIdx,:])
        while abs(Fz_min) > epsilon:
            min_idx = np.where(data[tt+tIdx_offset,xIdx,:] == Fz_min)
            step_mask[tt+tIdx_offset,xIdx,min_idx] = True

            #Find new minimum excluding previous minimum:
            not_min_idxs = np.where(data[tt+tIdx_offset,xIdx,:] > Fz_min)
            Fz_min = np.min(data[tt+tIdx_offset,xIdx,not_min_idxs])

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
         
        plt.show()
        pdb.set_trace()


if Fluxes == 1:

    if UseProxySz == 1: 
        #This uses dS/dz as a proxy for fluxes which is what Jerin did for his MSc project.
        #This ignores the variation of the flow.
        f = Sz
    if UseProxySz == 0:
        #compute the vertical mass flux (of salt):
        w = -d_dx(Psi,Nt,Nx,Nz,x)
        flux = np.multiply(rho,w)*dx    
        f = flux

    #tm1 from TrackSteps provides step region points in space and time.
    #We also need the points in between the steps but excluding boundary layers:
    NotSteps = tmp1 == False

    #Exclude boundary layer effects:
    if (N2 == 0.09) or (N2 == 0.25): zIdx_offset = int(.1/dz)
    if (N2 != 0.09) and (N2 != 0.25): zIdx_offset = int(.05/dz)
    NotBoundLayer = np.ones((Nt,Nx,Nz))
    NotBoundLayer[:,:,0:zIdx_offset] = 0
    NotBoundLayer[:,:,(Nz-zIdx_offset):Nz-1] = 0
    NotSteps = np.multiply(NotSteps,NotBoundLayer)
    NotSteps = NotSteps != 0
  
    #find max, min and means across space and time:
    f_max_steps = np.max(f[tmp1])
    f_min_steps = np.min(f[tmp1])
    f_mean_steps = np.mean(f[tmp1])
    f_total_steps = np.sum(f[tmp1])
  
    idxsNotSteps = np.where( tmp1 == False )
    f_max_Notsteps = np.max(f[idxsNotSteps])
    f_min_Notsteps = np.min(f[idxsNotSteps])
    f_mean_Notsteps = np.mean(f[idxsNotSteps])
    f_total_Notsteps = np.sum(f[idxsNotSteps])

    print(f_max_steps,f_min_steps,f_mean_steps,f_max_Notsteps,f_min_Notsteps,f_mean_Notsteps)
    print(f_total_steps,f_total_Notsteps)

    #Find max, min and mean across space as function of time.
    #First take absolute value to enable use of log scale:
    f_min_abs_steps_t = np.zeros((Nt))
    f_min_steps_t = np.zeros((Nt))
    f_max_abs_steps_t = np.zeros((Nt))
    f_max_steps_t = np.zeros((Nt))
    f_mean_abs_steps_t = np.zeros((Nt))
    f_mean_steps_t = np.zeros((Nt))
    f_min_abs_Notsteps_t = np.zeros((Nt))
    f_min_Notsteps_t = np.zeros((Nt))
    f_max_abs_Notsteps_t = np.zeros((Nt))
    f_max_Notsteps_t = np.zeros((Nt))
    f_mean_abs_Notsteps_t = np.zeros((Nt))
    f_mean_Notsteps_t = np.zeros((Nt))
    if UseProxySz == 0:
        f_total_steps_t = np.zeros((Nt))
        f_total_Notsteps_t = np.zeros((Nt))
    for tt in range(0,Nt):
        if np.any(tmp1[tt,:,:])==1:
            f_min_abs_steps_t[tt] 	= np.min( np.abs( f[tt,tmp1[tt,:,:]] ))
            f_min_steps_t[tt] 		= np.min( f[tt,tmp1[tt,:,:]] )
            f_max_abs_steps_t[tt] 	= np.max( np.abs( f[tt,tmp1[tt,:,:]] ))
            f_max_steps_t[tt] 		= np.max( f[tt,tmp1[tt,:,:]] )
            f_mean_abs_steps_t[tt] 	= np.mean( np.abs( f[tt,tmp1[tt,:,:]] ))
            f_mean_steps_t[tt] 		= np.mean( f[tt,tmp1[tt,:,:]] )
            f_min_abs_Notsteps_t[tt] 	= np.min( np.abs( f[tt,NotSteps[tt,:,:]] ))
            f_min_Notsteps_t[tt] 	= np.min( f[tt,NotSteps[tt,:,:]] )
            f_max_abs_Notsteps_t[tt] 	= np.max( np.abs( f[tt,NotSteps[tt,:,:]] ))
            f_max_Notsteps_t[tt] 	= np.max( f[tt,NotSteps[tt,:,:]] )
            f_mean_abs_Notsteps_t[tt] 	= np.mean( np.abs( f[tt,NotSteps[tt,:,:]] ))
            f_mean_Notsteps_t[tt] 	= np.mean( f[tt,NotSteps[tt,:,:]] )
            if UseProxySz == 0:
                f_total_steps_t[tt]   	= np.sum( f[tt,tmp1[tt,:,:]] )
                f_total_Notsteps_t[tt]	= np.sum( f[tt,NotSteps[tt,:,:]] )
        else:  
            f_min_abs_steps_t[tt] 	= np.nan 
            f_min_steps_t[tt] 		= np.nan 
            f_max_abs_steps_t[tt] 	= np.nan 
            f_max_steps_t[tt] 		= np.nan 
            f_mean_abs_steps_t[tt] 	= np.nan 
            f_mean_steps_t[tt] 		= np.nan 
            f_min_abs_Notsteps_t[tt] 	= np.nan 
            f_min_Notsteps_t[tt] 	= np.nan 
            f_max_abs_Notsteps_t[tt] 	= np.nan 
            f_max_Notsteps_t[tt] 	= np.nan 
            f_mean_abs_Notsteps_t[tt] 	= np.nan 
            f_mean_Notsteps_t[tt] 	= np.nan 
            if UseProxySz == 0:
                f_total_steps_t[tt]   	= np.nan
                f_total_Notsteps_t[tt]	= np.nan

    #plot results
    fig = plt.figure(1, figsize=(width,height))
    fig.set_tight_layout(True)
    grid = plt.GridSpec(1, 2, wspace=0.3, hspace=0.)
    ax1 = fig.add_subplot(grid[0,0])

    if UseProxySz == 1:
        ax1.semilogy(f_min_abs_steps_t, label=r'min$|S_z|$ in steps')
        ax1.semilogy(f_max_abs_steps_t, label=r'max$|S_z|$ in steps')
        ax1.semilogy(f_mean_abs_steps_t, label=r'$\overline{|S_z|}$ in steps')
        ax1.semilogy(f_min_abs_Notsteps_t, label=r'min$|S_z|$')
        ax1.semilogy(f_min_Notsteps_t, '-o', label=r'min$(S_z)$')
        ax1.semilogy(f_max_abs_Notsteps_t, label=r'max$|S_z|$')
        ax1.semilogy(f_max_Notsteps_t, label=r'max$(S_z)$')
        ax1.semilogy(f_mean_abs_Notsteps_t, label=r'$\overline{|S_z|}$')
        ax1.semilogy(f_mean_Notsteps_t, label=r'$\overline{S_z}$')
        ax1.legend()

        ax2 = fig.add_subplot(grid[0,1])
        ax2.plot(f_min_Notsteps_t, label=r'min$(S_z)$')
        ax2.plot(f_mean_Notsteps_t, label=r'$\overline{S_z}$')
        ax2.legend()
    if UseProxySz == 0:
        ax1.plot(f_total_steps_t, label=r'mean in steps')
        ax1.plot(f_total_Notsteps_t, label=r'mean')
        ax1.plot([0,Nt],[0,0], '--k')
        ax1.legend()


    plt.show()


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
        State_2 = np.zeros((Nt,Nx,Nz,nvars))

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

    #When rotating into space of waves using matrix exponential:
    if MakeCoordRotation == 1: State_R = np.zeros((Nt,Nx,Nz,nvars))

    if Modulated == 0:
        dataA = Psi
        dataB = S
        if nvars == 3: dataC = T
    else:
        dataA = Psi_r
        dataB = S_r
        if nvars == 3: dataC = T_r

    for tt in range(0,Nt):
        print(tt)
        
        if FullDomain == 1:
            tmp_Psi['g'] = dataA[tt,:,:]
            tmp_S['g'] = dataB[tt,:,:]
            if nvars == 3: tmp_T['g'] = dataC[tt,:,:]
        if SinglePoint == 1:
            tmp_Psi['g'][xIdx,zIdx] = dataA[tt,0,0]
            tmp_S['g'][xIdx,zIdx] = dataB[tt,0,0]
            if nvars == 3: tmp_T['g'][xIdx,zIdx] = dataC[tt,0,0]

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

        if MakeCoordRotation == 1:
            tmp_psi_r = domain.new_field()
            tmp_S_r = domain.new_field()
            tmp_psi_r.meta['z']['parity'] = -1
            tmp_S_r.meta['z']['parity'] = -1

        #Loop over wavenumbers:
        for jj in range(0,Nz):
            for ii in range(0,Nx):

                k = kk[ii]
                n = kk_cosine[jj]

                kvec    = np.array([k,n])
                kmag    = np.linalg.norm(kvec)
                if (BasisCheck2==1) and (tt==0): kmag_arr[ii,jj] = kmag

                if kmag != 0:

                    if nvars == 3: c1 = sqrt(-ct/bt)
                    c2      = sqrt(cs/bs) 
                    c3 = abs(k)/kmag*sqrt(-(ct*bt-cs*bs))          #N.B. omega = c3*sqrt(g)

                    if Modulated == 0:
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

                            r_1     = np.array([k/abs(k),1])
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
                        EigenVecsM = np.array([r_1,r0,r1]).transpose()
                        f_hat_vec = np.array([Psi_hat[ii,jj],T_hat[ii,jj],S_hat[ii,jj]])
                    if nvars == 2:
                        EigenVecsM = np.array([r_1,r1]).transpose()
                        f_hat_vec = np.array([Psi_hat[ii,jj],S_hat[ii,jj]])

                    #Make transformation to find amplitudes of Natural basis 
                    EigenVecsM_inv = linalg.inv(EigenVecsM)
                    sigma_kn = np.mat(EigenVecsM_inv)*np.mat(f_hat_vec).T

                    if nvars == 3:
                        sigma_1[ii,jj] = sigma_kn[0,0]
                        sigma0[ii,jj] = sigma_kn[1,0]
                        sigma1[ii,jj] = sigma_kn[2,0]
                    if nvars == 2:
                        sigma_1[ii,jj] = sigma_kn[0,0]
                        sigma1[ii,jj] = sigma_kn[1,0]

                    if MakeCoordRotation == 1 and ii < int(Nx/2.):
                        diagonalM = np.zeros((2,2))*1j
                        omega_1 = -c3*sqrt(g)*1j
                        omega1 = c3*sqrt(g)*1j
                        diagonalM[0,0] = np.exp(-omega_1*t[tt])
                        diagonalM[1,1] = np.exp(-omega1*t[tt])

                        MatrixExp = np.mat(EigenVecsM)*np.mat(diagonalM)*np.mat(EigenVecsM_inv)
                        fnc_r = np.mat(MatrixExp)*np.mat(f_hat_vec).T

                        #print(MatrixExp)
                        #print(fnc_r)
                        #print(fnc_r.shape)
                        tmp_psi_r['c'][ii,jj] = fnc_r[0,0]
                        tmp_S_r['c'][ii,jj] = fnc_r[1,0]
                    
                else:

                    if MakeCoordRotation and ii < int(Nx/2.):
                        tmp_psi_r['c'][ii,jj] = Psi_hat[ii,jj] 
                        tmp_S_r['c'][ii,jj] = S_hat[ii,jj]

        if MakeCoordRotation == 1:
            State_R[tt,:,:,0] = tmp_psi_r['g']
            State_R[tt,:,:,1] = tmp_S_r['g']

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

                State_2[tt,:,:,0] = tmp_Psi['g']
                State_2[tt,:,:,1] = tmp_T['g']
                State_2[tt,:,:,2] = tmp_S['g']
            if nvars == 2:
                tmp_Psi = domain.new_field()
                tmp_S = domain.new_field()
                tmp_Psi.meta['z']['parity'] = -1
                tmp_S.meta['z']['parity'] = -1

                tmp_Psi['c'] = Psi_hat[0:int(Nx/2.),:]
                tmp_S['c'] = S_hat[0:int(Nx/2.),:]

                State_2[tt,:,:,0] = tmp_Psi['g']
                State_2[tt,:,:,1] = tmp_S['g']

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
            if tt == 0 and Modulated == 0:
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

            if BasisCheck2==1 and tt==0 and Modulated==0:
                fnm_kmag = dir_ivec + 'kmag_arr.txt'
                np.savetxt(fnm_kmag,kmag_arr)


if StateS_2 == 1:
#New salinity after transformation to make linear operator Skew Hermitian:

    if NaturalBasis==1 and BasisCheck1==1:
        #New salinity computed using code to compute Natural basis,
        #that is, via adjusting the salinity's spectral coefficients:
        if FullDomain == 1: data = State_2[:,:,:,1]
        if SinglePoint == 1: data = State_2[:,xIdx,zIdx,1]
    else:
        #For salinity the transformation to make linear operator Skew Hermitian 
        #is only a constant factor and so if desired we can avoid computing all the 
        #eigenvectors here (i.e. avoid using the code to compute Natural basis).
        #This is not true for the streamfunction.
        factor = np.sqrt(g*cs/bs)
        S2 = S*factor 
        data = S2

    if (MakePlot == 1) and (PlotTZ==1 or PlotXZ==1):
        PlotTitle = ''
        FigNmBase = 'S_2'
        cmap="bwr"

        nlevs = 41
        SMin = -.02
        SMax = .02
        dS = (SMax-SMin)/(nlevs-1)
        clevels = np.arange(nlevs)*dS + SMin

        xlim=(0,60)

        PlotTitle=r'$S$ (g/kg)'

    ErrorCheck = 0
    if NaturalBasis==1 and ErrorCheck==1:
        #Make sure transformation to make linear operator Skew Hermitian
        #gives the same salinity field whether or not we use the code for computing Natural basis.
        #This should boil down to simply checking there is no difference if we adjust the salinity field 
        #in real or phase space.
        if FullDomain == 1:
            xIdx = int(Nx/2.)
            zIdx = int(Nz/2.)
            diff = data[:,xIdx,zIdx].flatten() -\
                      S[:,xIdx,zIdx].flatten()*np.sqrt(g*cs/bs)             
            if MakePlot == 1:
                plt.plot(t,S[:,xIdx,zIdx].flatten()*np.sqrt(g*cs/bs),'ok')
                plt.plot(t,data[:,xIdx,zIdx].flatten(),'+r',markersize=15)
                plt.show()
        if SinglePoint == 1: 
            diff = data.flatten() - S.flatten()*np.sqrt(g*cs/bs)
            if MakePlot == 1:
                plt.plot(t,S.flatten()*np.sqrt(g*cs/bs),'ok')
                plt.plot(t,data.flatten(),'+r',markersize=15)
                plt.show()
        print(np.max(abs(diff)))

    #Write out data for checking the Natural basis itself (used to reconstruct state in NaturalFreq.py):
    if w2f_analysis == 1:
        fnm = './StateS_2_N2_02_25_sp.txt'
        if FullDomain == 1: 
            xIdx = int(Nx/2.)
            zIdx = int(Nz/2.)
            np.savetxt(fnm,data[:,xIdx,zIdx].flatten())
        if SinglePoint == 1: np.savetxt(fnm,data.flatten())


if StateS == 1:
    #print(np.min(S), np.max(S))
    #pdb.set_trace()

    if NaturalBasis == 1 and MakeCoordRotation == 1:
        if FullDomain == 1: data = State_R[:,:,:,1]
        if SinglePoint == 1: data = State_R[:,xIdx,zIdx,1]
    if NaturalBasis == 0 and Modulated == 1: 
        data = S_r
    if NaturalBasis == 0 and Modulated == 0:
        data = S

    if MakePlot == 1:
        FigNmBase = 'S_r'
        cmap = 'bwr'

        if CoefficientSpace == 0:
            nlevs = 41
            if Modulated == 1:
                PlotTitle = r'$\zeta$ (g/kg)'
                SMin = -.02
                SMax = .02
                #SMin = -.2
                #SMax = .2
            if Modulated == 0:
                PlotTitle = r'$S$ (g/kg)'
                SMin = -20
                SMax = 20
                #SMin = -200
                #SMax = 200
                #SMin = np.min(State[int(Nx/2.),:,:,vIdx])
                #SMax = np.max(State[int(Nx/2.),:,:,vIdx])
            dS = (SMax-SMin)/(nlevs-1)
            clevels = np.arange(nlevs)*dS + SMin

            xlim=(0,60)
        
        if CoefficientSpace == 1:
            tIdx = 1
            clevels = 50
            plt.contourf(data[tIdx,:,:], clevels)
            plt.colorbar()
            plt.show()
            pdb.set_trace()

    ErrorCheck = 0
    if ErrorCheck == 1:

        #First write rotated salinity field from single point output 
        #to a text file to enable comparison with full domain data.
        #This program either reads single point of full domain data and not both currently.
        if w2f_analysis == 1:
            fnm = './data_sp.txt'
            np.savetxt(fnm,data)

        if w2f_analysis == 0:
            #Make sure you get same results for modulated flow
            #whether you use full domain or single point data:
            fnm = './data_sp.txt'
            data_sp = np.loadtxt(fnm)
            data_fulldomain = data[:,xIdx,zIdx].flatten() #(take central point for comparison)
            diff = data_fulldomain - data_sp
            print('max difference: ', np.max(abs(diff)))
            plt.plot(t,data_fulldomain,'k')
            plt.plot(t,diff,'r')
            plt.show()


if TestModulation == 1:
    #data = S-S_r
    #print('S: ', np.min(S),np.max(S))
    #print('S_r: ', np.min(S_r),np.max(S_r))
    #print('diff: ', np.max(data))
    #pdb.set_trace()

    tmp = S*S
    print(tmp.shape)

    tIdx=0
    data1 = np.sqrt(S[tIdx,:,:]*S[tIdx,:,:] + Psi[tIdx,:,:]*Psi[tIdx,:,:])
    data2 = np.sqrt(S_r[tIdx,:,:]*S_r[tIdx,:,:] + Psi_r[tIdx,:,:]*Psi_r[tIdx,:,:])
    data = data1-data2
    print(np.max(data))
    plt.contourf(data.transpose(),50)
    plt.colorbar()
    plt.show()
    pdb.set_trace()



if StatePsi == 1:
#New salinity after applying coordinate rotation (matrix exponential):

    if NaturalBasis == 1 and MakeCoordRotation == 1:
        if FullDomain == 1: data = State_R[:,:,:,0]
        if SinglePoint == 1: data = State_R[:,xIdx,zIdx,0]
    if NaturalBasis == 0 and Modulated == 1: 
        data = Psi_r
    if NaturalBasis == 0 and Modulated == 0:
        data = Psi

    if MakePlot == 1:
        PlotTitle = r'$\psi$'
        FigNmBase = 'Psi_r'
        cmap = 'bwr'

        if CoefficientSpace == 0:
            nlevs = 41
            #psiMin = -.0002
            #psiMax = .0002
            #psiMin = np.min(State[int(Nx/2.),:,:,0])
            #psiMax = np.max(State[int(Nx/2.),:,:,0])
            psiMin = np.min(Psi)
            psiMax = np.max(Psi)
            dpsi = (psiMax-psiMin)/(nlevs-1)
            clevels = np.arange(nlevs)*dpsi + psiMin
            #clevels = 50

            #xlim=(10,40)
        else:
            tIdx = 1
            clevels = 50
            plt.contourf(data[tIdx,:,:], clevels)
            plt.colorbar()
            plt.show()
            pdb.set_trace()


if ForwardTransform == 1:
    data = S
    #data = Psi
    data = State_R[:,:,:,1]

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
    plt.contourf(data_hat[1,0:int(Nx/2.),:],50)
    plt.colorbar()
    plt.show()
    pdb.set_trace()

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


if SpectralAnalysis == 1:
    if AnalyseS == 1:
        if Modulated == 0: data = S2
        if Modulated == 1: data = S_r
    if AnalyseS == 0:
        if Modulated == 0: data = Psi
        if Modulated == 1: data = Psi_r

    #idx0 = int(10./dt2)
    #tmp = data[idx0:,:,:]
    #data = tmp

    Welch = 1
    spectralCoef, freqvec, nperseg = spectral_analysis(data,dt2,Welch=Welch)
    npersegStr = str(nperseg)

    print('analysis timestep: ', dt2)
    print('frequency resolution: ', np.min(freqvec[1:]))

    #intPSD = np.trapz(spectralCoef.flatten(),freqvec)
    #print("integral of PSD: ", intPSD)

    if w2f_analysis == 1:
        if FullDomain == 1:
            Idx1 = int(Nx/2.)
            Idx2 = int(Nz/2.)
        else:
            Idx1 = 0
            Idx2 = 0
        f = spectralCoef[:,Idx1,Idx2]

        if N2 == 25:       fnm = './psd_N2_25' + '_' + npersegStr
        if N2 == 20.25:    fnm = './psd_N2_20_25' + '_' + npersegStr
        if N2 == 16:       fnm = './psd_N2_16' + '_' + npersegStr
        if N2 == 14.0625:  fnm = './psd_N2_14_0625' + '_' + npersegStr
        if N2 == 12.25:    fnm = './psd_N2_12_25' + '_' + npersegStr
        if N2 == 10.5625:  fnm = './psd_N2_10_5625' + '_' + npersegStr
        if N2 == 9:        fnm = './psd_N2_09' + '_' + npersegStr
        if N2 == 7.5625:   fnm = './psd_N2_07_5625' + '_' + npersegStr
        if N2 == 6.25:     fnm = './psd_N2_06_25' + '_' + npersegStr
        if N2 == 4:        fnm = './psd_N2_04' + '_' + npersegStr
        if N2 == 2.25:     fnm = './psd_N2_02_25' + '_' + npersegStr
        if N2 == 1:        fnm = './psd_N2_01' + '_' + npersegStr
        if N2 == 0.25:     fnm = './psd_N2_00_25' + '_' + npersegStr
        if N2 == 0.09:     fnm = './psd_N2_00_09' + '_' + npersegStr
        if CheckPSD == 1: fnm = fnm + '_' + str(dt2)
        if Modulated == 1: fnm = fnm + '_R'
        if MeanFlowAnalysis == 1: fnm = fnm + '_mf'
        fnm = fnm + '.txt'
        np.savetxt(fnm, (f,freqvec))

    if MakePlot == 1:
        yscale = 'log'
        #yscale = 'linear'
        PlotGrid = False

        if PSD_vs_N_plot==0 and CheckPSD==0 and MultiPoint==0:
            if PlotBigMode == 1:
                #plot dominant frequencies across domain: 
                rangeF = 1.5
                nf = 51
                df = rangeF/(nf-1)
                clevels = np.arange(nf)*df
                #cmap = 'cool'
                cmap = 'plasma'
                ax1.set_xlabel(r'$x$ (m)')
                ax1.set_ylabel(r'$z$ (m)')
                clabel = r'$\omega$ (rad/s)'

            #label = "{:<04.2}".format(zpnts[ii])
            if FullDomain == 1:
                Idx1 = int(Nx/2.)
                Idx2 = int(Nz/2.)
            else:
                Idx1 = 0
                Idx2 = 0
            data = spectralCoef[:,Idx1,Idx2]

            xgrid = freqvec*(2*np.pi)
            xlim = (0,5)
            ylim = (1e-18,1e+0)
            xlabel = r'$\omega$ (rad/s)'
            if Modulated == 0: ylabel = r'PSD ([$S$]$^2$/(rad/s))'
            if Modulated == 1: ylabel = r'PSD ($\left[{\zeta}\right]^2$/(rad/s))'
            PlotTitle = ''
            FigNmBase = 'psd_'

        if PSD_mod_unmod_plot == 1:
            fig1 = plt.figure(figsize=(width,height))
            grid1 = plt.GridSpec(1, 1, wspace=0., hspace=0.)
            ax1 = fig1.add_subplot(grid1[0,0])
            fig1.set_tight_layout(True) 

            if N2 == 0.25: 	fname = 'psd_N2_00_25'
            if N2 == 1: 	fname = 'psd_N2_01'
            if N2 == 2.25: 	fname = 'psd_N2_02_25'
            if N2 == 4: 	fname = 'psd_N2_04'
            if N2 == 6.25: 	fname = 'psd_N2_06_25'
            if N2 == 9: 	fname = 'psd_N2_09'
            if N2 == 12.25: 	fname = 'psd_N2_12_25'
            if N2 == 16: 	fname = 'psd_N2_16'
            if N2 == 20.25: 	fname = 'psd_N2_20_25'
            if N2 == 25: 	fname = 'psd_N2_25'

            data = np.loadtxt('./SpectralAnalysis/unmodulated/' + fname + '_' + npersegStr + '.txt')
            data_mod = np.loadtxt('./SpectralAnalysis/modulated/' + fname + '_' + npersegStr + '_R.txt')

            intPSD = np.trapz(data[0,:],data[1,:])
            intPSD_mod = np.trapz(data_mod[0,:],data_mod[1,:])
            print("integral of PSD: ", intPSD)
            print("integral of PSD for modulated system: ", intPSD_mod)
            print("intPSD/intPSD_mod: ", intPSD/intPSD_mod)

            ax1.plot(xgrid,data[0,:],'.k-', linewidth=2, label=r'$S$')
            ax1.plot(xgrid,data_mod[0,:],'.-', color='lightgrey', linewidth=1, label=r'$\zeta$')
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel(ylabel)
            ax1.set_yscale("log")
            ax1.set_xlim(0,5)
            ax1.set_ylim(ylim)

            #Load tracked spectrum features files so we can overplot omega_well and 
            #bandwidths of Meanflow and IGW:
            datMF = np.loadtxt('./meanflowarr.txt')
            datIGW = np.loadtxt('./psdIGWarr.txt')
            N_vec = [0.5, 1, 1.5, 2, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.5, 5]
            Nidx = np.where(N_vec == np.sqrt(N2))
            omega_well = datMF[Nidx,3].flatten()[0]
            IGW_mid = datIGW[Nidx,4].flatten()[0]
            IGW_maxf = (datIGW[Nidx,4] + datIGW[Nidx,2]/2).flatten()[0]
            PSD_well = datMF[Nidx,2].flatten()[0] 
            ax1.plot([omega_well,omega_well],[min(ylim),max(ylim)],'k', label=r'$\omega_{well}$')
            ax1.plot([IGW_maxf,IGW_maxf],[min(ylim),max(ylim)],'--k', label=r'$\omega_{well} + \Delta\omega_{IGW}$')
            ax1.text(IGW_mid, 1e-16, r'$\Delta\omega_{IGW}$', horizontalalignment='center', verticalalignment='center')
            xoffset = 0.2
            ax1.plot([omega_well,IGW_mid-xoffset], [1e-16,1e-16], 'k')
            ax1.plot([IGW_mid+xoffset,IGW_maxf], [1e-16,1e-16], 'k')
            ax1.text(omega_well/2, 1e-16, r'$\Delta\omega_{MF}$', horizontalalignment='center', verticalalignment='center')
            ax1.scatter(omega_well,PSD_well, s=100, facecolors='none', edgecolors='k')
            ax1.scatter(IGW_maxf,PSD_well, s=100, facecolors='none', edgecolors='k')
            plt.legend(frameon=False)
            plt.show()

        if PSD_vs_N_plot == 1 and CheckPSD == 0:
            fig1 = plt.figure(figsize=(width,height))
            grid1 = plt.GridSpec(1, 1, wspace=0.5, hspace=0.)
            ax1 = fig1.add_subplot(grid1[0,0])

            #choices for GAFD seminar 2018:
            #nperseg = '900'
            #data1 = np.loadtxt('./psd_N2_00_09' + '_' + npersegStr + '.txt')
            #data2 = np.loadtxt('./psd_N2_01' + '_' + npersegStr + '.txt')
            #data3 = np.loadtxt('./psd_N2_03_83' + '_' + npersegStr + '.txt')
            #data4 = np.loadtxt('./psd_N2_09' + '_' + npersegStr + '.txt')
            #data5 = np.loadtxt('./psd_N2_25' + '_' + npersegStr + '.txt')
            #labelvec = ('0.03','1', '1.96', '3', '5')
            #colorvec = ('k','grey','g','b','c')
            #lstyle_vec('-','-','-','-','-')
            #lwidth_vec = (1,1,1,1,1)

            if Modulated == 1: 
                fdir = './SpectralAnalysis/modulated/'
                tag_R = '_R'
            else: 
                fdir = './SpectralAnalysis/unmodulated/'
                tag_R = ''
            
            #data1 = np.loadtxt(fdir + 'psd_N2_00_25' + '_' + npersegStr + tag_R + '.txt')
            #data2 = np.loadtxt(fdir + 'psd_N2_01' + '_' + npersegStr + tag_R + '.txt')
            #data3 = np.loadtxt(fdir + 'psd_N2_02_25' + '_' + npersegStr + tag_R + '.txt')
            #data4 = np.loadtxt(fdir + 'psd_N2_04' + '_' + npersegStr + tag_R + '.txt')
            #data5 = np.loadtxt(fdir + 'psd_N2_16' + '_' + npersegStr + tag_R + '.txt')
            #data6 = np.loadtxt(fdir + 'psd_N2_25' + '_' + npersegStr + tag_R + '.txt')

            data1 = np.loadtxt(fdir + 'psd_N2_00_25' + '_' + npersegStr + tag_R + '.txt')
            data2 = np.loadtxt(fdir + 'psd_N2_01' + '_' + npersegStr + tag_R + '.txt')
            data3 = np.loadtxt(fdir + 'psd_N2_04' + '_' + npersegStr + tag_R + '.txt')
            data4 = np.loadtxt(fdir + 'psd_N2_16' + '_' + npersegStr + tag_R + '.txt')

            #labelvec = ('0.5','1','1.5','2','3','5')
            #colorvec = ('grey','grey','k','k','k','k','k')
            #lstyle_vec = ('-','-','-','-','-',':',':')
            #lwidth_vec = (3,1,3,2,1,2,1)
            #ax1.set_xlim(0,8)

            labelvec = ('0.5','1','2','4')
            colorvec = ('k','k','k','k')
            lstyle_vec = ('-','-','-','-')
            lwidth_vec = (1,2,3,4)
            ax1.set_xlim(0,6)

            ax1.set_xlabel(r'$\omega$ (rad/s)')
            if Modulated == 0: 
                #ax1.set_ylim(1e-12,1e+4)
                ax1.set_ylim(1e-18,1e+0)
                ax1.set_ylabel(r'PSD ([$S$]$^2$/(rad/s))')
            else: 
                ax1.set_ylim(1e-18,1e+0)
                ax1.set_ylabel(r'PSD ($\left[{\zeta}\right]^2$/(rad/s))')
            c = 2*np.pi
            ax1.semilogy(data1[1,:]*c,data1[0,:], color=colorvec[0], linestyle=lstyle_vec[0], linewidth=lwidth_vec[0], label=labelvec[0])
            ax1.semilogy(data2[1,:]*c,data2[0,:], color=colorvec[1], linestyle=lstyle_vec[1], linewidth=lwidth_vec[1], label=labelvec[1])
            ax1.semilogy(data3[1,:]*c,data3[0,:], color=colorvec[2], linestyle=lstyle_vec[2], linewidth=lwidth_vec[2], label=labelvec[2])
            ax1.semilogy(data4[1,:]*c,data4[0,:], color=colorvec[3], linestyle=lstyle_vec[3], linewidth=lwidth_vec[3], label=labelvec[3])
            #ax1.semilogy(data5[1,:]*c,data5[0,:], color=colorvec[4], linestyle=lstyle_vec[4], linewidth=lwidth_vec[4], label=labelvec[4])
            #ax1.semilogy(data6[1,:]*c,data6[0,:], color=colorvec[5], linestyle=lstyle_vec[5], linewidth=lwidth_vec[5], label=labelvec[5])
            #ax1.semilogy(data7[1,:]*c,data7[0,:], color=colorvec[6], linestyle=lstyle_vec[6], linewidth=lwidth_vec[6], label=labelvec[6])

            ax1.legend(frameon=False, title=r'$N$ (rad/s)')

            #some plot annotations:
            ax1.semilogy([0.5,0.5],[1e-18,1e+0], color=colorvec[0], linestyle=lstyle_vec[0], linewidth=lwidth_vec[0])
            ax1.semilogy([1,1],[1e-18,1e+0], color=colorvec[1], linestyle=lstyle_vec[1], linewidth=lwidth_vec[1])
            ax1.semilogy([2,2],[1e-18,1e+0], color=colorvec[2], linestyle=lstyle_vec[2], linewidth=lwidth_vec[2])
            ax1.semilogy([4,4],[1e-18,1e+0], color=colorvec[3], linestyle=lstyle_vec[3], linewidth=lwidth_vec[3])

            plt.show()

        if CheckPSD == 1:

            fig1 = plt.figure(figsize=(width,height))
            grid1 = plt.GridSpec(1, 1, wspace=0.5, hspace=0.)
            ax1 = fig1.add_subplot(grid1[0,0])

            if dt==0.1:
                data0 = np.loadtxt('./psd_N2_02_25_3000' + '_0.1' + '.txt')
                data1 = np.loadtxt('./psd_N2_02_25_1500' + '_0.2' + '.txt')
                data2 = np.loadtxt('./psd_N2_02_25_750' + '_0.4' + '.txt')
                labelvec = (r'$\Delta t$=0.1', r'$\Delta t$=0.2', r'$\Delta t$=0.4')

            #data0 = np.loadtxt('./psd_N2_02_25_2000' + '_0.01' + '.txt')
            #data1 = np.loadtxt('./psd_N2_02_25_2000' + '_0.02' + '.txt')
            #data2 = np.loadtxt('./psd_N2_02_25_2000' + '_0.04' + '.txt')
            #data3 = np.loadtxt('./psd_N2_02_25_2000' + '_0.08' + '.txt')
            #data4 = np.loadtxt('./psd_N2_02_25_2000' + '_0.16' + '.txt')

            if dt == 0.01:
                data0 = np.loadtxt('./psd_N2_02_25_30000' + '_0.01' + '.txt')
                data1 = np.loadtxt('./psd_N2_02_25_15000' + '_0.02' + '.txt')
                data2 = np.loadtxt('./psd_N2_02_25_7500' + '_0.04' + '.txt')
                data3 = np.loadtxt('./psd_N2_02_25_3750' + '_0.08' + '.txt')
                data4 = np.loadtxt('./psd_N2_02_25_1875' + '_0.16' + '.txt')
                labelvec = (r'$\Delta t$=0.01', r'$\Delta t$=0.02', r'$\Delta t$=0.04', r'$\Delta t$=0.08',r'$\Delta t$=0.16')

            colorvec = ('g','b','c','grey','k','m','r')
            ax1.xlim(0,5)
            ax1.xlabel(r'$f$ (Hz)')
            #ax1.ylim(1e-16,1e0)
            ax1.ylabel('PSD')
            ax1.semilogy(data0[1,:],data0[0,:], color=colorvec[0], label=labelvec[0])
            ax1.semilogy(data1[1,:],data1[0,:], color=colorvec[1], label=labelvec[1])
            ax1.semilogy(data2[1,:],data2[0,:], color=colorvec[2], label=labelvec[2])
            #ax1.semilogy(data3[1,:],data3[0,:], color=colorvec[3], label=labelvec[3])
            #ax1.semilogy(data4[1,:],data4[0,:], color=colorvec[4], label=labelvec[4])
            #ax1.semilogy(data5[1,:],data5[0,:], color=colorvec[5], label=labelvec[5])
            ax1.legend()
            plt.show()

        if MultiPoint == 1:
            fig1 = plt.figure(figsize=(width*1.5,height))
            #fig1.set_tight_layout(True)
            grid1 = plt.GridSpec(1, 2, wspace=0.4, hspace=0.)
            ax0 = fig1.add_subplot(grid1[0,0])
            ax1 = fig1.add_subplot(grid1[0,1])

            xgrid = freqvec*(2*np.pi)
            
            linewvec = [1,2,3,1,2,3,1,2,3]
            colorvec = ['k','k','k','gray','gray','gray','silver','silver','silver']
            ltypevec = ['-','-','-','-','-','-','-','-','-']

            countA = 0
            for ii in range(0,Nx2):
                for jj in range(0,Nz2):

                    label = '(' + str(round(x[xIdx[jj]],2)) + ',' + str(round(z[zIdx[(Nx2-1)-ii]],2)) + ')'
                    ax0.plot(t,data[:,ii,jj], linewidth=linewvec[countA], linestyle=ltypevec[countA], color=colorvec[countA], label=label)
                    ax1.semilogy(xgrid,spectralCoef[:,ii,jj], linewidth=linewvec[countA],  linestyle=ltypevec[countA], color=colorvec[countA], label=label)
                    countA += 1

            ax0.set_xlabel(r'$t$ (s)')
            ax1.legend(frameon=False,fontsize=10)
            ax1.set_xlim(0,4)
            ax1.set_xlabel(r'$\omega$ (rad/s)')
            ax1.set_ylabel(r'PSD')
            if AnalyseS==0 and Modulated==0: 
                ax0.set_ylim(-.004,.002)
                ax0.set_ylabel(r'$\psi$')
            if AnalyseS==0 and Modulated==1:
                #ax0.set_ylim(-.004,.002)
                ax0.set_ylabel(r'$\psi_r$')
            if AnalyseS == 1 and Modulated==0:
                ax1.set_ylim(1e-11,1e2)
                ax0.set_ylabel(r'$S$')            
            if AnalyseS == 1 and Modulated==1:
                #ax1.set_ylim(1e-14,1e-2)
                ax0.set_ylabel(r'$\zeta$')            
            if AnalyseS==1:
                if Modulated==0: fig1.savefig('psd_S' + RunName + '_multip.png')
                if Modulated==1: fig1.savefig('psd_Sr' + RunName + '_multip.png')
            if AnalyseS==0:
                if Modulated==0: fig1.savefig('psd_Psi' + RunName + '_multip.png')
                if Modulated==1: fig1.savefig('psd_Psi_r' + RunName + '_multip.png')
            plt.close(fig1)

            if w2f_analysis==1:
                if AnalyseS==1: np.savetxt('ts_S' + RunName + '_multip.csv', data.reshape(Nt,-1), delimiter=',')
                if AnalyseS==0: np.savetxt('ts_psi' + RunName + '_multip.csv', data.reshape(Nt,-1), delimiter=',')



if TimescaleSeparation == 1:

    #User input:
    nperseg = 1000

    npersegStr = str(nperseg)
    if Modulated == 0: N_vec = np.array((0.5, 1, 1.5, 2, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.5, 5))
    if Modulated==1 or IGWmethod==1: 
        #N_vec = np.array((0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5))
        N_vec = np.array((0.5, 1, 1.5, 2, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.5, 5))
    c = 2*np.pi

    if Modulated == 1:
        fdir = '/home/ubuntu/BoussinesqLab/dedalus/SpectralAnalysis/modulated/' 
        tag_R = '_R'
    else: 
        fdir = '/home/ubuntu/BoussinesqLab/dedalus/SpectralAnalysis/unmodulated/' 
        tag_R = ''

    data1 = np.loadtxt(fdir + 'psd_N2_00_25' + '_' + npersegStr + tag_R + '.txt')
    data2 = np.loadtxt(fdir + 'psd_N2_01' + '_' + npersegStr + tag_R + '.txt')
    data3 = np.loadtxt(fdir + 'psd_N2_02_25' + '_' + npersegStr + tag_R + '.txt')
    data4 = np.loadtxt(fdir + 'psd_N2_04' + '_' + npersegStr + tag_R + '.txt')
    data5 = np.loadtxt(fdir + 'psd_N2_06_25' + '_' + npersegStr + tag_R + '.txt')
    if IGWmethod==0: data6 = np.loadtxt(fdir + 'psd_N2_07_5625' + '_' + npersegStr + tag_R + '.txt')
    data7 = np.loadtxt(fdir + 'psd_N2_09' + '_' + npersegStr + tag_R + '.txt')
    if IGWmethod==0: data8 = np.loadtxt(fdir + 'psd_N2_10_5625' + '_' + npersegStr + tag_R + '.txt')
    data9 = np.loadtxt(fdir + 'psd_N2_12_25' + '_' + npersegStr + tag_R + '.txt')
    if IGWmethod==0: data10 = np.loadtxt(fdir + 'psd_N2_14_0625' + '_' + npersegStr + tag_R + '.txt')
    data11 = np.loadtxt(fdir + 'psd_N2_16' + '_' + npersegStr + tag_R + '.txt')
    data12 = np.loadtxt(fdir + 'psd_N2_20_25' + '_' + npersegStr + tag_R + '.txt')
    data13 = np.loadtxt(fdir + 'psd_N2_25' + '_' + npersegStr + tag_R + '.txt')

    if IGWmethod == 1:
        fdir_R = '/home/ubuntu/BoussinesqLab/dedalus/SpectralAnalysis/modulated/'
        data1_R = np.loadtxt(fdir_R + 'psd_N2_00_25' + '_' + npersegStr + '_R' + '.txt')
        data2_R = np.loadtxt(fdir_R + 'psd_N2_01' + '_' + npersegStr + '_R' + '.txt')
        data3_R = np.loadtxt(fdir_R + 'psd_N2_02_25' + '_' + npersegStr + '_R' + '.txt')
        data4_R = np.loadtxt(fdir_R + 'psd_N2_04' + '_' + npersegStr + '_R' + '.txt')
        data5_R = np.loadtxt(fdir_R + 'psd_N2_06_25' + '_' + npersegStr + '_R' + '.txt')
        data7_R = np.loadtxt(fdir_R + 'psd_N2_09' + '_' + npersegStr + '_R' + '.txt')
        data9_R = np.loadtxt(fdir_R + 'psd_N2_12_25' + '_' + npersegStr + '_R' + '.txt')
        data11_R = np.loadtxt(fdir_R + 'psd_N2_16' + '_' + npersegStr + '_R' + '.txt')
        data12_R = np.loadtxt(fdir_R + 'psd_N2_20_25' + '_' + npersegStr + '_R' + '.txt')
        data13_R = np.loadtxt(fdir_R + 'psd_N2_25' + '_' + npersegStr + '_R' + '.txt')

    #Search for largest psd frequencies excluding mean flow. 
    #This method also identifies the mean flow freq and psd, 
    #and the first minimum between mean flow and IGWs.
    Nn = len(N_vec)
    Nf = int(nperseg/2.+1)
    meanflowarr = np.zeros((Nn,4))
    if Modulated == 0:
        psdIGWarr = np.zeros((Nn,6))
        #The coordinate rotation eliminates the IGWs from the signal.

    #error check:
    #plt.semilogy(data7[1,:]*c, data7[0,:], '.k-')
    #plt.semilogy(data9[1,:]*c, data9[0,:], '.b-')
    #plt.show()

    for nn in range(0,Nn):

        #get psd data:
        if IGWmethod==0: 
            if nn == 0: fhat = data1
            if nn == 1: fhat = data2
            if nn == 2: fhat = data3
            if nn == 3: fhat = data4
            if nn == 4: fhat = data5
            if nn == 5: fhat = data6
            if nn == 6: fhat = data7
            if nn == 7: fhat = data8
            if nn == 8: fhat = data9
            if nn == 9: fhat = data10
            if nn == 10: fhat = data11
            if nn == 11: fhat = data12
            if nn == 12: fhat = data13
        if IGWmethod==1:
            if nn == 0: 
                fhat = data1
                fhat_R = data1_R
            if nn == 1: 
                fhat = data2
                fhat_R = data2_R
            if nn == 2: 
                fhat = data3
                fhat_R = data3_R
            if nn == 3: 
                fhat = data4
                fhat_R = data4_R
            if nn == 4: 
                fhat = data5
                fhat_R = data5_R
            if nn == 5: 
                fhat = data7
                fhat_R = data7_R
            if nn == 6: 
                fhat = data9
                fhat_R = data9_R
            if nn == 7: 
                fhat = data11
                fhat_R = data11_R
            if nn == 8: 
                fhat = data12
                fhat_R = data12_R
            if nn == 9: 
                fhat = data13
                fhat_R = data13_R

        #if N_vec[nn]==1:
            #plt.semilogy(fhat[1,:],fhat[0,:])
            #plt.xlim(0,1)
            #plt.show()
            #pdb.set_trace()

        #find max psd and associated frequency (i.e. the mean flow)
        psdMax = np.max(fhat[0,:])
        idxMax = np.where(fhat[0,:] == psdMax)
        idxMax = np.asarray(idxMax).flatten()[0]
        #meanflowarr[nn,0] = psdMax 
        meanflowarr[nn,1] = fhat[1,idxMax]*c 

        #starting from mean flow peak run up through frequencies and find first minima.
        #Then search for max psd above this minima to find max psd associated with IGWs.
        #Then use the max psd of IGW bandwidth to find last peak in IGW bandwidth.
        flag1=0
        flag2=0
        for ff in range(idxMax,Nf-1):

            psd0 = fhat[0,ff] #(this is psdMax initially)
            psd1 = fhat[0,ff+1]

            if (psd1 <= psd0) and (flag1==0):
                psdMinIdx = ff+1

                #carry this to find upper bound of IGW bandwidth, but 
                #don't use until energy well has been found.
                psdMin = fhat[0,psdMinIdx]
                if Modulated == 0:
                    psdMaxIGW = np.max(fhat[0,psdMinIdx:])
                    idx = np.where( fhat[0,psdMinIdx:] == psdMaxIGW )
                    psdMaxIGWidx = int(np.asarray(idx) + psdMinIdx)
 
            #When energy well has been found stop changing the well idx:
            if psd1 > psd0: flag1=1

            if Modulated == 0:                   
                if IGWmethod == 0: 
                    lowerBoundIdx = psdMinIdx
                    #find upper bound of IGW bandwidth and so compute IGW bandwidth.
                    #Different method required as not always increase of psd after IGW peaks.
                    #I used the energy well psd instead:

                    #The case of N=1 is unique and treated as such:
                    if N_vec[nn] != 1: logical2 = (psd1 >= psdMin)
                    if N_vec[nn] == 1: logical2 = (psd1 <= psd0)

                    if (flag2==0) and (ff > psdMaxIGWidx) and logical2:
                        upperBoundIdx = ff+1

                    if N_vec[nn] != 1:
                        #Add stop condition to avoid missing upper (high freq.) part of IGW bandwidth:
                        #This code searches an interval of frequencies to check for expected trends: 
                        df = fhat[1,1]-fhat[1,0]
                        fwindow = 0.05
                        search_width = int(fwindow/df)
                        logical3 = fhat[0,(ff+1):(ff+1)+search_width] < psdMin
                    if N_vec[nn] == 1: logical3=True

                    if N_vec[nn] != 1: logical4 = (psd1 < psdMin)
                    if N_vec[nn] == 1: logical4 = (psd1 > psd0)
                    #When upper bound has been found stop changing the upper bound idx:
                    if (ff > psdMaxIGWidx) and logical4 and all(logical3): 
                        flag2=1

        if IGWmethod == 1:
            #use crossing points of unmodulated and modulated systems to define IGW bandwidth
            tmp = np.where(fhat_R[0,psdMinIdx:] < fhat[0,psdMinIdx:])
            lowerBoundIdx = np.min( tmp[0] ) + psdMinIdx
            tmp = np.where(fhat_R[0,psdMaxIGWidx:] > fhat[0,psdMaxIGWidx:])
            upperBoundIdx = np.min( tmp[0] ) + psdMaxIGWidx
            print(N_vec[nn], fhat[1,lowerBoundIdx]*c, fhat[1,upperBoundIdx]*c)
            #pdb.set_trace()

        if Modulated == 0:
            #Reverse the search from the upper bound to find the fequency of last IGW peak:
            psd0 = fhat[0,upperBoundIdx]
            psd1 = fhat[0,upperBoundIdx-1]
            lastPeakIdx = upperBoundIdx
            while psd1 > psd0:
                lastPeakIdx = lastPeakIdx-1
                psd0 = fhat[0,lastPeakIdx] 
                psd1 = fhat[0,lastPeakIdx-1] 

        meanflowarr[nn,0] = np.sum( fhat[0,0:psdMinIdx] )
        WellMode = fhat[1,psdMinIdx]
        meanflowarr[nn,2] = fhat[0,psdMinIdx]
        meanflowarr[nn,3] = WellMode*c
        #store results:
        if Modulated == 0:
            psdIGWarr[nn,0] = psdMaxIGW
            psdIGWarr[nn,1] = fhat[1,psdMaxIGWidx]*c
            psdIGWarr[nn,2] = (fhat[1,upperBoundIdx] - fhat[1,lowerBoundIdx])*c
            psdIGWarr[nn,3] = np.sum( fhat[0,lowerBoundIdx:upperBoundIdx] )
            psdIGWarr[nn,4] = ((fhat[1,upperBoundIdx] - fhat[1,lowerBoundIdx])*c)/2. + fhat[1,lowerBoundIdx]*c
            psdIGWarr[nn,5] = fhat[1,lastPeakIdx]*c

    if MakePlot == 1:
        fig=plt.figure(figsize=(width,height))
        grid = plt.GridSpec(1, 2, wspace=0.5, hspace=0.0)

        ax1 = fig.add_subplot(grid[0,0])
        if Modulated == 0:
            i1 = ax1.plot(N_vec,psdIGWarr[:,4], '.k', fillstyle='none', label=r'$\overline{\omega}_{IGW}$')
            i1b = ax1.plot(N_vec,psdIGWarr[:,5], 'k', marker='s', linestyle = 'None', fillstyle='none', label=r'$\omega_{IGW_E}$')
            i1c = ax1.plot(N_vec,psdIGWarr[:,1], '^k', fillstyle='none', label=r'$\omega^{\prime}_{IGW}$')
            i2 = ax1.plot(N_vec,psdIGWarr[:,2], 'ok', fillstyle='none', label=r'$\Delta \omega_{IGW}$')
        i3 = ax1.plot(N_vec,meanflowarr[:,1], 'ok', label=r'$\omega^{\prime}_{MF}$')
        #i4 = ax1.plot(N_vec0meanflowarr[:,3], '^k', fillstyle='none', label=r'$\omega_{well}$')
        ax1.set_xlabel(r'$N$ (rad/s)')
        ax1.set_ylabel(r'$\omega$ (rad/s)')
        ax1.set_xlim(0,6)
        ax1.set_ylim(0,6)

        if Modulated == 0:
            #Overplot linear models:
            # model for last peak of IGW bandwidth:
            c1 = 0.02462
            c2 = 0.98154 
            m1 = c1+c2*N_vec
            l1 = '$0.0246+0.982\,N$'
            i5 = ax1.plot(N_vec,m1,'-k',linewidth=1)

        ax1.legend(frameon=False, loc=2, labelspacing=.3)

        #ax2 = ax1.twinx()
        ax2 = fig.add_subplot(grid[0,1])
        if Modulated == 0:
            i7 = ax2.plot(N_vec,psdIGWarr[:,0], '^', c='grey', fillstyle='none', label=r'PSD($\omega^{\prime}_{IGW}$)')
            i8 = ax2.plot(N_vec,psdIGWarr[:,3], 'o', c='grey', fillstyle='none', label=r'PSD($\Delta \omega_{IGW}$)')
        i9 = ax2.plot(N_vec,meanflowarr[:,0], 'o', c='grey', label=r'PSD($\Delta \omega_{MF}$)')
        #i10 = ax2.plot(N_vec,meanflowarr[:,3], '^', c='grey', fillstyle='none', label=r'PSD($\omega_{well}$)')

        if OverlayModulated == 1:
            data = np.loadtxt('/home/ubuntu/BoussinesqLab/dedalus/meanflowarr_modulated.txt')
            #N_vec_mod = np.array((0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5))
            N_vec_mod = np.array((0.5, 1, 1.5, 2, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.5, 5))
            i10 = ax2.plot(N_vec_mod,data[:,0], 'o', c='grey', fillstyle='none', markersize=10, label=r'PSD($\Delta \omega_{MF\zeta}$)')

        #ax2.yaxis.label.set_color('grey')
        #ax2.spines['right'].set_color('grey')
        #ax2.tick_params(axis='y', colors='grey')
        ax2.set_xlabel(r'$N$ (rad/s)')
        #ax2.set_xlim(0,6)
        ax2.set_ylabel(r'PSD ([$S$]$^2$/(rad/s))')
        ax2.set_yscale('log')
        #if Modulated == 0: ax2.set_ylim(1e-4,1e+4)
        if Modulated == 0: ax2.set_ylim(1e-10,1e0)
        if Modulated == 1: ax2.set_ylim(1e-10,1e-2)
        #ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax2.legend(frameon=False)

        #Make combined legend:
        #lns = i1+i2+i3+i4+i5
        #labs = [l.get_label() for l in lns]
        #ax1.legend(lns, labs, loc=0, frameon=False, ncol=2)

        plt.show()

        #check contributions of IGW signal to mean flow using PSD data:
        #F_IGW = psdIGWarr[:,3]/meanflowarr[:,0]
        #plt.figure()
        #plt.plot(N_vec,F_IGW, 'ok')
        #plt.show() 

    if step_prediction == 1:

        omega_well = np.loadtxt('./meanflowarr.txt')[:,3]
        Nvec = np.loadtxt('./steptracking_means.txt')[0,:]
        #print(Nvec)
        Nsteps = np.loadtxt('./steptracking_means.txt')[1,:]

        fig=plt.figure(figsize=(width,height))
        grid = plt.GridSpec(1, 2, wspace=0.5, hspace=0.0)

        ax1 = fig.add_subplot(grid[0,0])
        ax1.plot(Nvec,omega_well,'o', color='k')
        ax1.set_xlabel(r'$N$ (rad/s)') 
        ax1.set_ylabel(r'$\omega_{well}$ (rad/s)') 
      
        m1 = -1.74 + 8.41*omega_well #linear model from R code.
        ax2 = fig.add_subplot(grid[0,1])
        ax2.plot(omega_well,Nsteps, 'o', c='k')
        ax2.plot(omega_well,m1, c='k')
        ax2.set_xlabel(r'$\omega_{well}$ (rad/s)') 
        ax2.set_ylabel(r'average # of steps') 

        plt.show()

    if w2f_analysis == 1:
        if Modulated == 0:
            fnm1 = './psdIGWarr.txt'
            fnm2 = './meanflowarr.txt'
            np.savetxt(fnm1,psdIGWarr)
            np.savetxt(fnm2,meanflowarr)
        if Modulated == 1: 
            fnm1 = './meanflowarr_modulated.txt'
            np.savetxt(fnm1,meanflowarr)


#Generic stastical processing:
if xMean == 1:
    data = x_mean(data)
    data = data.transpose()
    if 'data2' in locals(): 
        data2 = x_mean(data2)
        data2 = data2.transpose()
    #data has shape (Nz,Nt)
if tMean == 1:
    data = t_mean(data)
    if 'data2' in locals(): data2 = t_mean(data2)
    #data has shape (Nx,Nz)
if tMean_slide == 1:
    data = t_mean_slide(data,Nt,Nx,Nz,wing)
    if 'data2' in locals(): data2 = t_mean_slide(data2,Nt,Nx,Nz,wing)
    #data has shape (Nt,Nx,Nz)


# General section for writing analysis to a file:
if w2f_analysis == 1:
    dir_analysis = './Analysis/'
    #Create directory if it doesn't exist:
    #if not os.path.exists(dir_analysis):
    #    os.makedirs(dir_analysis)

    #fnm_analysis = dir_analysis + 'tmp.txt'
    #np.savetxt(fnm_analysis,data)


#Plotting section:
#The intention was to try and avoid repeating the bulk of the code below 
#numerous times for different variables. Repitition has certainly been 
#significantly reduced, however, it is very hard to make a fully general
#plotting program for research purposes, which are exploratory. Currently this section 
#only includes some of the time series analysis plotting.  
if MakePlot >= 1:

    if PlotStairStartEnd == 1:
        dat = np.loadtxt('./stairStartEnd.txt')
        #print(dat[0,:])
        idxN = np.where(dat[0,:]==np.sqrt(N2))
        tau0 = dat[1,int(np.array(idxN))]
        tauE = dat[2,int(np.array(idxN))]

    if PlotXZ >= 1:
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
                if 'data2' in locals(): data2 = data2[tIdx,:,:].transpose()
        if tMean == 1:
            data = data.transpose()    
            if 'data2' in locals(): data2 = data2.transpose()
 
        #Set arrays for contouring:
        xgrid = x2d
        ygrid = z2d

    if PlotTZ >= 1:
        #We define 3 options here:
        #1) consider how a single column evolves over time.
        #2) consider how the average column evolves over time.
        #3) apply a sliding time-mean to the data and then consider a single column.
        #If an average is desired then the correct switch should be used.
        if xMean != 1:
            xIdx = int(Nx/2.)
            #xIdx = int(Nx/4.)
            data = data[:,xIdx,:].transpose()
            if 'data2' in locals():
                data2 = data2[:,xIdx,:].transpose()

        #Set arrays for contouring:
        xgrid = t2d_z
        ygrid = z2d_t

    if PlotT >= 1:
       if SpectralAnalysis == 0:
           xIdx = int(Nx/2.)
           zIdx = int(Nz/2.)
           data = data[:,xIdx,zIdx]
       if 'xgrid' not in locals(): xgrid = t

    if PlotZ >= 1:
       xIdx = int(Nx/2.)
       #tIdx = 67
       tIdx = 190
       data = data[tIdx,xIdx,:]
       if 'zgrid' not in locals(): zgrid = z

    if MakeMovie != 1:
            
        fig=plt.figure(figsize=(width,height))
        if MakePlot == 1:
            grid = plt.GridSpec(1, 1, wspace=0.0, hspace=0.0)
            ax1 = fig.add_subplot(grid[0,0])
        if MakePlot == 2:
            grid = plt.GridSpec(1, 2, wspace=0.4, hspace=0.0)
            ax1 = fig.add_subplot(grid[0,0])
            ax2 = fig.add_subplot(grid[0,1])

        if PlotXZ >= 1 or PlotTZ >= 1: 
            if filledContour == 1:
                i1=ax1.contourf(xgrid,ygrid,data,clevels,cmap=cmap,extend="both")
                if 'clabel' not in locals(): clabel=''
                if NoPlotLabels == 0: fig.colorbar(i1,label=clabel)
                if PlotXZ==2 or PlotTZ==2 or (PlotXZ==1 and PlotTZ==1):
                    i2=ax2.contourf(xgrid,ygrid,data2,clevels2,cmap=cmap,extend="both")
            else:
                i1=ax1.contour(xgrid,ygrid,data,clevels,colors=colorvec,linewidths=1,linestyles='-')
                i1.clabel(clevels, inline=True, fontsize=6)
                if PlotXZ==2 or PlotTZ==2 or (PlotXZ==1 and PlotTZ==1):
                    i2=ax2.contour(xgrid,ygrid,data2,clevels2,cmap=cmap,extend="both")
            if PlotT==1:
                ax2.plot(xgrid,data.flatten(), '-', c='k', linewidth=2)
            if PlotZ==1:
                ax2.plot(data.flatten(),zgrid, '-', c='k', linewidth=2)

        if PlotXZ==0 and PlotTZ==0:
            if PlotT == 1: 
                i1=ax1.plot(xgrid,data.flatten(), '-', c='k', linewidth=2)
                if 'yscale' not in locals(): yscale='linear'
                ax1.set_yscale(yscale)
                if 'PlotGrid' not in locals(): PlotGrid=False
                ax1.grid(PlotGrid)
            if PlotZ == 1:
                i1=ax1.plot(data.flatten(),zgrid, '-', c='k', linewidth=2)

        #Plot labelling section:
        if MakePlot == 1:
            if PlotXZ == 1: 
                ax1.set_xlim(0,Lx)
                ax1.set_xlabel(r'$x$ (m)')
                ax1.set_ylim(0,Lz)
                ax1.set_ylabel(r'$z$ (m)')
                #ax1.set_title( r'$' + PlotTitle + '$' + ", " + str("%5.1f" % t[tIdx]) + " s" )
                ax1.set_title(PlotTitle)
                start, end = ax1.get_xlim()
                ax1.xaxis.set_ticks((0,0.05,0.1,0.15,0.2))
            if PlotTZ == 1:
                if 'xlim' not in locals(): xlim=(0,t[Nt-1])
                ax1.set_xlim(xlim)
                ax1.set_xlabel(r'$t$ (s)')
                ax1.set_ylim(0,Lz)
                ax1.set_ylabel(r'$z$ (m)')
                ax1.set_title(PlotTitle)
                if PlotStairStartEnd==1: 
                    ax1.plot([tau0,tau0],[0,Lz], c='silver')
                    ax1.plot([tauE,tauE],[0,Lz], c='silver')
            if PlotT == 1:
                if 'xlim' not in locals(): xlim=(np.min(xgrid),np.max(xgrid))
                if 'ylim' not in locals(): ylim=(np.min(data),np.max(data))
                ax1.set_xlim(xlim)
                ax1.set_ylim(ylim)
                if 'xlabel' not in locals(): xlabel=r'$t$ (s)'
                if 'ylabel' not in locals(): ylabel=''
                ax1.set_xlabel(xlabel)
                ax1.set_ylabel(ylabel)
                ax1.set_title(PlotTitle)
            if PlotZ == 1:
                ax1.set_title( r'$t$ =' + str("%5.1f" % t[tIdx]) + " s" )
                ax1.set_ylim(0,Lz)
                ax1.set_xlabel(r'$S$ (g/kg)')
                ax1.set_ylabel(r'$z$ (m)')
                #ax1.set_xlim(0,300)

        if 'data2' in locals():
            fig.colorbar(i2)
            ax2.set_ylim(0,Lz)
            if PlotXZ >= 1: 
                ax2.set_xlim(0,Lx)
                #ax2.set_title( PlotTitle2 + ", " + str("%5.1f" % t[tIdx]) + " s" )
                ax2.set_title(PlotTitle2)
            if PlotTZ >= 1:
                ax2.set_title(PlotTitle2)
                ax2.set_xlabel(r'$t$ (s)')

        if NoPlotLabels == 1:
            ax1.axis('off')
            
        #Make sure plot axis labels fit within plot window:
        plt.tight_layout()

        if len(sys.argv) > 2:
            tmp = RunName.split('_')
            #Pad with zeros to correctly order results:
            tmp[1:]=[str(item).zfill(3) for item in tmp[1:]]
            separator = '_'
            RunName=separator.join(tmp)

        #plt.show()
        if PlotTZ == 1: plt.savefig(FigNmBase + RunName + '_tz_' + str(nfiles) + '.png')
        if PlotZ == 1: plt.savefig(FigNmBase + RunName + '_z' + '.png')
        if PlotT==1 and SpectralAnalysis==0: plt.savefig(FigNmBase + RunName + '_t' + '.png')
        if PlotT==1 and SpectralAnalysis==1: plt.savefig(FigNmBase + RunName + '_f' + '.png')
        plt.close(fig)

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
            if PlotZ == 1:
                grid = plt.GridSpec(1, 2, wspace=0.4, hspace=0.0)
                ax1 = fig.add_subplot(grid[0,0])
            if 'data2' not in locals() and PlotZ != 1:
                grid = plt.GridSpec(1, 1, wspace=0.0, hspace=0.0)
                ax1 = fig.add_subplot(grid[0,0])

            i1=ax1.contourf(xgrid,ygrid,data[tt,:,:].transpose(),clevels,cmap=cmap,extend="both")
            fig.colorbar(i1)
            ax1.set_xlim(0,Lx)
            ax1.set_ylim(0,Lz)
            ax1.set_xlabel(r'$x$ (m)')
            ax1.set_ylabel(r'$z$ (m)')
            ax1.set_title(PlotTitle + ", " + str("%5.1f" % t[tt]) + " s")


            if PlotZ == 1:
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
                    if 'xlim' not in locals(): xlim=(0,t[Nt-1])
                    ax2.set_xlim(xlim)
                    ax2.set_xlabel(r'$t$ (s)')
                    ax2.set_title(PlotTitle2)

            FigNm = FigNmBase + '_' + str("%04d" % tt) + '.png'
            fig.savefig(FigPath+FigNm)
            plt.close(fig)
            #plt.show()

#pdb.set_trace()
