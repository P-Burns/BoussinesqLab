#Code to plot step-tracking results


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


#Program control:
#Write data to file for statistical analysis using R:
w2f 		= 0
#Add Gusto data:
AddGusto 	= 1
#Choose statistical measure:
Mean 		= 1
Median 		= 0
PertVsN		= 1
ParkStepSize	= 0


#define useful function:
def round_down(num, divisor):
    return num - (num%divisor)


#Initialise arrays:
N_vec = [0.5, 1, 1.5, 2, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.5, 5]
name_vec = ['StateN2_00_25','StateN2_01','StateN2_02_25','StateN2_04','StateN2_06_25','StateN2_07_5625',\
          'StateN2_09','StateN2_10_5625','StateN2_12_25','StateN2_14_0625','StateN2_16','StateN2_20_25','StateN2_25']
dt2 = 0.1
t_offset_vec0 = np.array([19., 8.8, 6.7, 4.1, 2.9, 2.6, 2.4, 2.2, 1.9, 1.7, 1.6, 1.3, 1.1])

t_offset_vec = (t_offset_vec0/dt2).astype(int)
#t_offset_vec = np.arange(len(N_vec))*0
Nmins = 8
Nt = Nmins*60./dt2
Nt = int(Nt)
t = np.arange(Nt)*dt2
steps_arr = np.zeros((Nt,len(N_vec)))
steps_dz = np.zeros((Nt,50,len(N_vec)))
steps_dS = np.zeros((Nt,50,len(N_vec)))


#Read in data:
for i in range(0,len(N_vec)):    
    dir_state = './Results/' + name_vec[i] + '/TrackSteps/'
    fnm1 = dir_state + 'steps_t.txt'
    fnm2 = dir_state + 'steps_dz.txt'
    fnm3 = dir_state + 'steps_dS.txt'
    tmp1 = np.loadtxt(fnm1)
    tmp2 = np.loadtxt(fnm2)
    tmp3 = np.loadtxt(fnm3)
    idxE = len(tmp1)
    steps_arr[t_offset_vec[i]:idxE,i] = tmp1[t_offset_vec[i]:]
    steps_dz[t_offset_vec[i]:idxE,:,i] = tmp2[t_offset_vec[i]:,:]
    steps_dS[t_offset_vec[i]:idxE,:,i] = tmp3[t_offset_vec[i]:,:]


if AddGusto == 1:
    N_vec_gusto = [0.5,1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    name_vec_gusto = ['StateN2_00_25','StateN2_01','StateN2_02_25','StateN2_04','StateN2_06_25',\
                     'StateN2_09','StateN2_12_25','StateN2_16','StateN2_20_25','StateN2_25']
    dt2_g = 0.1
    t_offset_vec0_gusto = np.array([16.6,6.9,5.,3.3,2.6,2.2,1.9,1.6,1.1,0.8])

    t_offset_vec_gusto = (t_offset_vec0_gusto/dt2_g).astype(int)
    #t_offset_vec_gusto = np.arange(len(N_vec_gusto))*0
    Nmins = 8
    Nt2 = Nmins*60./dt2_g
    Nt2 = int(Nt2)
    t_g = np.arange(Nt2)*dt2_g
    steps_arr_gusto = np.zeros((Nt2,len(N_vec_gusto)))
    steps_dz_gusto = np.zeros((Nt2,50,len(N_vec_gusto)))
    steps_dS_gusto = np.zeros((Nt2,50,len(N_vec_gusto)))

    for i in range(0,len(N_vec_gusto)):
        dir_state = './Results/' + name_vec_gusto[i] + '_gusto' + '/TrackSteps/'
        fnm1 = dir_state + 'steps_t.txt'
        fnm2 = dir_state + 'steps_dz.txt'
        fnm3 = dir_state + 'steps_dS.txt'
        tmp1 = np.loadtxt(fnm1)
        tmp2 = np.loadtxt(fnm2)
        tmp3 = np.loadtxt(fnm3)
        idxE = len(tmp1)
        steps_arr_gusto[t_offset_vec_gusto[i]:idxE,i] = tmp1[t_offset_vec_gusto[i]:]
        steps_dz_gusto[t_offset_vec_gusto[i]:idxE,:,i] = tmp2[t_offset_vec_gusto[i]:,:]
        steps_dS_gusto[t_offset_vec_gusto[i]:idxE,:,i] = tmp3[t_offset_vec_gusto[i]:,:]


#Set general plotting parameters assuming A4 page size:
A4Width = 8.27
MarginWidth = 1
width = A4Width-2*MarginWidth
height = 4.5
#For scaling the A4 plot dimensions:
ScaleFactor = 1
#ScaleFactor = 0.7
width = width*ScaleFactor
height = height*ScaleFactor


#Make plots:
#Plot distributions of step counts:
fig0, axs0 = plt.subplots(len(N_vec)-5,3, figsize=(width,height), sharex='col')
fig0.subplots_adjust(wspace=0.5, hspace=0.0)
#fig0.set_tight_layout(True)

label_vec = [str(item) for item in N_vec]
bins0 = np.arange(20)+1
bins1 = (np.arange(30)+1)*0.01
#bins2 = (np.arange(3000)+1)*1
bins2 = np.logspace(0,np.log10(3000),100)
density = False
alpha=1.

#Exclude 7.5625, 10.5625, 12.25, 14.0625 and 20.25 data to make histogram panel plot more clear:
label_vec2 = label_vec[0:5] + [label_vec[6]] + [label_vec[10]] +  [label_vec[12]] 
steps_arr2 = steps_arr[:,[0,1,2,3,4,6,10,12]]
steps_dz2 = steps_dz[:,:,[0,1,2,3,4,6,10,12]]
steps_dS2 = steps_dS[:,:,[0,1,2,3,4,6,10,12]]

#set NaN points to zero:
idxsNaN = np.isnan(steps_dS2)
steps_dS2[idxsNaN] = 0

#print(len(N_vec)-5)
#pdb.set_trace()

for i in range(0,len(N_vec)-5):

    axs0[i,0].hist(steps_arr2[:,i], bins=bins0, color='k', label=label_vec2[i], alpha=alpha, density=density)
    h0,tmp = np.histogram(steps_arr2[:,i], bins=bins0, density=density)
    rounded = round_down(np.max(h0),10)
    if rounded >= 10: tickval = rounded
    else: tickval = 5
    axs0[i,0].set_yticks([tickval])
    #axs0[i,0].set_yticks([],[])
    #axs0[i,0].locator_params(which='y', tight=True, nbins=1)
    axs0[i,0].set_xlabel('# of steps')    
    axs0[i,0].legend(handlelength=0, frameon=False)

    axs0[i,1].hist(steps_dz2[:,:,i].flatten(),bins=bins1, color='k', label=label_vec2[i], alpha=alpha, density=density)
    h1,tmp = np.histogram(steps_dz2[:,:,i].flatten(), bins=bins1, density=density)
    rounded = round_down(np.max(h1),10)
    if rounded >= 10: tickval = rounded
    else: tickval = 5
    axs0[i,1].set_yticks([tickval])
    #axs0[i,1].set_yticks([],[])
    axs0[i,1].set_xlabel(r'$h_s$ (m)') 
    #axs0[i,1].legend()

    axs0[i,2].hist(np.abs(steps_dS2[:,:,i].flatten()), bins=bins2, color='k', label=label_vec2[i], alpha=alpha, density=density)
    h2,tmp = np.histogram(np.abs(steps_dS2[:,:,i].flatten()), bins=bins2, density=density)
    #print(tmp)
    rounded = round_down(np.max(h2),10)
    if rounded >= 10: tickval = rounded
    else: tickval = 5
    axs0[i,2].set_yticks([tickval])
    axs0[i,2].set_xlabel(r'$|\overline{S_z}|$ (g/kg/m)')
    #axs0[i,2].legend()
    #axs0[i,2].set_xscale("log", nonposx='clip')
    #axs0[i,2].set_xlim(1e-10,5e03)
    #axs0[i,2].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

#Plot mean of distributions:
fig1, axs1 = plt.subplots(1,3, figsize=(width,height*2/3.))
fig1.subplots_adjust(wspace=0.75, hspace=0.)
fig1.set_tight_layout(True)

data0 = steps_arr
data0[data0 == 0] = np.nan
if Mean == 1: means0 = np.nanmean(data0, axis=0)
if Median == 1: means0 = np.nanmedian(data0, axis=0)
std0 = np.nanstd(data0, axis=0)
axs1[0].plot(N_vec, means0, '.k')
axs1[0].fill_between(N_vec, means0-std0, means0 + std0, color='gray', alpha=0.4)
axs1[0].set_xlabel(r'$N$ (rad/s)')
axs1[0].set_ylabel(r'Average # of steps')
#Plot linear model (adding data points proved linear model to be invalid):
#c1 = 0.313
#c2 = 0.912
#m1 = c1 + c2*np.asarray(N_vec)
#axs1[1,0].plot(N_vec, m1,'-k')

data1 = steps_dz
data1[data1 == 0] = np.nan
if Mean == 1: means1 = np.nanmean(data1.reshape((Nt*50,len(N_vec))), axis=0)
if Median == 1: means1 = np.nanmedian(data1.reshape((Nt*50,len(N_vec))), axis=0)
std1 = np.nanstd(data1.reshape((Nt*50,len(N_vec))), axis=0)
axs1[1].plot(N_vec, means1, '.k')
axs1[1].fill_between(N_vec, means1-std1, means1 + std1, color='gray', alpha=0.4)
axs1[1].set_xlabel(r'$N$ (rad/s)')
axs1[1].set_ylabel(r'$\overline{h_s}$ (m)')

data2 = steps_dS
data2[data2 == 0] = np.nan
if Mean == 1: means2 = np.nanmean(data2.reshape((Nt*50,len(N_vec))), axis=0)
if Median == 1: means2 = np.nanmedian(data2.reshape((Nt*50,len(N_vec))), axis=0)
std2 = np.nanstd(data2.reshape((Nt*50,len(N_vec))), axis=0)
axs1[2].plot(N_vec, means2, '.k')
axs1[2].fill_between(N_vec, means2-std2, means2 + std2, color='gray', alpha=0.4)
axs1[2].set_xlabel(r'$N$ (rad/s)')
axs1[2].set_ylabel(r'$\overline{\overline{S_z}}$ (g/kg/m)')
axs1[2].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

if AddGusto == 1:
    data0_g = steps_arr_gusto
    data0_g[data0_g == 0] = np.nan
    if Mean == 1: means0_g = np.nanmean(data0_g, axis=0)
    if Median == 1: means0_g = np.nanmedian(data0_g, axis=0)
    std0_g = np.nanstd(data0_g, axis=0)
    axs1[0].plot(N_vec_gusto, means0_g, 'ok', fillstyle='none')
    axs1[0].fill_between(N_vec_gusto, means0_g-std0_g, means0_g + std0_g, color='gray', alpha=0.4)

    data1_g = steps_dz_gusto
    data1_g[data1_g == 0] = np.nan
    if Mean == 1: means1_g = np.nanmean(data1_g.reshape((Nt2*50,len(N_vec_gusto))), axis=0)
    if Median == 1: means1_g = np.nanmedian(data1_g.reshape((Nt2*50,len(N_vec_gusto))), axis=0)
    std1_g = np.nanstd(data1_g.reshape((Nt2*50,len(N_vec_gusto))), axis=0)
    axs1[1].plot(N_vec_gusto, means1_g, 'ok', fillstyle='none')
    axs1[1].fill_between(N_vec_gusto, means1_g-std1_g, means1_g + std1_g, color='gray', alpha=0.4)

    data2_g = steps_dS_gusto
    data2_g[data2_g == 0] = np.nan
    if Mean == 1: means2_g = np.nanmean(data2_g.reshape((Nt2*50,len(N_vec_gusto))), axis=0)
    if Median == 1: means2_g = np.nanmedian(data2_g.reshape((Nt2*50,len(N_vec_gusto))), axis=0)
    std2_g = np.nanstd(data2_g.reshape((Nt2*50,len(N_vec_gusto))), axis=0)
    axs1[2].plot(N_vec_gusto, means2_g, 'ok', fillstyle='none')
    axs1[2].fill_between(N_vec_gusto, means2_g-std2_g, means2_g + std2_g, color='gray', alpha=0.4)


if ParkStepSize==1:
    #choose Park runs for constant mechanical mixing: U=2.42cm/s and D=2.26cm:
    ParkStepsDz = np.array([6.2,7.6,5.5,5.3,8.7])/100.
    ParkNvec    = np.array([1.3,1.16,1.57,1.96,1.03])
    axs1[1].scatter(ParkNvec, ParkStepsDz, color='k', marker='+', s=100)
    #...not used due to too much uncertainty about units and also due to clear differences in experimental setup.

if w2f == 1:
    fnm = './steptracking_means.txt'
    np.savetxt(fnm,(N_vec,means0,means1,means2))


#Plot number of steps over time for each N:
fig2, axs2 = plt.subplots(2,1, figsize=(width,height))
fig2.subplots_adjust(wspace=0, hspace=0.4)

#(Exclude 7.5625, 10.5625, 12.25, 14.0625 and 20.25 data to make plot more clear)
t_offset_vec2 = t_offset_vec[[0,1,2,3,4,6,10,12]]
color_vec = ['k','k','grey','grey','grey','k','k','k']
lwidth_vec = (3,2,1,3,2,1,3,2,1)
line_vec = [':',':','-','-','-','-','-','-']
for i in range(0,len(N_vec)-5):

    axs2[0].plot(t[t_offset_vec2[i]:], steps_arr2[t_offset_vec2[i]:,i],\
    linestyle=line_vec[i], color=color_vec[i], linewidth=lwidth_vec[i], label=label_vec2[i])

axs2[0].set_xlabel(r'$t$ (s)')
axs2[0].set_ylabel('# of steps')
axs2[0].set_xlim((0,60))
axs2[0].legend(ncol=3, frameon=False, title=r'$N$ (rad/s)')


#Persistence plots:
#Initialise arrays for persistence data:
stairAge = np.zeros(len(N_vec))
for i in range(0,len(N_vec)):

    #Find age of staircase for each N:
    tIdxs = np.where( steps_arr[t_offset_vec[i]:,i] == 1 )
    stairAge[i] = (np.max(tIdxs) + t_offset_vec[i])*dt2

axs2[1].plot( N_vec, stairAge, 'ok-', linewidth=2, label=r'$\tau_{steps}$ (s)')
axs2[1].plot( N_vec, t_offset_vec0, 'o-', linewidth=2, color='gray', label=r'$\tau_0$ (s)')
axs2[1].set_xlabel(r'$N$ (rad/s)')
axs2[1].set_ylabel(r'$t$ (s)')
#axs2[1].set_yscale('linear')
axs2[1].set_yscale('log')
axs2[1].legend(frameon=False)

if w2f == 1:
    dat = np.vstack([N_vec,t_offset_vec0,stairAge])
    #print(dat.shape)
    np.savetxt('./stairStartEnd.txt', dat)

if AddGusto == 1:
    stairAge_g = np.zeros(len(N_vec_gusto))

    for i in range(0,len(N_vec_gusto)):
        #Find age of staircase for each N:
        tIdxs_g = np.where( steps_arr_gusto[t_offset_vec_gusto[i]:,i] == 1 )
        stairAge_g[i] = (np.max(tIdxs_g) + t_offset_vec_gusto[i])*dt2_g

    axs2[1].plot( N_vec_gusto, stairAge_g, 'ok', markersize=10, fillstyle='none', label=r'$\tau_{steps}$ (s), Gusto')
    axs2[1].plot( N_vec_gusto, t_offset_vec0_gusto, 'o', color='gray', markersize=10, fillstyle='none', label=r'$\tau_0$ (s), Gusto')
    #axs2[1].legend(frameon=False)


if PertVsN == 1:
    g = 9.81
    Lz = 0.45
    cs = 7.6*10**(-4.)

    #Typical density perturbations for Park et al lab run 13.
    #Scaled off plot:
    dgamma = 100./3
    dz_b = 2./100
    a0 = 100.
    z_a = Lz/2.
    rhoprime13 = dgamma*z_a + a0*dz_b + dgamma/2*dz_b

    #We now scale run 13 perturbations using the background 
    #density field:
    drho0_dz13 = -122.09
    N2_18 = 3.83
    drho0_dz_18 = -425.9
    rho0 = -g/N2_18*drho0_dz_18
    rhoprime = rhoprime13 * drho0_dz_18/drho0_dz13
    Spert0 = rhoprime * 1./(rho0*cs)

    #Buoyancy force of perturbations:
    ForcePert = -rhoprime/rho0*g
    #Restoring force of background field over length scale of perturbations:
    dz_pert = 1./14
    #print(dz_pert)
    #print(np.asarray(N_vec)**2)
    drho_dz_vec = -np.asarray(N_vec)**2/g*rho0/2
    #print(drho_dz_vec)
    rhoprimeN = np.abs(drho_dz_vec)*dz_pert
    ForceN = -g/rho0*rhoprimeN
    Ratio = ForcePert/ForceN

    plt.rcParams.update({'font.size': 22})
    fig=plt.figure(figsize=(width,height))
    fig.set_tight_layout(True)
    grid = plt.GridSpec(1, 1, wspace=0.0, hspace=0.0)
    ax1 = fig.add_subplot(grid[0,0])  
    ax1.semilogy(N_vec,Ratio,'o',c='k')
    ax1.plot([0,max(N_vec)],[1,1],'--k')
    ax1.set_xlabel(r'$N$ (rad/s)')
    ax1.set_ylabel(r'$F_{pert}\,/\, F_{N}$')
    plt.show()


plt.show()
