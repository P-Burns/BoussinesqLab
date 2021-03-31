import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pdb #to pause execution use pdb.set_trace()


#Control section:
t_series = 0
panel_plot = 0
saveFig = 0



#Initialise arrays:
N_vec = np.array([.3,.5,1.,1.5,2.,2.5,2.75,3.,3.25,3.5,3.75,4.,4.5,5.])
Ns = len(N_vec)
force_vec = np.array([13,55,90,125,167,216,286]) #this vector corresponds to file names that approximate n_vec
H = 0.45
L = 0.2
k = 4.*(2*np.pi)/L
n_vec = np.array([2,8,13,18,24,31,41])*np.pi/H #central vertical wavenumber of Gaussian force
Forces = len(force_vec)
#Initialise arrays to analyse results as a function of N and the force parameter:
s1 = np.zeros((Ns,Forces))
s2 = np.zeros((Ns,Forces))
s3 = np.zeros((Ns,Forces))
s4 = np.zeros((Ns,Forces))
c1 = np.zeros((Ns,Forces))
c2 = np.zeros((Ns,Forces))
#Also create a mask array to identify/plot parameter space of no steps.
mask = np.zeros((Ns,Forces),dtype='bool')

#Look at relationship between induced force and Natural frequency 
#as cause of step instability:
d1 = np.zeros((Ns,Forces))
#Look at ratio of external force to background restoring force:
d2 = np.zeros((Ns,Forces))


#Time parameters:
dt = 0.1
Nt = int(30*60./dt)
t = np.arange(Nt)*dt
#Try offsetting start of analysis to test importance of spin-up
t_offset = 0
#t_offset = int(100./dt)
#t_offset = int(200./dt)
#t_offset = int(300./dt)

#Set general plotting parameters assuming A4 page size:
A4Width = 8.27
MarginWidth = 1
height = 8
#width = A4Width-2*MarginWidth
width = height*1
#For scaling the A4 plot dimensions:
ScaleFactor = 1
#ScaleFactor = 0.7
width = width*ScaleFactor
height = height*ScaleFactor



Nfiles = len(sys.argv)-1
#Nfiles = 1

#Main loop to read in data from results directories and save 
#computed diagnositcs in arrays for plotting/analysis:
for ff in range(0,Nfiles):

    #print(ff)

    #Construct data root directory
    RunName = sys.argv[ff+1]
    dir0 = "Results/" + RunName + "/TrackSteps/"

    #Create filename for saving results which are correctly ordered.
    #This includes removing first/last special characters and padding numbers with zeros.
    RunName = RunName[1:len(RunName)-1] 
    separator = '_'
    RunName = RunName.split(separator)
    fnm=[str(item).zfill(3) for item in RunName]
    fnm=separator.join(fnm)

    #Find N idx and force idx for contouring results:
    idxN = int(RunName[0])
    forceID = int(RunName[2])
    idxForce = int(np.where(force_vec==forceID)[0])

    #Read in data for some N and force choice:
    steps_t = np.loadtxt(dir0 + 'steps_t.txt')
    steps_dz = np.loadtxt(dir0 + 'steps_dz.txt')
    ddt_steps = np.loadtxt(dir0 + 'd_dt_step.txt')
    ddt_stepsDz = np.loadtxt(dir0 + 'd_dt_stepDz.txt')
    ddt_steps_z = np.loadtxt(dir0 + 'd_dt_step_z.txt')

    #make a copy before manipulating data:
    steps_t_cp 		= np.copy(steps_t)
    ddt_steps_cp 	= np.copy(ddt_steps)
    steps_dz_cp 	= np.copy(steps_dz)
    ddt_stepsDz_cp 	= np.copy(ddt_stepsDz)
    ddt_steps_z_cp	= np.copy(ddt_steps_z)

    #Look at when there are no steps:
    if np.min(steps_t)==0:
        idxs = np.where(steps_t==0)[0]
        #print(np.min(idxs), np.max(idxs), np.min(np.diff(idxs)), np.max(np.diff(idxs)) )
    
    #Set time points with no steps to NaN so that the averages aren't 
    #biased by varying time periods of no steps (e.g. with N). For the 2D array, steps_dz, this also 
    #removes unused columns that exist due to varying numbers of steps over time.
    tIdxsNaN = np.where(steps_t==0)[0]
    steps_t_cp[tIdxsNaN] = np.nan
    steps_dz_cp[steps_dz==0] = np.nan

    #Compute diagnostics:
    #Only consider absolute values since we are interested in step stability. For instance, this avoids
    #cancellation of rates with different signs from different steps within a staircase during averaging over steps below.

    #Average quantities over the steps for each time point.  
    #These step averages provide some kind of measure of staircase stability over time 
    #for each run (i.e. for each N and force characteristics). I have chosen to do this 
    #manually so I am sure what is happening during the averaging (and integration below):
    steps_dz_abs_ave = np.zeros((Nt))
    ddt_stepsDz_abs_ave = np.zeros((Nt))
    ddt_steps_z_abs_ave = np.zeros((Nt))

    #External force parameters:
    epsilon_t = 20 
    Force = np.sqrt(epsilon_t)

    for tt in range(0,Nt):
        if steps_t[tt] != 0:
            steps_dz_abs_ave[tt] = np.mean(np.abs(steps_dz[tt,0:int(steps_t[tt])]))
            ddt_stepsDz_abs_ave[tt] = np.mean(np.abs(ddt_stepsDz[tt,0:int(steps_t[tt])]))
            ddt_steps_z_abs_ave[tt] = np.mean(np.abs(ddt_steps_z[tt,0:int(steps_t[tt])]))
    #n.b. for times with no steps the result is zero.  This works well since these points
    #will not contribute to the integrals computed next. Setting points to NaN won't work 
    #with the integrating function trapz. Setting zero points to NaN and then taking the subset of 
    #finite points will introduce errors into the integration (think area under curve). 
 
    #Integrate rates over time to obtain stability measures for each run
    #(i.e. for some N and force parameter).  Integrating step heights and 
    #step mid height results in quantities related to change in PE.
    if np.sum(steps_t[t_offset:]) != 0:
        s1[idxN,idxForce] = np.trapz(np.abs(ddt_steps[t_offset:]),dx=dt)
        s2[idxN,idxForce] = np.trapz(ddt_stepsDz_abs_ave[t_offset:],dx=dt)
        s3[idxN,idxForce] = np.trapz(ddt_steps_z_abs_ave[t_offset:],dx=dt)
        #Consider standard deviation of step heights as a measure of instability: 
        s4[idxN,idxForce] = np.nanstd(steps_dz_cp[t_offset:,:])
        #Also look at staircase spatial characteristics like we did for unforced case:
        c1[idxN,idxForce] = np.nanmean(steps_dz_cp[t_offset:,:])
        c2[idxN,idxForce] = np.nanmean(steps_t_cp[t_offset:])
    else:
        s1[idxN,idxForce] = 0
        s2[idxN,idxForce] = 0
        s3[idxN,idxForce] = 0
        s4[idxN,idxForce] = 0
        c1[idxN,idxForce] = 0
        c2[idxN,idxForce] = 0
        mask[idxN,idxForce] = True

    #Compute ratio of induced force frequency to Natural frequency:
    k0mag = np.sqrt(k*k+n_vec[idxForce]*n_vec[idxForce])
    omegaForce = np.abs(k)/k0mag*N_vec[idxN]
    print(omegaForce/N_vec[idxN])
    if omegaForce/N_vec[idxN]==0: pdb.set_trace()
    d1[idxN,idxForce] = omegaForce/N_vec[idxN]

    #Compute ratio of external force to background restoring force:
    d2[idxN,idxForce] = Force/N_vec[idxN]/(2*np.pi)


    #Plot time series results:
    if t_series==1:    
        fig = plt.figure(figsize=(width,height))
        if panel_plot==1: grid = plt.GridSpec(4, 3, wspace=0.5, hspace=0.6)
        else: grid = plt.GridSpec(1, 1, wspace=0., hspace=0.)

        #d_dt_step:
        ax1 = fig.add_subplot(grid[0,0])
        ax1.plot(t,np.abs(ddt_steps), color='k')
        ax1.plot(t,ddt_steps, color='b', linewidth=0.5)
        ax1.set_ylim(-20,20)
        ax1.set_title( fnm + ' , ' + str(N_vec[idxN]) + ' , ' + str(round(n_vec[idxForce],2)) )
        ax1.set_xlabel(r'$t$ (s)')
        ax1.set_ylabel(r'd_dt(# of steps)')
        if panel_plot==0 and saveFig==1: 
            fig.savefig(fnm + '_d_dt_steps.png')
            plt.close(fig)

        #d_dt_stepDz:
        if panel_plot==1: 
            ax2 = fig.add_subplot(grid[0,1])
        else:
            fig = plt.figure(figsize=(width,height)) 
            grid = plt.GridSpec(1, 1, wspace=0., hspace=0.)
            ax2 = fig.add_subplot(grid[0,0])

        #Plot raw data:
        #raw data has dimensions (Nt,50)
        #print(d_dt_stepDz.shape)
        #Some of the columns are empty. Avoid this data by 
        #computing the sum of the step heights over all time for each column.
        #If the sum is zero then clearly the column was not used by the tracking algorithm.
        #I used one of the original step characteristics (step height here) since I 
        #thought it safer than using one of the derived rates of change, since rates of 
        #change could potentially (if unlikely) be zero across all time even with steps present.

        for i in range(0,len(ddt_stepsDz[0,:])):
            if np.sum(steps_dz[:,i]) != 0:
                ax2.plot(t,ddt_stepsDz[:,i], color='b', linewidth=0.5)

        ax2.plot(t,ddt_stepsDz_abs_ave, color='k')
        ax2.set_ylim(-1,1)
        ax2.set_title( fnm + ' , ' + str(N_vec[idxN]) + ' , ' + str(round(n_vec[idxForce],2)) )
        ax2.set_xlabel(r'$t$ (s)')
        ax2.set_ylabel(r'ave d_dt(step height)')
        if panel_plot==0 and saveFig==1: 
            fig.savefig(fnm + '_d_dt_stepsDz.png')
            plt.close(fig)

        #d_dt_step_z:
        if panel_plot==1: 
            ax3 = fig.add_subplot(grid[0,2])
        else:
            fig = plt.figure(figsize=(width,height))
            grid = plt.GridSpec(1, 1, wspace=0., hspace=0.) 
            ax3 = fig.add_subplot(grid[0,0])

        #Plot raw data:
        for i in range(0,len(ddt_steps_z[0,:])):
            if np.sum(steps_dz[:,i]) != 0:
                ax3.plot(t,ddt_steps_z[:,i], color='b', linewidth=0.5)

        ax3.plot(t,ddt_steps_z_abs_ave, color='k')
        ax3.set_ylim(-3,3)
        ax3.set_title( fnm + ' , ' + str(N_vec[idxN]) + ' , ' + str(round(n_vec[idxForce],2)) )
        ax3.set_xlabel(r'$t$ (s)')
        ax3.set_ylabel(r'ave d_dt(mid height)')
        if panel_plot==0 and saveFig==1: 
            fig.savefig(fnm + '_d_dt_steps_z.png')
            plt.close(fig)

    if panel_plot==1:
    #These are normalised versions of above plots:
        ax4 = fig.add_subplot(grid[1,0])
        arr1 = np.abs(ddt_steps)/np.max(np.abs(ddt_steps))
        ax4.plot(t,arr1, color='k')
        ax5 = fig.add_subplot(grid[1,1])
        arr2 = ddt_stepsDz_abs_ave/np.max(ddt_stepsDz_abs_ave)
        ax5.plot(t,arr2, color='k')
        ax6 = fig.add_subplot(grid[1,2])
        arr3 = ddt_steps_z_abs_ave/np.max(ddt_steps_z_abs_ave)
        ax6.plot(t,arr3, color='k')

        #Investigate relationships between rates using scatter plots:
        ax7 = fig.add_subplot(grid[2,0])
        ax7.scatter(arr1,arr2, s=1, c='k')
        ax7.set_xlabel(r'norm abs d_dt(# of steps)')
        ax7.set_ylabel(r'norm ave abs d_dt(step height)')

        ax8 = fig.add_subplot(grid[2,1])
        ax8.scatter(arr1,arr3, s=1, c='k')
        ax8.set_xlabel(r'norm abs d_dt(# of steps)')
        ax8.set_ylabel(r'norm ave abs d_dt(mid height)')

        ax9 = fig.add_subplot(grid[2,2])
        ax9.scatter(arr2,arr3, s=1, c='k')
        ax9.set_xlabel(r'norm ave abs d_dt(step height)')
        ax9.set_ylabel(r'norm ave abs d_dt(mid height)')

        ax10 = fig.add_subplot(grid[3,0])
        arr4 = np.abs(ddt_stepsDz)
        arr4 = arr4/np.max(arr4)
        arr5 = np.abs(ddt_steps_z)
        arr5 = arr5/np.max(arr5)
        ax10.scatter(arr4,arr5, s=1, c='k')
        ax10.set_xlabel(r'd_dt(step height)')
        ax10.set_ylabel(r'd_dt(mid height)')

        if saveFig==1:
            fig.savefig(fnm + 'panel.png')
            plt.close(fig)
        if saveFig==0: plt.show()



#Plot results 
fig = plt.figure(figsize=(width,height))
fig.set_tight_layout(True)

#Contour plots:
#grid = plt.GridSpec(1, 3, wspace=0., hspace=0.)
#L = 0.2
#k_vec = np.repeat(4*2*np.pi/L,7)
#mag_vec = np.sqrt(k_vec**2 + n_vec**2)
#omega_vec =

#ax1 = fig.add_subplot(grid[0,0])
#levels=100 
#i1 = ax1.contourf(n_vec,N_vec,s1, levels)
#plt.colorbar(i1)
#ax1.set_title(r'$s_1$')
#ax1.set_xlabel(r'$n$ (rad/m)')
#ax1.set_ylabel(r'$N$ (rad/s)')

#ax2 = fig.add_subplot(grid[0,1])
#i2 = ax2.contourf(n_vec,N_vec,s2, levels)
#plt.colorbar(i2)
#ax2.set_title(r'$s_2$')
#ax2.set_xlabel(r'$n$ (rad/m)')
#ax2.set_ylabel(r'$N$ (rad/s)')

#ax3 = fig.add_subplot(grid[0,2])
#i3 = ax3.contourf(n_vec,N_vec,s3, levels)
#plt.colorbar(i3)
#ax3.set_title(r'$s_3$')
#ax3.set_xlabel(r'$n$ (rad/m)')
#ax3.set_ylabel(r'$N$ (rad/s)')


#Plot raster images:
xMin = min(n_vec)
xMax = max(n_vec)
yMin = min(N_vec)
yMax = max(N_vec)
#cmap = 'Purples'
#cmap = 'cubehelix'
cmap = 'ocean'
#cmap = 'gnuplot'


#Find no step points:
maskIdxs = np.where(mask==1)
#print(idxs)
yvec2 = N_vec[maskIdxs[0]]
xvec2 = n_vec[maskIdxs[1]]
#print(xvec2)
#print(yvec2)


xvec = np.append(n_vec,57*np.pi/H)
yvec = np.append(N_vec,5.5)

interp = 'none'
interp = 'nearest'
rows = 3
columns = 2
meshwidth=0.1
snap=True

fig.add_subplot(rows,columns,1)
#plt.imshow(s1, extent=(xMin,xMax,yMin,yMax), interpolation=interp, cmap=cmap, aspect='auto', origin='lower')
plt.pcolormesh(xvec,yvec,s1, cmap=cmap, edgecolors='gray', linewidths=meshwidth, snap=snap)
plt.xlim(0,400)
plt.ylim(0,6)
plt.colorbar()
plt.scatter(xvec2,yvec2, marker='+', s=10, color='k')
plt.title(r'$s_1$')
#plt.xlabel(r'$n$ (rad/m)')
plt.ylabel(r'$N$ (rad/s)')

fig.add_subplot(rows,columns,2)
#plt.imshow(s2, extent=(xMin,xMax,yMin,yMax), interpolation=interp, cmap=cmap, aspect='auto', origin='lower')
plt.pcolormesh(xvec,yvec,s2, cmap=cmap, edgecolors='gray', linewidths=meshwidth, snap=snap)
plt.xlim(0,400)
plt.ylim(0,6)
plt.colorbar()
plt.scatter(xvec2,yvec2, marker='+', s=10, color='k')
plt.title(r'$s_2$')
#plt.xlabel(r'$n$ (rad/m)')
#plt.ylabel(r'$N$ (rad/s)')

fig.add_subplot(rows,columns,3)
#plt.imshow(s3, extent=(xMin,xMax,yMin,yMax), interpolation=interp, cmap=cmap, aspect='auto', origin='lower')
plt.pcolormesh(xvec,yvec,s3, cmap=cmap, edgecolors='gray', linewidths=meshwidth, snap=snap)
plt.xlim(0,400)
plt.ylim(0,6)
plt.colorbar()
plt.scatter(xvec2,yvec2, marker='+', s=10, color='k')
plt.title(r'$s_3$')
#plt.xlabel(r'$n$ (rad/m)')
plt.ylabel(r'$N$ (rad/s)')

fig.add_subplot(rows,columns,4)
#plt.imshow(s4, extent=(xMin,xMax,yMin,yMax), interpolation=interp, cmap=cmap, aspect='auto', origin='lower')
plt.pcolormesh(xvec,yvec,s4, cmap=cmap, edgecolors='gray', linewidths=meshwidth, snap=snap)
plt.xlim(0,400)
plt.ylim(0,6)
plt.colorbar()
plt.scatter(xvec2,yvec2, marker='+', s=10, color='k')
plt.title(r'$s_4$')
#plt.xlabel(r'$n$ (rad/m)')
#plt.ylabel(r'$N$ (rad/s)')

fig.add_subplot(rows,columns,5)
#plt.imshow(c1, extent=(xMin,xMax,yMin,yMax), interpolation=interp, cmap=cmap, aspect='auto', origin='lower')
plt.pcolormesh(xvec,yvec,c1, cmap=cmap, edgecolors='gray', linewidths=meshwidth, snap=snap)
plt.xlim(0,400)
plt.ylim(0,6)
plt.colorbar()
plt.scatter(xvec2,yvec2, marker='+', s=10, color='k')
plt.title(r'$c_1$')
plt.xlabel(r'$n$ (rad/m)')
plt.ylabel(r'$N$ (rad/s)')

fig.add_subplot(rows,columns,6)
#plt.imshow(c2, extent=(xMin,xMax,yMin,yMax), interpolation=interp, cmap=cmap, aspect='auto', origin='lower')
plt.pcolormesh(xvec,yvec,c2, cmap=cmap, edgecolors='gray', linewidths=meshwidth, snap=snap)
plt.xlim(0,400)
plt.ylim(0,6)
plt.colorbar()
plt.scatter(xvec2,yvec2, marker='+', s=10, color='k')
plt.title(r'$c_2$')
plt.xlabel(r'$n$ (rad/m)')
#plt.ylabel(r'$N$ (rad/s)')

plt.savefig('stability.png')
plt.close()




#Plot relationship between induced force and Natural frequency:
fig = plt.figure(figsize=(width,height))
fig.set_tight_layout(True)

interp = 'none'
rows = 2
columns = 2
meshwidth=0.1
snap=True

fig.add_subplot(rows,columns,1)
plt.pcolormesh(xvec,yvec,d1, cmap=cmap, edgecolors='gray', linewidths=meshwidth, snap=snap)
plt.xlim(0,400)
plt.ylim(0,6)
plt.colorbar()
plt.scatter(xvec2,yvec2, marker='+', s=10, color='k')
plt.title(r'$d_1$')
plt.xlabel(r'$n$ (rad/m)')
plt.ylabel(r'$N$ (rad/s)')

maskIdxs = np.where(mask==0)
fig.add_subplot(rows,columns,2)
plt.scatter(s1[maskIdxs[0],maskIdxs[1]].flatten(),d1[maskIdxs[0],maskIdxs[1]].flatten(), c='k')
plt.xlabel(r'$s_1$')
plt.ylabel(r'$d_1$')

fig.add_subplot(rows,columns,3)
plt.scatter(s2[maskIdxs[0],maskIdxs[1]].flatten(),d1[maskIdxs[0],maskIdxs[1]].flatten(), c='k')
plt.xlabel(r'$s_2$')
plt.ylabel(r'$d_1$')

fig.add_subplot(rows,columns,4)
plt.scatter(s3[maskIdxs[0],maskIdxs[1]].flatten(),d1[maskIdxs[0],maskIdxs[1]].flatten(), c='k')
plt.xlabel(r'$s_3$')
plt.ylabel(r'$d_1$')

plt.savefig('d1.png')
plt.close()


#Plot relationship between number of steps and central vertical wavenumber of force:
fig = plt.figure(figsize=(width,height))
fig.set_tight_layout(True)

narr = np.zeros((Ns,Forces))
for i in range(0,Ns): narr[i,:] = n_vec

fig.add_subplot(1,1,1)
plt.scatter(c2[maskIdxs[0],maskIdxs[1]].flatten(),narr[maskIdxs[0],maskIdxs[1]].flatten(), c='k')
plt.xlabel(r'$c_2$')
plt.ylabel(r'$n_0$')

plt.savefig('c2_vs_n0.png')
plt.close()


#Look at relationship between layer features and ratio of induced force to Natural frequency:
fig = plt.figure(figsize=(width,height))
fig.set_tight_layout(True)

interp = 'none'
rows = 2
columns = 3
meshwidth=0.1
snap=True

fig.add_subplot(rows,columns,1)
plt.pcolormesh(xvec,yvec,d2, cmap=cmap, edgecolors='gray', linewidths=meshwidth, snap=snap)
plt.xticks(n_vec[::2])
plt.xlim(0,400)
plt.ylim(0,6)
plt.colorbar()
plt.scatter(xvec2,yvec2, marker='+', s=10, color='k')
plt.title(r'$d_2$')
plt.xlabel(r'$n$ (rad/m)')
plt.ylabel(r'$N$ (rad/s)')

fig.add_subplot(rows,columns,2)
plt.plot(N_vec,Force/N_vec,'-ok')
plt.plot([np.min(N_vec),np.max(N_vec)],[1,1],'--k')
plt.xlabel(r'$N$ (rad/s)')
plt.ylabel(r'$F/N$')

maskIdxs = np.where(mask==0)
fig.add_subplot(rows,columns,4)
plt.scatter(s1[maskIdxs[0],maskIdxs[1]].flatten(),d2[maskIdxs[0],maskIdxs[1]].flatten(), c='k')
plt.xlabel(r'$s_1$')
plt.ylabel(r'$d_2$')

fig.add_subplot(rows,columns,5)
plt.scatter(s2[maskIdxs[0],maskIdxs[1]].flatten(),d2[maskIdxs[0],maskIdxs[1]].flatten(), c='k')
plt.xlabel(r'$s_2$')
plt.ylabel(r'$d_2$')

fig.add_subplot(rows,columns,6)
plt.scatter(s3[maskIdxs[0],maskIdxs[1]].flatten(),d2[maskIdxs[0],maskIdxs[1]].flatten(), c='k')
plt.xlabel(r'$s_3$')
plt.ylabel(r'$d_2$')

plt.savefig('d2.png')
plt.close()
