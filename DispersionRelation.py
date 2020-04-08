#Plot Dispersion Relation for IGWs

from dedalus import public as de
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import pdb as pdb


Speeds = 0


#Physical parameters:
Lx,Lz = (0.2,0.45)
N_vec = np.array([0.5, 1, 1.5, 2, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.5, 5])

#Grid parameters:
Nx = 80
Nz = 180
x_basis = de.Fourier('x', Nx, interval=(0, Lx), dealias=3/2)
z_basis = de.SinCos('z', Nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

#Set-up wavenumbers for Dedalus grid:
k = np.zeros(Nx)
kkx = x_basis.wavenumbers
k[0:int(Nx/2.)] = kkx
dk = kkx[1]-kkx[0]
nyquist_f = -(np.max(kkx) + dk)
k[int(Nx/2.)] = nyquist_f
kkx_neg = np.flipud(kkx[1:]*(-1))
k[int(Nx/2.)+1:] = kkx_neg
n = z_basis.wavenumbers


#Compute Dispersion Relation for alpha=+1:
DR = np.zeros( (Nx,Nz,len(N_vec)) )
C = np.zeros( (Nx,Nz,len(N_vec)) )
Cgx = np.zeros( (Nx,Nz,len(N_vec)) )
Cgz = np.zeros( (Nx,Nz,len(N_vec)) )
for nn in range(0,len(N_vec)):
    for ii in range(0,Nx):
        for jj in range(0,Nz):
            if k[ii] != 0:
                kvec = np.array([k[ii],n[jj]])
                DR[ii,jj,nn] 	= np.abs(k[ii])/np.linalg.norm(kvec)*N_vec[nn]
                C[ii,jj,nn] 	= DR[ii,jj,nn]/np.linalg.norm(kvec)
                Cgx[ii,jj,nn] 	= N_vec[nn]/np.linalg.norm(kvec)*(-np.abs(k[ii])*k[ii]/np.linalg.norm(kvec)**2+1) 
                Cgz[ii,jj,nn] 	= -N_vec[nn]/np.linalg.norm(kvec)**3*np.abs(k[ii])*n[jj]


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
#Plot dispersion relation:
fig0, axs0 = plt.subplots(2,2, figsize=(width,height))
fig0.subplots_adjust(wspace=0.5, hspace=0.75)

#xvec = n
#yvec = k
xvec = np.arange(0,Nz)
yvec = np.arange(0,Nx)

kIdx1	= 2
kIdx2	= 40
Ni	= 7
vec1	= ['dotted','-','-','-','-','-','dashed']
vec2	= np.repeat(1,7)
for i in range(0,Ni):
    if i == 0:
        axs0[0,0].set_title(r'$k^{\prime}/Lx=$'+str(int(round(k[kIdx1]/(2*np.pi))))+' (1/m)')
        axs0[0,0].set_xlabel(r'$n$ (rad/m)')
        axs0[0,0].set_ylabel(r'$\Lambda_{\bf k}^{\alpha}$ (rad/s)')

    if i==0 or i==Ni-1: label=str(N_vec[i])
    else: label=''
    axs0[0,0].plot(n,DR[kIdx1,:,i*2], 'k', linestyle=vec1[i], linewidth=vec2[i], label=label)

    if i == (Ni-1):
        axs0[0,0].legend(title=r'$N$ (rad/s)')

for i in range(0,Ni):
    if i == 0:    
        axs0[0,1].set_title(r'$k^{\prime}/Lx=$'+str(int(round(k[kIdx2]/(2*np.pi))))+' (1/m)')
        axs0[0,1].set_xlabel(r'$n$ (rad/m)')
        axs0[0,1].set_ylabel(r'$\Lambda_{\bf k}^{\alpha}$ (rad/s)')

    axs0[0,1].plot(n,DR[kIdx2,:,i*2], 'k', linestyle=vec1[i], linewidth=vec2[i])

#contour plots of dispersion relation:
axs0[1,0].set_title(r'$N= $' + str(N_vec[0]) + ' (rad/s)')
c1 = axs0[1,0].contourf(xvec,yvec,DR[:,:,0], cmap='Blues', levels=20)
axs0[1,0].plot([0,Nz],[kIdx1,kIdx1],'--k')
axs0[1,0].plot([0,Nz],[kIdx2,kIdx2],'--k')
fig0.colorbar(c1,ax=axs0[1,0])

axs0[1,1].set_title(r'$N= $' + str(N_vec[12]) + ' (rad/s)')
c2 = axs0[1,1].contourf(xvec,yvec,DR[:,:,12], cmap='Blues', levels=20)
axs0[1,1].plot([0,Nz],[kIdx1,kIdx1],'--k')
axs0[1,1].plot([0,Nz],[kIdx2,kIdx2],'--k')
fig0.colorbar(c2,ax=axs0[1,1])
plt.show()


if Speeds == 1:

    axs0[1,0].set_title(r'$N= $' + str(N_vec[0]) + ' (rad/s)')
    c1 = axs0[1,0].contourf(xvec,yvec,DR[:,:,0], cmap='Blues', levels=20)
    axs0[1,0].plot([0,Nz],[kIdx1,kIdx1],'--k')
    axs0[1,0].plot([0,Nz],[kIdx2,kIdx2],'--k')
    fig0.colorbar(c1,ax=axs0[1,0])
    axs0[1,1].set_title(r'$N= $' + str(N_vec[12]) + ' (rad/s)')
    c2 = axs0[1,1].contourf(xvec,yvec,DR[:,:,12], cmap='Blues', levels=20)
    axs0[1,1].plot([0,Nz],[kIdx1,kIdx1],'--k')
    axs0[1,1].plot([0,Nz],[kIdx2,kIdx2],'--k')
    fig0.colorbar(c2,ax=axs0[1,1])
    plt.show()

    #plot phase and group velocity:
    fig1, axs1 = plt.subplots(4,2, figsize=(width,height*2))
    fig1.subplots_adjust(wspace=0.5, hspace=0.5)

    levels = [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0]

    #C
    axs1[0,0].set_title(r'$C$, $N= $' + str(N_vec[0]) + ' (m/s)')
    c1 = axs1[0,0].contourf(xvec,yvec,C[:,:,0], cmap='Blues', levels=levels, locator=ticker.LogLocator())
    axs1[0,0].plot([0,Nz],[kIdx1,kIdx1],'--k')
    fig1.colorbar(c1,ax=axs1[0,0])

    axs1[0,1].set_title(r'$C$, $N= $' + str(N_vec[12]) + ' (m/s)')
    c2 = axs1[0,1].contourf(xvec,yvec,C[:,:,12], cmap='Blues', levels=levels, locator=ticker.LogLocator())
    axs1[0,1].plot([0,Nz],[kIdx1,kIdx1],'--k')
    fig1.colorbar(c2,ax=axs1[0,1])

    #Cgx
    axs1[1,0].set_title(r'$C_{g_x}$, $N= $' + str(N_vec[0]) + ' (m/s)')
    c3 = axs1[1,0].contourf(xvec,yvec,Cgx[:,:,0], cmap='Blues', levels=levels, locator=ticker.LogLocator())
    axs1[1,0].plot([0,Nz],[kIdx1,kIdx1],'--k')
    fig1.colorbar(c3,ax=axs1[1,0])

    axs1[1,1].set_title(r'$C_{g_x}$, $N= $' + str(N_vec[12]) + ' (m/s)')
    c4 = axs1[1,1].contourf(xvec,yvec,Cgx[:,:,12], cmap='Blues', levels=levels, locator=ticker.LogLocator())
    axs1[1,1].plot([0,Nz],[kIdx1,kIdx1],'--k')
    fig1.colorbar(c4,ax=axs1[1,1])

    #Cgz
    axs1[2,0].set_title(r'$|C_{g_z}|$, $N= $' + str(N_vec[0]) + ' (m/s)')
    c5 = axs1[2,0].contourf(xvec,yvec,np.abs(Cgz[:,:,0]), cmap='Blues', levels=levels, locator=ticker.LogLocator())
    axs1[2,0].plot([0,Nz],[kIdx1,kIdx1],'--k')
    fig1.colorbar(c5,ax=axs1[2,0])

    axs1[2,1].set_title(r'$|C_{g_z}|$, $N= $' + str(N_vec[12]) + ' (m/s)')
    c6 = axs1[2,1].contourf(xvec,yvec,np.abs(Cgz[:,:,12]), cmap='Blues', levels=levels, locator=ticker.LogLocator())
    axs1[2,1].plot([0,Nz],[kIdx1,kIdx1],'--k')
    fig1.colorbar(c6,ax=axs1[2,1])

    #CgMag
    axs1[3,0].set_title(r'$|{\bf C_g}|$, $N= $' + str(N_vec[0]) + ' (m/s)')
    CgMag = np.sqrt(Cgx[:,:,0]**2 + Cgz[:,:,0]**2)
    c7 = axs1[3,0].contourf(xvec,yvec,CgMag, cmap='Blues', levels=levels, locator=ticker.LogLocator())
    axs1[3,0].plot([0,Nz],[kIdx1,kIdx1],'--k')
    fig1.colorbar(c7,ax=axs1[3,0])

    axs1[3,1].set_title(r'$|{\bf C_g}|$, $N= $' + str(N_vec[12]) + ' (m/s)')
    CgMag = np.sqrt(Cgx[:,:,12]**2 + Cgz[:,:,12]**2)
    c8 = axs1[3,1].contourf(xvec,yvec,CgMag, cmap='Blues', levels=levels, locator=ticker.LogLocator())
    axs1[3,1].plot([0,Nz],[kIdx1,kIdx1],'--k')
    fig1.colorbar(c8,ax=axs1[3,1])


plt.show()


