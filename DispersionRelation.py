#Plot Dispersion Relation for IGWs

import numpy as np
import matplotlib.pyplot as plt
import pdb as pdb

#Physical parameters:
Lx,Lz = (0.2,0.45)
N_vec = np.array([0.5, 1, 1.5, 2, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.5, 5])

#Grid parameters:
Nx = 80
Nz = 180
k = np.arange(Nx)*2*np.pi/Lx
n = np.arange(Nz)*np.pi/Lz

print(type(k),k.shape)

#Compute Dispersion Relation:
DR = np.zeros( (Nx,Nz,len(N_vec)) )
for nn in range(0,len(N_vec)):
    for ii in range(0,Nx):
        for jj in range(0,Nz):
            kvec = np.array([k[ii],n[jj]])
            DR[ii,jj,nn] = np.abs(k[ii])/np.linalg.norm(kvec)*N_vec[nn]


kIdx=2
#kIdx=20

plt.title(r'$k^{\prime}/Lx=$'+str(int(round(k[kIdx]/(2*np.pi))))+' (1/m)')
plt.xlabel(r'$n$ (rad/m)')
plt.ylabel(r'$\Lambda_{\bf k}^{\alpha}$ (rad/s)')
plt.plot(n,DR[kIdx,:,0], label=str(N_vec[0]))
plt.plot(n,DR[kIdx,:,2])
plt.plot(n,DR[kIdx,:,4])
plt.plot(n,DR[kIdx,:,6])
plt.plot(n,DR[kIdx,:,8])
plt.plot(n,DR[kIdx,:,10])
plt.plot(n,DR[kIdx,:,12], label=str(N_vec[12]))
plt.legend(title=r'$N$ (rad/s)')
plt.show()


