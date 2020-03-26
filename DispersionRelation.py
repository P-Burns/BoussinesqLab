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

#Compute Dispersion Relation:
DR = np.zeros( (Nx,Nz,len(N_vec)) )
for nn in range(0,len(N_vec)):
    for ii in range(0,Nx):
        for jj in range(0,Nz):
            kvec = (k[ii],n[jj])
            DR[ii,jj,nn] = np.abs(k[ii])/np.abs(kvec)*N_vec[nn]


plt.plot(n,DR[40,:,0])
plt.plot(n,DR[40,:,2])
plt.plot(n,DR[40,:,4])
plt.plot(n,DR[40,:,6])
plt.plot(n,DR[40,:,8])
plt.plot(n,DR[40,:,10])
plt.savefig(DR.png)


