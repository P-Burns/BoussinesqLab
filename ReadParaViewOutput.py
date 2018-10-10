 
import numpy as np
import matplotlib.pyplot as plt
import pdb as pdb

fnm_180 = "./results/Vprofile_180.csv"
fnm_360 = "./results/Vprofile_360.csv"
data180 = np.loadtxt(fnm_180,skiprows=1,delimiter=',')
data360 = np.loadtxt(fnm_360,skiprows=1,delimiter=',')

Lz = 0.45
Nz = 180
dz = Lz/Nz

Zcoords180 = np.arange(Nz+1)*dz
Zcoords360 = np.arange(2*Nz+1)*(dz/2)

plt.plot(data180[:,0],Zcoords180,'b.-')
plt.plot(data360[:,0],Zcoords360,'r.-')
plt.show()


pdb.set_trace()
