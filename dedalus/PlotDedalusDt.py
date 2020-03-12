#Simple script to plot Dedalus results


#Import required libraries:
import h5py
import matplotlib.pyplot as plt
import pdb #to pause execution use pdb.set_trace()
import numpy as np


plt.rcParams.update({'font.size': 12})


#Read in dt data from each run:

fnm1 = 'Results/StateN2_02_25/dt.txt'
fnm2 = 'Results/StateN2_02_25_dt2/dt.txt'
fnm3 = 'Results/StateN2_02_25_dt3/dt.txt'
fnm4 = 'Results/StateN2_02_25_dt4/dt.txt'
data1 =  np.loadtxt(fnm1, delimiter=',')
data2 =  np.loadtxt(fnm2, delimiter=',')
data3 =  np.loadtxt(fnm3, delimiter=',')
data4 =  np.loadtxt(fnm4, delimiter=',')

tIdx_e1 = np.min(np.where(data1[:,0]>10.))
tIdx_e2 = np.min(np.where(data2[:,0]>10.))
tIdx_e3 = np.min(np.where(data3[:,0]>10.))
tIdx_e4 = np.min(np.where(data4[:,0]>10.))

symsize=5
plt.semilogy(data1[0:tIdx_e1,0],data1[0:tIdx_e1,1], 'ok-', markersize=symsize)
plt.semilogy(data2[0:tIdx_e2,0],data2[0:tIdx_e2,1], '.c-', markersize=symsize)
plt.semilogy(data3[0:tIdx_e3,0],data3[0:tIdx_e3,1], '.b-', markersize=symsize)
plt.semilogy(data4[0:tIdx_e4,0],data4[0:tIdx_e4,1], '.g-', markersize=symsize)
label=[r'max($\Delta t$)=1/10',r'max($\Delta t$)=1/20', r'max($\Delta t$)=1/50', r'max($\Delta t$)=1/100']
plt.legend(label)

#xlim_max = 150
#dtGustoNonLinear = 0.001
#dtGustoLinear = dtGustoNonLinear*4
#plt.semilogy([0,xlim_max],[dtGustoLinear,dtGustoLinear], 'k-', label='Gusto linear')
#plt.semilogy([0,xlim_max],[dtGustoNonLinear,dtGustoNonLinear], 'm-', label='Gusto non-linear')

#plt.xlim(-5,xlim_max)
plt.ylabel(r'$\Delta t$ (s)')
plt.xlabel('Model time (s)')

#plt.legend()

plt.show()


pdb.set_trace()
