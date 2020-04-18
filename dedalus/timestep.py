import numpy as np
import matplotlib.pyplot as plt

dat = np.loadtxt('./dt.txt', delimiter=',')

print(dat.shape)
Nt = dat.shape[0]
dt = 0.005
print(Nt,Nt*dt/60.)

out = np.diff(dat[:,0], n=1, axis=0)
print(out)

plt.plot(dat[0:Nt-1,0],out)
plt.plot([0,np.max(dat[:,0])],[dt,dt])
plt.show()


