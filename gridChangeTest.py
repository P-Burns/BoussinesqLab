#Program to test effect of changing grid on simple sine wave

import numpy as np
import pdb as pdb
import matplotlib.pyplot as plt

N = 30
L = 1.
dx1 = L/N
dx2 = L/(N+2)
x1 = np.arange(N)*dx1
x2 = np.arange(N)*dx2+dx2


#f1 = np.sin(x1) + 2*np.cos(x1)
#f1 = np.sin(x1) + np.cos(x1)
#f1 = np.exp(-3*x1)*np.sin(x1)*np.cos(x1)
f1 = np.random.uniform(low=-1,high=1,size=N)

plt.plot(x1,f1,'b')
plt.plot(x2,f1,'r')
plt.show()

pdb.set_trace()
