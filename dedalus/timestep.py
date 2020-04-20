import numpy as np
import matplotlib.pyplot as plt

dat = np.loadtxt('./dt.txt', delimiter=',')

print(dat.shape)
print('min dt: ', np.min(dat[1:,1]))

plt.plot(dat[:,0],dat[:,1])
plt.show()


