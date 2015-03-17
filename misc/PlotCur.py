import numpy as np
import pylab as plt
fldr = "~/cuda/cudanw/"
cur = np.loadtxt("/home/dhansel//cuda/cudanw/currents.csv", delimiter = ";")
dt = 0.025
t = np.arange(5 * 4000) * dt
print t.shape
print cur.shape
plt.plot(t, cur[:, 0], 'k')
plt.plot(t, cur[:, 1], 'k')
plt.plot(t, sum(cur, 1), 'k')
plt.show()
