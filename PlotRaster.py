import numpy as np
import pylab as plt
fldr = "~/cuda/cudanw/"
st = np.loadtxt("/home/dhansel//cuda/cudanw/spkTimes.csv")
# plt.ion()
plt.plot(st[:, 0], st[:, 1], 'xk')
plt.draw()

# plt.figure()
# plt.hist(st[:, 0], 500)
plt.show()

counts = plt.hist(np.diff(np.sort(st[:, 0])))
print "CV = ", np.mean(counts[0]) / np.std(counts[0])
