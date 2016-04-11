basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import pylab as plt
import sys
sys.path.append(basefolder)
sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf

bf = '/homecentral/srao/db/data/'


ctc = np.load(bf+'cur_tuningCurves_bidirI2I_p0p0rho5_xi0.8.npy')
ctc8 = np.load(bf+'cur_tuningCurves_bidirI2I_p8p8rho5_xi0.8.npy')

fre = np.zeros((8, ))
fre8 = np.zeros((8,))
fri = np.zeros((8, ))
fri8 = np.zeros((8,))

for i in range(200):          
    idxe = np.argmax(ctc[0, i, :])
#    idxi = np.argmax(ctc[1, i, :])
    idxi = idxe
    fre = fre + np.roll(ctc[0, i, :], -1 *idxe + 4)
    fri = fri + np.roll(ctc[1, i, :], -1 * idxi + 4)
    idxe8 = np.argmax(ctc8[0, i, :])
#    idxi = np.argmax(ctc[1, i, :])
    idxi8 = idxe8
    fre8 = fre8 + np.roll(ctc8[0, i, :], -1 * idxe8 + 4)
    fri8 = fri8 + np.roll(ctc8[1, i, :], -1 * idxi8 + 4)

    # plt.plot(np.roll(ctc[1, i, :], -1 * idxi + 4))
    # plt.plot(np.roll(ctc8[1, i, :], -1 * idxi8 + 4))

thetas = np.arange(-90, 90, 22.5)
plt.subplot(2, 1, 1)

fre  = fre / 200.
fri = fri / 200.
fre8 = fre8 / 200.
fri8 = fri8 / 200.
plt.plot(thetas, fre / (fre.max()), 'ko-')
plt.plot(thetas, fre8 /(fre8.max()), 'o-')
plt.legend(['control', 'p = 0.8'], frameon = False, loc = 2, numpoints = 1)

plt.subplot(2, 1, 2)
plt.plot(thetas, fri / ( np.abs(fri.min())), 'ko-')
plt.plot(thetas, fri8 / (np.abs(fri8.min())), 'o-')

# plt.figure()
netcur = (fre + fri) / 200.
netcur8 = (fre8 + fri8) / 200.
# plt.plot(netcur)
# plt.plot(netcur8)

plt.figure()
nfre = fre / (fre.max())
nfri = fri / ( np.abs(fri.min()))
nfre8 = fre8 / (fre8.max())
nfri8 = fri8 / ( np.abs(fri8.min()))
plt.plot(nfre / (-1 * nfri))
plt.plot(nfre8 / (-1 * nfri8))

#plt.subplot(3, 1, 3)
plt.figure()
plt.plot(thetas, netcur / netcur.max(), 'ko-')
plt.plot(thetas, netcur8 / netcur8.max(), 'o-')
plt.xlabel('PO')
plt.title('Mean peak normalized net input current into I neurons')
plt.legend(['control', 'p = 0.8'], frameon = False, loc = 2, numpoints = 1)
plt.show()

