#basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import pylab as plt
import sys
#sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf


def PlotSpkMeanVarScatter(ax, spkCntMean, spkCntVar, NE):
#    fig, ax = plt.subplots()
#    figI, axI = plt.subplots()
    ax.plot(spkCntMean[NE:], spkCntVar[NE:], 'r.', markersize = 1.0, label = 'I')
    ax.plot(spkCntMean[:NE], spkCntVar[:NE], 'k.', markersize = 1.0, label = 'E')
    ax.legend(loc = 2, frameon = False, numpoints = 1)
    ax.plot(range(80), range(80), 'g')
    ax.set_xlabel('E[Spike count]')
    ax.set_ylabel('Var[Spike count]')
    ax.set_aspect('equal')

if __name__ == '__main__':
    NE = 20000
    p = [0, 2, 5]
    ax0 = plt.subplot(1, 3, 1)
    ax1 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)
    ax = [ax0, ax1, ax3]
    for k, kp in enumerate(p):
        sc = np.load('./spkcnt/spkCnt_mean_var_p%s.npy'%(kp))
        PlotSpkMeanVarScatter(ax[k], sc[0, :], sc[1, :], NE)
    plt.show()

    

