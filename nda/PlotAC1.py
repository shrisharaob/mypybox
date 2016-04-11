#basefolder = "/homecentral/srao/Documents/code/mypybox"
import numpy as np
import pylab as plt
import sys
#sys.path.append(basefolder + "/utils")
from Print2Pdf import Print2Pdf


def plotac(ac, ax, legendlabel, titletext):
    ax.plot(ac, label = legendlabel)
    ax.set_xlabel('Time lag(ms)')
    ax.set_ylabel('Mean activity(Hz)')
#    ax.set_title(titletext)

if __name__ == '__main__':
    plt.ioff()
    alphas = [0, 2, 5, 8]
    legendLabels = ['p = 0.%s'%(x[1]) for x in enumerate(alphas)]
    legendLabels[0] = 'control'
    print legendLabels
    filenames = ['bidirI2I_p%s'%(x[1]) for x in enumerate(alphas)]
#    filenames.insert(0, 'control')

    maxLag = 50
    figE, axE = plt.subplots()
    figI, axI = plt.subplots()
    for n, nFileName in enumerate(filenames):
        nAC = np.load('long_tau_vs_ac_mat_' + nFileName + '.npy')
        plotac(nAC[:, 0], axE, legendLabels[n], 'Population averaged AC, E')
        plotac(nAC[:, 1], axI, legendLabels[n], 'Population averaged AC, I')

    axE.set_xlim(0, maxLag)
    axI.set_xlim(0, maxLag)
    plt.figure(figE.number)
    plt.legend()
    plt.figure(figI.number)
    plt.legend()
    figFolder = '' #'/homecentral/srao/cuda/data/poster/figs/bidir/e2e/'
    filename = 'AC_E_bidirI2I' 
    Print2Pdf(figE,  figFolder + filename,  [4.6,  4.0], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.142, 0.15, .77, .74])
    filename = 'AC_I_bidirI2I' 
    Print2Pdf(figI,  figFolder + filename,  [4.6,  4.0], figFormat='png', labelFontsize = 12, tickFontsize=12, titleSize = 12.0, IF_ADJUST_POSITION = True, axPosition = [0.142, 0.15, .77, .74])
    
